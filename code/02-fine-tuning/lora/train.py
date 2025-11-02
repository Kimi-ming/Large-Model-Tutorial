"""
LoRA微调训练脚本

使用LoRA方法微调CLIP模型进行图像分类
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from code.utils.model_loader import ModelLoader
from dataset import DogBreedDataset, create_dataloaders


def set_seed(seed: int):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class CLIPClassifier(nn.Module):
    """
    CLIP分类器
    
    在CLIP视觉编码器基础上添加分类头
    """
    
    def __init__(self, clip_model: CLIPModel, num_classes: int):
        super().__init__()
        self.clip_model = clip_model
        self.vision_model = clip_model.vision_model
        
        # 获取视觉编码器的输出维度
        hidden_size = self.vision_model.config.hidden_size
        
        # 添加分类头
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # 初始化分类头
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 通过CLIP视觉编码器
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        
        # 获取[CLS] token的输出
        pooled_output = vision_outputs.pooler_output
        
        # 分类
        logits = self.classifier(pooled_output)
        
        return logits


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器
        total_steps = len(train_loader) * config['training']['num_epochs']
        warmup_steps = int(total_steps * config['training']['warmup_ratio'])
        
        if config['training']['lr_scheduler']['type'] == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps
            )
        elif config['training']['lr_scheduler']['type'] == 'linear':
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(self.optimizer)
        else:
            self.scheduler = None
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # TensorBoard
        log_dir = config['output']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # 输出目录
        self.output_dir = config['output']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 训练状态
        self.global_step = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # 混合精度训练
        self.use_amp = config['hardware']['mixed_precision'] and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✅ 启用混合精度训练（FP16）")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (pixel_values, labels) in enumerate(pbar):
            pixel_values = pixel_values.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(pixel_values)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(pixel_values)
                loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config['training']['max_grad_norm'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # 梯度裁剪
                if self.config['training']['max_grad_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                
                self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
            
            # 记录日志
            if self.global_step % self.config['evaluation']['logging_steps'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/accuracy', 100.0 * correct / total, self.global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for pixel_values, labels in tqdm(self.val_loader, desc="Evaluating"):
            pixel_values = pixel_values.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            logits = self.model(pixel_values)
            loss = self.criterion(logits, labels)
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.output_dir, f'checkpoint-epoch-{epoch}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存LoRA权重
        self.model.clip_model.save_pretrained(checkpoint_dir)
        
        # 保存分类头
        torch.save(
            self.model.classifier.state_dict(),
            os.path.join(checkpoint_dir, 'classifier.pt')
        )
        
        # 保存训练状态
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'best_val_acc': self.best_val_acc,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
        }, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        print(f"✅ 检查点已保存: {checkpoint_dir}")
    
    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        num_epochs = self.config['training']['num_epochs']
        early_stopping_patience = self.config['training']['early_stopping']['patience']
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n📊 Epoch {epoch}/{num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            print(f"   训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            
            # 评估
            val_metrics = self.evaluate()
            print(f"   验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # 记录到TensorBoard
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
            
            # 保存最佳模型
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, val_metrics)
                self.patience_counter = 0
                print(f"   🎉 新的最佳验证准确率: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.config['training']['early_stopping']['enabled']:
                if self.patience_counter >= early_stopping_patience:
                    print(f"\n⚠️  早停触发！验证准确率已 {early_stopping_patience} 轮未提升")
                    break
        
        print("\n" + "=" * 60)
        print(f"✅ 训练完成！最佳验证准确率: {self.best_val_acc:.2f}%")
        print("=" * 60)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="LoRA微调训练脚本")
    parser.add_argument(
        '--config',
        type=str,
        default='code/02-fine-tuning/lora/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='数据集目录（覆盖配置文件）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='输出目录（覆盖配置文件）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设置设备
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 加载处理器
    print("\n📦 加载CLIP处理器...")
    processor = CLIPProcessor.from_pretrained(
        config['model']['name'],
        cache_dir=config['model']['cache_dir']
    )
    
    # 创建数据加载器
    print("\n📊 准备数据...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data']['data_dir'],
        processor=processor,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # 获取类别数
    num_classes = len(train_loader.dataset.classes)
    print(f"   类别数: {num_classes}")
    
    # 加载预训练模型
    print("\n🤖 加载预训练CLIP模型...")
    clip_model = CLIPModel.from_pretrained(
        config['model']['name'],
        cache_dir=config['model']['cache_dir']
    )
    
    # 配置LoRA
    print("\n⚙️  配置LoRA...")
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type="FEATURE_EXTRACTION"
    )
    
    # 应用LoRA
    clip_model.vision_model = get_peft_model(clip_model.vision_model, lora_config)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in clip_model.parameters())
    print(f"   可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # 创建分类器
    model = CLIPClassifier(clip_model, num_classes)
    model = model.to(device)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()

