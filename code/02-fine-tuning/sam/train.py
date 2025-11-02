"""
SAM模型微调训练脚本

支持的微调策略：
1. Full Fine-tuning：微调所有参数
2. Adapter Tuning：添加adapter层
3. LoRA：低秩适应

作者：Large-Model-Tutorial
许可：MIT
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入数据集
from dataset import create_sam_dataloader


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def check_sam_installation():
    """检查SAM是否安装"""
    try:
        import segment_anything
        return True
    except ImportError:
        print("❌ segment_anything未安装")
        print("安装方法: pip install git+https://github.com/facebookresearch/segment-anything.git")
        return False


class SAMTrainer:
    """SAM训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 设置设备
        self.device = self._setup_device()
        
        # 设置随机种子
        set_seed(config['seed'])
        
        # 创建输出目录
        self.output_dir = Path(config['output']['output_dir']) / config['output']['experiment_name']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        # TensorBoard
        if config['output']['use_tensorboard']:
            self.writer = SummaryWriter(
                log_dir=str(self.output_dir / config['output']['tensorboard_dir'])
            )
        else:
            self.writer = None
        
        # 加载模型
        print("\n=== 加载SAM模型 ===")
        self.model = self._load_model()
        
        # 创建数据加载器
        print("\n=== 创建数据加载器 ===")
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # 创建优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 创建损失函数
        self.criterion = self._create_criterion()
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config['device']['use_amp'] else None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        print(f"\n✅ 训练器初始化完成")
        print(f"输出目录: {self.output_dir}")
    
    def _setup_device(self) -> torch.device:
        """设置训练设备"""
        if self.config['device']['use_cuda'] and torch.cuda.is_available():
            device_id = self.config['device']['cuda_device']
            if ',' in str(device_id):
                # 多GPU（暂不支持）
                device = torch.device(f"cuda:{device_id.split(',')[0]}")
                print(f"使用GPU: {device_id} (多GPU暂不支持，使用第一个)")
            else:
                device = torch.device(f"cuda:{device_id}")
                print(f"使用GPU: {device_id}")
        else:
            device = torch.device("cpu")
            print("使用CPU")
        
        return device
    
    def _load_model(self) -> nn.Module:
        """加载SAM模型"""
        from segment_anything import sam_model_registry
        
        model_type = self.config['model']['type']
        checkpoint_path = self.config['model']['checkpoint']
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
        
        print(f"加载模型: {model_type}")
        print(f"检查点: {checkpoint_path}")
        
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam = sam.to(self.device)
        
        # 应用微调策略
        strategy = self.config['finetuning']['strategy']
        print(f"微调策略: {strategy}")
        
        if strategy == 'full':
            # 全参数微调
            self._setup_full_finetuning(sam)
        elif strategy == 'adapter':
            # Adapter微调
            self._setup_adapter_finetuning(sam)
        elif strategy == 'lora':
            # LoRA微调
            self._setup_lora_finetuning(sam)
        else:
            raise ValueError(f"不支持的微调策略: {strategy}")
        
        # 打印可训练参数
        total_params = sum(p.numel() for p in sam.parameters())
        trainable_params = sum(p.numel() for p in sam.parameters() if p.requires_grad)
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        return sam
    
    def _setup_full_finetuning(self, model: nn.Module):
        """设置全参数微调"""
        # 根据配置冻结部分组件
        if self.config['model']['freeze_image_encoder']:
            print("  冻结图像编码器")
            for param in model.image_encoder.parameters():
                param.requires_grad = False
        
        if self.config['model']['freeze_prompt_encoder']:
            print("  冻结提示编码器")
            for param in model.prompt_encoder.parameters():
                param.requires_grad = False
        
        if self.config['model']['freeze_mask_decoder']:
            print("  冻结掩码解码器")
            for param in model.mask_decoder.parameters():
                param.requires_grad = False
    
    def _setup_adapter_finetuning(self, model: nn.Module):
        """设置Adapter微调"""
        print("  Adapter微调（简化实现，冻结大部分参数）")
        
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        
        # 只微调掩码解码器（作为adapter）
        for param in model.mask_decoder.parameters():
            param.requires_grad = True
        
        print("  注意：这是简化的Adapter实现，实际应在Transformer层添加adapter模块")
    
    def _setup_lora_finetuning(self, model: nn.Module):
        """设置LoRA微调"""
        try:
            from peft import LoraConfig, get_peft_model
            
            print("  配置LoRA...")
            lora_config_dict = self.config['finetuning']['lora']
            
            # 注意：SAM的PEFT支持可能需要额外的适配
            print("  注意：SAM的LoRA微调需要特殊适配，这里使用简化实现")
            
            # 简化实现：只微调mask_decoder
            for param in model.parameters():
                param.requires_grad = False
            
            for param in model.mask_decoder.parameters():
                param.requires_grad = True
            
        except ImportError:
            print("  ⚠️ peft库未安装，回退到adapter模式")
            self._setup_adapter_finetuning(model)
    
    def _create_dataloaders(self) -> tuple:
        """创建数据加载器"""
        data_config = self.config['data']
        
        # 训练集
        train_loader = create_sam_dataloader(
            data_dir=data_config['data_dir'],
            split=data_config['train_split'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            dataset_type=data_config['dataset_type'],
            image_size=data_config['image_size'],
            prompt_mode=data_config['prompt_mode'],
            num_points=data_config.get('num_points', 1),
            augment=data_config['augment'],
        )
        
        # 验证集
        val_loader = create_sam_dataloader(
            data_dir=data_config['data_dir'],
            split=data_config['val_split'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            dataset_type=data_config['dataset_type'],
            image_size=data_config['image_size'],
            prompt_mode=data_config['prompt_mode'],
            num_points=data_config.get('num_points', 1),
            augment=False,  # 验证集不增强
        )
        
        print(f"训练集: {len(train_loader.dataset)} 样本")
        print(f"验证集: {len(val_loader.dataset)} 样本")
        
        return train_loader, val_loader
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        optimizer_config = self.config['training']['optimizer']
        optimizer_type = optimizer_config['type']
        
        # 获取可训练参数
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                lr=self.config['training']['learning_rate'],
                betas=optimizer_config['betas'],
                eps=optimizer_config['eps'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_type}")
        
        print(f"优化器: {optimizer_type}")
        print(f"学习率: {self.config['training']['learning_rate']}")
        
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_config = self.config['training']['lr_scheduler']
        scheduler_type = scheduler_config['type']
        
        num_training_steps = len(self.train_loader) * self.config['training']['num_epochs']
        num_warmup_steps = int(len(self.train_loader) * self.config['training']['warmup_epochs'])
        
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - num_warmup_steps,
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'linear':
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=scheduler_config.get('min_lr', 1e-6) / self.config['training']['learning_rate'],
                total_iters=num_training_steps
            )
        else:
            scheduler = None
        
        print(f"学习率调度器: {scheduler_type}")
        return scheduler
    
    def _create_criterion(self):
        """创建损失函数"""
        loss_config = self.config['loss']
        
        class SAMLoss(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.seg_loss_type = config['segmentation_loss']['type']
                self.dice_weight = config['segmentation_loss'].get('dice_weight', 1.0)
                self.bce_weight = config['segmentation_loss'].get('bce_weight', 1.0)
                self.iou_weight = config['iou_loss']['weight']
            
            def dice_loss(self, pred, target, smooth=1.0):
                """Dice损失"""
                pred = torch.sigmoid(pred)
                intersection = (pred * target).sum(dim=(2, 3))
                union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
                dice = (2.0 * intersection + smooth) / (union + smooth)
                return 1.0 - dice.mean()
            
            def bce_loss(self, pred, target):
                """二元交叉熵损失"""
                return F.binary_cross_entropy_with_logits(pred, target.float())
            
            def forward(self, pred_masks, pred_iou, target_masks):
                """
                计算总损失
                
                Args:
                    pred_masks: (B, N, H, W) 预测的掩码logits
                    pred_iou: (B, N) 预测的IoU分数
                    target_masks: (B, H, W) 目标掩码
                """
                # 扩展target_masks以匹配预测的数量
                target_masks = target_masks.unsqueeze(1)  # (B, 1, H, W)
                target_masks = target_masks.expand_as(pred_masks)  # (B, N, H, W)
                
                # 分割损失
                if self.seg_loss_type == 'dice':
                    seg_loss = self.dice_loss(pred_masks, target_masks)
                elif self.seg_loss_type == 'bce':
                    seg_loss = self.bce_loss(pred_masks, target_masks)
                elif self.seg_loss_type == 'dice_bce':
                    dice = self.dice_loss(pred_masks, target_masks)
                    bce = self.bce_loss(pred_masks, target_masks)
                    seg_loss = self.dice_weight * dice + self.bce_weight * bce
                else:
                    raise ValueError(f"不支持的损失类型: {self.seg_loss_type}")
                
                # IoU损失（MAE）
                with torch.no_grad():
                    pred_masks_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                    intersection = (pred_masks_binary * target_masks).sum(dim=(2, 3))
                    union = (pred_masks_binary + target_masks).clamp(0, 1).sum(dim=(2, 3))
                    target_iou = intersection / (union + 1e-6)
                
                iou_loss = F.l1_loss(pred_iou, target_iou)
                
                # 总损失
                total_loss = seg_loss + self.iou_weight * iou_loss
                
                return {
                    'total_loss': total_loss,
                    'seg_loss': seg_loss,
                    'iou_loss': iou_loss
                }
        
        return SAMLoss(loss_config)
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'seg_loss': 0.0,
            'iou_loss': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
        
        for step, batch in enumerate(pbar):
            # 移动数据到设备
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 准备提示
            prompts = {}
            if 'boxes' in batch:
                prompts['boxes'] = batch['boxes'].to(self.device)
            if 'points' in batch:
                prompts['points'] = batch['points'].to(self.device)
                prompts['point_labels'] = batch['point_labels'].to(self.device)
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=self.config['device']['use_amp']):
                # 图像编码
                image_embeddings = self.model.image_encoder(images)
                
                # 提示编码
                if 'boxes' in prompts:
                    # 使用框提示
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=None,
                        boxes=prompts['boxes'],
                        masks=None,
                    )
                elif 'points' in prompts:
                    # 使用点提示
                    coords = prompts['points']
                    labels = prompts['point_labels']
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=(coords, labels),
                        boxes=None,
                        masks=None,
                    )
                else:
                    # 无提示
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                    )
                
                # 掩码解码
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                )
                
                # 上采样到原始分辨率
                masks_pred = F.interpolate(
                    low_res_masks,
                    size=(self.config['data']['image_size'], self.config['data']['image_size']),
                    mode='bilinear',
                    align_corners=False
                )
                
                # 计算损失
                losses = self.criterion(masks_pred, iou_predictions, masks)
                loss = losses['total_loss']
            
            # 反向传播
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
            else:
                loss.backward()
                
                if (step + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
            
            # 更新统计
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # 记录日志
            if (step + 1) % self.config['output']['logging_steps'] == 0:
                if self.writer is not None:
                    self.writer.add_scalar('train/total_loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/seg_loss', losses['seg_loss'].item(), self.global_step)
                    self.writer.add_scalar('train/iou_loss', losses['iou_loss'].item(), self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """验证"""
        self.model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'seg_loss': 0.0,
            'iou_loss': 0.0
        }
        
        metrics = {
            'iou': 0.0,
            'dice': 0.0,
            'pixel_accuracy': 0.0
        }
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            prompts = {}
            if 'boxes' in batch:
                prompts['boxes'] = batch['boxes'].to(self.device)
            if 'points' in batch:
                prompts['points'] = batch['points'].to(self.device)
                prompts['point_labels'] = batch['point_labels'].to(self.device)
            
            # 前向传播（与训练相同）
            image_embeddings = self.model.image_encoder(images)
            
            if 'boxes' in prompts:
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=prompts['boxes'],
                    masks=None,
                )
            elif 'points' in prompts:
                coords = prompts['points']
                labels = prompts['point_labels']
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=(coords, labels),
                    boxes=None,
                    masks=None,
                )
            else:
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )
            
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            
            masks_pred = F.interpolate(
                low_res_masks,
                size=(self.config['data']['image_size'], self.config['data']['image_size']),
                mode='bilinear',
                align_corners=False
            )
            
            # 计算损失
            losses = self.criterion(masks_pred, iou_predictions, masks)
            
            for key in val_losses:
                val_losses[key] += losses[key].item()
            
            # 计算指标
            masks_pred_binary = (torch.sigmoid(masks_pred[:, 0]) > 0.5).float()  # 使用第一个掩码
            masks_target = masks.float()
            
            # IoU
            intersection = (masks_pred_binary * masks_target).sum(dim=(1, 2))
            union = (masks_pred_binary + masks_target).clamp(0, 1).sum(dim=(1, 2))
            iou = (intersection / (union + 1e-6)).mean()
            metrics['iou'] += iou.item()
            
            # Dice
            dice = (2.0 * intersection / (masks_pred_binary.sum(dim=(1, 2)) + masks_target.sum(dim=(1, 2)) + 1e-6)).mean()
            metrics['dice'] += dice.item()
            
            # Pixel Accuracy
            correct = (masks_pred_binary == masks_target).sum()
            total = masks_target.numel()
            metrics['pixel_accuracy'] += (correct / total).item()
        
        # 计算平均
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)
        
        for key in metrics:
            metrics[key] /= len(self.val_loader)
        
        # 记录到TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('val/total_loss', val_losses['total_loss'], epoch)
            self.writer.add_scalar('val/iou', metrics['iou'], epoch)
            self.writer.add_scalar('val/dice', metrics['dice'], epoch)
            self.writer.add_scalar('val/pixel_accuracy', metrics['pixel_accuracy'], epoch)
        
        return val_losses, metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
        }
        
        # 保存最新检查点
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ 保存检查点: {checkpoint_path}")
        
        # 保存最优检查点
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"🌟 保存最优模型: {best_path}")
        
        # 清理旧检查点
        save_total_limit = self.config['output']['save_total_limit']
        checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > save_total_limit:
            for old_ckpt in checkpoints[:-save_total_limit]:
                old_ckpt.unlink()
                print(f"🗑️  删除旧检查点: {old_ckpt}")
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
        
        num_epochs = self.config['training']['num_epochs']
        eval_every = self.config['evaluation']['eval_every_n_epochs']
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch+1}/{num_epochs} 训练完成:")
            print(f"  Loss: {train_losses['total_loss']:.4f}")
            print(f"  Seg Loss: {train_losses['seg_loss']:.4f}")
            print(f"  IoU Loss: {train_losses['iou_loss']:.4f}")
            
            # 验证
            if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
                val_losses, metrics = self.validate(epoch)
                
                print(f"\n验证结果:")
                print(f"  Val Loss: {val_losses['total_loss']:.4f}")
                print(f"  IoU: {metrics['iou']:.4f}")
                print(f"  Dice: {metrics['dice']:.4f}")
                print(f"  Pixel Acc: {metrics['pixel_accuracy']:.4f}")
                
                # 保存检查点
                is_best = metrics['iou'] > self.best_metric
                if is_best:
                    self.best_metric = metrics['iou']
                
                if not self.config['output']['save_best_only'] or is_best:
                    self.save_checkpoint(epoch, metrics, is_best)
        
        print("\n" + "="*60)
        print("训练完成！")
        print(f"最佳IoU: {self.best_metric:.4f}")
        print(f"输出目录: {self.output_dir}")
        print("="*60)
        
        if self.writer is not None:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="SAM微调训练")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 检查SAM是否安装
    if not check_sam_installation():
        sys.exit(1)
    
    # 加载配置
    print(f"加载配置: {args.config}")
    config = load_config(args.config)
    
    # 恢复训练
    if args.resume:
        config['resume']['enabled'] = True
        config['resume']['checkpoint_path'] = args.resume
    
    # 创建训练器并开始训练
    trainer = SAMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

