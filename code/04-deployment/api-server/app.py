"""
CLIP推理API服务

基于FastAPI的CLIP模型推理服务
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
import io
from typing import List, Dict, Union
import time
import os
from collections import defaultdict
from datetime import datetime, timedelta

# 尝试导入slowapi（可选依赖）
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    print("⚠️  slowapi未安装，限流功能不可用")
    print("   安装方法: pip install slowapi")
    SLOWAPI_AVAILABLE = False
    Limiter = None


class CLIPInferenceService:
    """
    CLIP推理服务（内嵌版本）
    
    支持图文匹配、图像特征提取、文本特征提取
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_fp16: bool = False
    ):
        """
        初始化推理服务
        
        Args:
            model_path: 模型路径或HuggingFace模型名称
            device: 计算设备 ("cuda", "cpu", "mps")
            use_fp16: 是否使用FP16混合精度
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        
        print(f"🚀 初始化CLIP推理服务...")
        print(f"   设备: {self.device}")
        print(f"   FP16: {self.use_fp16}")
        
        # 加载模型和处理器
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        
        # 移动到设备
        self.model = self.model.to(self.device)
        
        # 转换为FP16
        if self.use_fp16:
            self.model = self.model.half()
        
        # 设置为评估模式
        self.model.eval()
        
        print(f"✅ 模型加载完成: {model_path}")
    
    @torch.no_grad()
    def predict_image_text(
        self,
        image: Union[str, Image.Image],
        texts: List[str],
        return_probs: bool = True
    ) -> Dict:
        """
        图文匹配推理
        
        Args:
            image: 图像路径或PIL Image对象
            texts: 候选文本列表
            return_probs: 是否返回概率（否则返回logits）
            
        Returns:
            预测结果字典
        """
        start_time = time.time()
        
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("image必须是文件路径或PIL.Image对象")
        
        # 预处理
        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v 
                     for k, v in inputs.items()}
        
        # 推理
        outputs = self.model(**inputs)
        logits = outputs.logits_per_image[0]
        
        # 计算概率或返回logits
        if return_probs:
            scores = logits.softmax(dim=0)
        else:
            scores = logits
        
        # 构建结果
        results = [
            {
                "text": text,
                "score": float(score),
                "rank": idx + 1
            }
            for idx, (text, score) in enumerate(
                sorted(zip(texts, scores.cpu().numpy()), 
                      key=lambda x: x[1], reverse=True)
            )
        ]
        
        inference_time = time.time() - start_time
        
        return {
            "results": results,
            "inference_time_ms": inference_time * 1000,
            "device": str(self.device),
            "fp16": self.use_fp16
        }
    
    @torch.no_grad()
    def get_image_features(
        self,
        images: Union[List[str], List[Image.Image]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            images: 图像路径列表或PIL Image列表
            normalize: 是否归一化特征向量
            
        Returns:
            图像特征张量 (batch_size, feature_dim)
        """
        # 加载图像
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert('RGB'))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                raise ValueError("图像必须是文件路径或PIL.Image对象")
        
        # 预处理
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v 
                     for k, v in inputs.items()}
        
        # 提取特征
        features = self.model.get_image_features(**inputs)
        
        # 归一化
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu()
    
    @torch.no_grad()
    def get_text_features(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        提取文本特征
        
        Args:
            texts: 文本列表
            normalize: 是否归一化特征向量
            
        Returns:
            文本特征张量 (batch_size, feature_dim)
        """
        # 预处理
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v 
                     for k, v in inputs.items()}
        
        # 提取特征
        features = self.model.get_text_features(**inputs)
        
        # 归一化
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu()

# 配置
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}

# 创建限流器（如果可用）
if SLOWAPI_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    # 如果slowapi不可用，使用空装饰器
    class DummyLimiter:
        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    limiter = DummyLimiter()

# 创建FastAPI应用
app = FastAPI(
    title="CLIP推理服务",
    description="基于CLIP的图文匹配推理API",
    version="1.0.0"
)

# 注册限流异常处理器（如果可用）
if SLOWAPI_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型实例
model_service = None


def validate_image_file(file: UploadFile) -> None:
    """
    验证上传的图像文件
    
    Args:
        file: 上传的文件
        
    Raises:
        HTTPException: 文件验证失败
    """
    # 检查文件类型
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file.content_type}. 支持的类型: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )
    
    # 检查文件大小
    file.file.seek(0, 2)  # 移动到文件末尾
    file_size = file.file.tell()
    file.file.seek(0)  # 重置到文件开头
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大: {file_size / 1024 / 1024:.2f}MB. 最大允许: {MAX_FILE_SIZE / 1024 / 1024}MB"
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="文件为空"
        )


@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global model_service
    
    print("=" * 60)
    print("启动CLIP推理服务")
    print("=" * 60)
    
    try:
        model_service = CLIPInferenceService(
            model_path="openai/clip-vit-base-patch32",
            device="cuda",
            use_fp16=True
        )
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """关闭时清理资源"""
    print("🛑 关闭服务...")


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "CLIP推理服务",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "image_features": "/image_features",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model_service is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.post("/predict")
@limiter.limit("30/minute")  # 限制每分钟30次请求
async def predict(
    request: Request,
    image: UploadFile = File(..., description="上传的图像文件"),
    texts: str = Form(..., description="逗号分隔的候选文本")
):
    """
    图文匹配推理
    
    限流: 30次/分钟
    文件大小限制: 10MB
    支持格式: JPEG, PNG, WEBP
    
    Args:
        image: 上传的图像文件
        texts: 逗号分隔的候选文本列表
        
    Returns:
        预测结果
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 验证文件
    validate_image_file(image)
    
    try:
        # 解析文本
        text_list = [t.strip() for t in texts.split(',')]
        
        if len(text_list) == 0:
            raise HTTPException(status_code=400, detail="文本列表不能为空")
        
        # 读取图像
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # 推理
        results = model_service.predict_image_text(
            image=pil_image,
            texts=text_list
        )
        
        return JSONResponse({
            "success": True,
            "data": results
        })
    
    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@app.post("/image_features")
@limiter.limit("20/minute")  # 限制每分钟20次请求
async def extract_image_features(
    request: Request,
    image: UploadFile = File(..., description="上传的图像文件"),
    normalize: bool = Form(True, description="是否归一化特征")
):
    """
    提取图像特征
    
    限流: 20次/分钟
    文件大小限制: 10MB
    
    Args:
        image: 上传的图像文件
        normalize: 是否归一化特征向量
        
    Returns:
        图像特征向量
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 验证文件
    validate_image_file(image)
    
    try:
        # 读取图像
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # 提取特征
        features = model_service.get_image_features(
            images=[pil_image],
            normalize=normalize
        )
        
        return JSONResponse({
            "success": True,
            "data": {
                "features": features[0].tolist(),
                "shape": list(features.shape),
                "normalized": normalize
            }
        })
    
    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@app.post("/text_features")
@limiter.limit("50/minute")  # 限制每分钟50次请求（文本特征提取较快）
async def extract_text_features(
    request: Request,
    texts: str = Form(..., description="逗号分隔的文本列表"),
    normalize: bool = Form(True, description="是否归一化特征")
):
    """
    提取文本特征
    
    限流: 50次/分钟
    
    Args:
        texts: 逗号分隔的文本列表
        normalize: 是否归一化特征向量
        
    Returns:
        文本特征向量
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 解析文本
        text_list = [t.strip() for t in texts.split(',')]
        
        if len(text_list) == 0:
            raise HTTPException(status_code=400, detail="文本列表不能为空")
        
        # 提取特征
        features = model_service.get_text_features(
            texts=text_list,
            normalize=normalize
        )
        
        return JSONResponse({
            "success": True,
            "data": {
                "features": features.tolist(),
                "shape": list(features.shape),
                "normalized": normalize,
                "texts": text_list
            }
        })
    
    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

