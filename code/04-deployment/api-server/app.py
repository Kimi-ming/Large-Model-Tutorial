"""
CLIP推理API服务

基于FastAPI的CLIP模型推理服务
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
import io
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from code.deployment.nvidia.basic.pytorch_inference import CLIPInferenceService

# 创建FastAPI应用
app = FastAPI(
    title="CLIP推理服务",
    description="基于CLIP的图文匹配推理API",
    version="1.0.0"
)

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
async def predict(
    image: UploadFile = File(..., description="上传的图像文件"),
    texts: str = Form(..., description="逗号分隔的候选文本")
):
    """
    图文匹配推理
    
    Args:
        image: 上传的图像文件
        texts: 逗号分隔的候选文本列表
        
    Returns:
        预测结果
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
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
async def extract_image_features(
    image: UploadFile = File(..., description="上传的图像文件"),
    normalize: bool = Form(True, description="是否归一化特征")
):
    """
    提取图像特征
    
    Args:
        image: 上传的图像文件
        normalize: 是否归一化特征向量
        
    Returns:
        图像特征向量
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
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
async def extract_text_features(
    texts: str = Form(..., description="逗号分隔的文本列表"),
    normalize: bool = Form(True, description="是否归一化特征")
):
    """
    提取文本特征
    
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

