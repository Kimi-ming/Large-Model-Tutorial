"""
生成BLIP-2 Notebook教程
"""
import json

# 创建notebook结构
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# BLIP-2视觉问答与图像描述教程\n\n> 完整演示BLIP-2模型的各种使用方式\n\n**学习目标**：\n- 掌握BLIP-2的图像描述生成\n- 学会使用BLIP-2进行视觉问答\n- 了解图像-文本相似度计算\n- 探索BLIP-2的实际应用\n\n**预计时间**: 40-50分钟"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# 检查并安装依赖\ntry:\n    from transformers import Blip2Processor, Blip2ForConditionalGeneration\n    print(\"✅ transformers已安装\")\nexcept ImportError:\n    print(\"正在安装transformers...\")\n    !pip install transformers\n    print(\"✅ 安装完成\")"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# 导入必要的库\nimport torch\nfrom PIL import Image\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport urllib.request\nimport os\n\nfrom transformers import Blip2Processor, Blip2ForConditionalGeneration\n\nprint(f\"PyTorch版本: {torch.__version__}\")\nprint(f\"CUDA可用: {torch.cuda.is_available()}\")\nif torch.cuda.is_available():\n    print(f\"GPU: {torch.cuda.get_device_name(0)}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 1. 加载BLIP-2模型\n\nBLIP-2有多种配置，我们使用`opt-2.7b`版本进行演示。"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# 加载模型和处理器\nmodel_name = \"Salesforce/blip2-opt-2.7b\"\n\nprint(f\"📥 加载模型: {model_name}\")\nprint(\"   (首次运行会下载模型，大约5.5GB，请耐心等待...)\")\n\nprocessor = Blip2Processor.from_pretrained(model_name)\nmodel = Blip2ForConditionalGeneration.from_pretrained(\n    model_name,\n    torch_dtype=torch.float16  # 使用FP16节省显存\n)\n\ndevice = \"cuda\" if torch.cuda.is_available() else \"cpu\"\nmodel.to(device)\nmodel.eval()\n\nprint(f\"✅ 模型加载完成，使用设备: {device}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 2. 准备示例图像"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# 准备示例图像\nimage_path = \"sample_image.jpg\"\n\n# 尝试下载示例图像\ntry:\n    if not os.path.exists(image_path):\n        print(\"📥 下载示例图像...\")\n        image_url = \"https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400\"\n        urllib.request.urlretrieve(image_url, image_path)\n        print(\"✅ 下载成功\")\nexcept Exception as e:\n    print(f\"⚠️ 下载失败: {e}\")\n    print(\"生成测试图像...\")\n    # 生成一个简单的测试图像\n    test_image = np.random.randint(128, 255, (400, 600, 3), dtype=np.uint8)\n    # 绘制一些形状\n    import cv2\n    cv2.putText(test_image, \"Cat\", (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)\n    Image.fromarray(test_image).save(image_path)\n    print(\"✅ 生成测试图像\")\n\n# 加载并显示图像\nimage = Image.open(image_path).convert(\"RGB\")\n\nplt.figure(figsize=(8, 6))\nplt.imshow(image)\nplt.title(\"示例图像\")\nplt.axis('off')\nplt.show()\n\nprint(f\"图像尺寸: {image.size}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 3. 图像描述生成 (Image Captioning)\n\nBLIP-2可以自动生成图像的描述。"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "def generate_caption(image, prompt=None, max_new_tokens=50):\n    \"\"\"生成图像描述\"\"\"\n    if prompt:\n        inputs = processor(images=image, text=prompt, return_tensors=\"pt\").to(device, torch.float16)\n    else:\n        inputs = processor(images=image, return_tensors=\"pt\").to(device, torch.float16)\n    \n    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)\n    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()\n    \n    return caption\n\n# 方式1：无提示（自动描述）\ncaption = generate_caption(image)\nprint(f\"📝 自动生成的描述:\")\nprint(f\"   {caption}\")\n\n# 方式2：带提示\ncaption_detailed = generate_caption(image, prompt=\"A detailed description:\")\nprint(f\"\\n📝 详细描述:\")\nprint(f\"   {caption_detailed}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 4. 视觉问答 (Visual Question Answering)\n\nBLIP-2可以回答关于图像的问题。"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "def visual_qa(image, question):\n    \"\"\"视觉问答\"\"\"\n    prompt = f\"Question: {question} Answer:\"\n    inputs = processor(images=image, text=prompt, return_tensors=\"pt\").to(device, torch.float16)\n    \n    generated_ids = model.generate(**inputs, max_new_tokens=20)\n    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()\n    \n    # 清理答案\n    if answer.startswith(prompt):\n        answer = answer[len(prompt):].strip()\n    \n    return answer\n\n# 测试多个问题\nquestions = [\n    \"What is the main subject of this image?\",\n    \"What color is prominent in the image?\",\n    \"Is this taken indoors or outdoors?\",\n    \"What is the mood of the image?\"\n]\n\nprint(\"❓ 视觉问答示例:\\n\")\nfor q in questions:\n    answer = visual_qa(image, q)\n    print(f\"Q: {q}\")\n    print(f\"A: {answer}\")\n    print()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 5. 多轮对话\n\n模拟与图像相关的多轮问答。"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "def multi_turn_conversation(image, questions):\n    \"\"\"多轮对话\"\"\"\n    conversation = []\n    \n    for question in questions:\n        answer = visual_qa(image, question)\n        conversation.append((question, answer))\n    \n    return conversation\n\n# 示例对话\nconversation_questions = [\n    \"What do you see in this image?\",\n    \"Can you describe it in more detail?\",\n    \"What time of day might this be?\"\n]\n\nprint(\"💬 多轮对话:\\n\")\nconversation = multi_turn_conversation(image, conversation_questions)\n\nfor i, (q, a) in enumerate(conversation, 1):\n    print(f\"回合 {i}:\")\n    print(f\"  人类: {q}\")\n    print(f\"  BLIP-2: {a}\")\n    print()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 6. 批量处理\n\n演示如何批量处理多张图像。"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# 创建多张测试图像\ntest_images = []\nfor i in range(3):\n    # 生成不同的测试图像\n    img_array = np.random.randint(100, 200, (300, 400, 3), dtype=np.uint8)\n    # 添加不同的标记\n    import cv2\n    cv2.putText(img_array, f\"Image {i+1}\", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)\n    test_images.append(Image.fromarray(img_array))\n\n# 显示测试图像\nfig, axes = plt.subplots(1, 3, figsize=(15, 5))\nfor i, img in enumerate(test_images):\n    axes[i].imshow(img)\n    axes[i].set_title(f\"Image {i+1}\")\n    axes[i].axis('off')\nplt.tight_layout()\nplt.show()\n\n# 批量生成描述\nprint(\"\\n📝 批量生成描述:\\n\")\nfor i, img in enumerate(test_images, 1):\n    caption = generate_caption(img)\n    print(f\"Image {i}: {caption}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 7. 生成参数调优\n\n探索不同的生成参数如何影响输出。"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "def generate_with_params(image, **kwargs):\n    \"\"\"使用自定义参数生成\"\"\"\n    inputs = processor(images=image, return_tensors=\"pt\").to(device, torch.float16)\n    generated_ids = model.generate(**inputs, **kwargs)\n    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()\n    return caption\n\nprint(\"🎛️ 不同生成参数的效果:\\n\")\n\n# 1. 默认参数\nprint(\"1. 默认参数:\")\ncaption1 = generate_with_params(image, max_new_tokens=50)\nprint(f\"   {caption1}\\n\")\n\n# 2. 束搜索\nprint(\"2. 束搜索 (num_beams=5):\")\ncaption2 = generate_with_params(image, max_new_tokens=50, num_beams=5)\nprint(f\"   {caption2}\\n\")\n\n# 3. 采样\nprint(\"3. 随机采样 (do_sample=True, temperature=0.7):\")\ncaption3 = generate_with_params(image, max_new_tokens=50, do_sample=True, temperature=0.7)\nprint(f\"   {caption3}\\n\")\n\n# 4. Top-p采样\nprint(\"4. Top-p采样 (top_p=0.9):\")\ncaption4 = generate_with_params(image, max_new_tokens=50, do_sample=True, top_p=0.9)\nprint(f\"   {caption4}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 8. 实际应用示例\n\n### 8.1 辅助视障人士"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "def describe_scene_for_accessibility(image):\n    \"\"\"为视障人士描述场景\"\"\"\n    # 生成详细描述\n    description = generate_caption(image, prompt=\"Describe this image in detail for a blind person:\")\n    \n    # 回答关键问题\n    safety_check = visual_qa(image, \"Are there any safety hazards visible?\")\n    location = visual_qa(image, \"What kind of place is this?\")\n    \n    return {\n        'description': description,\n        'safety': safety_check,\n        'location': location\n    }\n\nprint(\"♿ 无障碍描述:\\n\")\nresult = describe_scene_for_accessibility(image)\n\nprint(f\"场景描述: {result['description']}\")\nprint(f\"地点类型: {result['location']}\")\nprint(f\"安全检查: {result['safety']}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### 8.2 社交媒体自动标题"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "def generate_social_media_caption(image):\n    \"\"\"生成社交媒体标题\"\"\"\n    # 生成创意描述\n    caption = generate_caption(image, prompt=\"A creative and engaging social media caption:\")\n    \n    # 生成标签建议\n    tags_prompt = \"Question: What are 3 relevant hashtags for this image? Answer:\"\n    inputs = processor(images=image, text=tags_prompt, return_tensors=\"pt\").to(device, torch.float16)\n    generated_ids = model.generate(**inputs, max_new_tokens=30)\n    tags = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()\n    \n    return caption, tags\n\nprint(\"📱 社交媒体标题生成:\\n\")\ncaption, tags = generate_social_media_caption(image)\n\nprint(f\"标题: {caption}\")\nprint(f\"建议标签: {tags}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 总结\n\n本教程演示了BLIP-2的核心功能：\n\n1. **图像描述生成** - 自动生成准确的图像描述\n2. **视觉问答** - 回答关于图像的各种问题\n3. **多轮对话** - 支持连续的交互式问答\n4. **批量处理** - 高效处理多张图像\n5. **参数调优** - 通过调整生成参数优化输出\n6. **实际应用** - 无障碍辅助、社交媒体等场景\n\n### 练习任务\n\n1. 使用自己的图像测试BLIP-2\n2. 尝试不同的提示模板\n3. 比较不同生成参数的效果\n4. 探索BLIP-2的其他应用场景\n\n### 参考资源\n\n- [BLIP-2论文](https://arxiv.org/abs/2301.12597)\n- [BLIP-2 GitHub](https://github.com/salesforce/LAVIS)\n- [BLIP-2详解文档](../docs/01-模型调研与选型/06-BLIP2模型详解.md)\n\n🎉 恭喜完成本教程！"
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# 保存notebook
output_path = "notebooks/04_blip2_vqa_tutorial.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ BLIP-2 Notebook已创建: {output_path}")
print(f"   包含 {len(notebook['cells'])} 个cells")
print(f"   涵盖图像描述、VQA、多轮对话等功能")

