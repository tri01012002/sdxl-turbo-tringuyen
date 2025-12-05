# app.py
# SDXL-Turbo Text-to-Image Tiếng Việt siêu nhanh (< 3 giây)
# Repo: https://github.com/ten-cua-ban/sdxl-turbo-vietnamese

import torch
from diffusers import AutoPipelineForText2Image
import gradio as gr
from vietnamese_prompt import enhance_prompt   # ← import ở đây

# ====================== LOAD MODEL ======================
print("Đang tải SDXL-Turbo... (lần đầu sẽ mất 1-3 phút)")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,          # tắt checker để nhanh hơn (chỉ dùng cho demo cá nhân)
    requires_safety_checker=False
)
pipe.to("cuda")

# Tối ưu tốc độ (rất quan trọng!)
pipe.enable_xformers_memory_efficient_attention()   # giảm VRAM + tăng tốc ~30%
pipe.enable_attention_slicing()                      # nếu VRAM nhỏ (T4 16GB vẫn ok)

# ====================== HÀM SINH ẢNH ======================
def generate(prompt_vn: str):
    """
    Nhận prompt tiếng Việt → trả về ảnh đẹp chuẩn Việt Nam
    """
    # Bước 1: Tăng chất lượng prompt bằng file vietnamese_prompt.py
    full_prompt = enhance_prompt(prompt_vn.strip())

    # Bước 2: Sinh ảnh – SDXL-Turbo chỉ cần 1–4 steps
    image = pipe(
        prompt=full_prompt,
        height=1024,
        width=1024,
        num_inference_steps=4,      # Turbo: 1–4 là đủ
        guidance_scale=0.0,         # Turbo hoạt động tốt nhất khi = 0.0
        num_images_per_prompt=1
    ).images[0]

    return image

# ====================== GIAO DIỆN GRADIO ======================
title = "SDXL Turbo – Text-to-Image Tiếng Việt Siêu Nhanh"
description = """
Gõ bất kỳ câu tiếng Việt nào → ảnh 1024×1024 trong **dưới 3 giây**  
Model: stabilityai/sdxl-turbo | GPU: A4000 / T4
"""

examples = [
    ["cô gái Việt Nam mặc áo dài đứng giữa cánh đồng lúa vàng"],
    ["mèo đội nón lá ăn phở Hà Nội"],
    ["Hội An về đêm lung linh đèn lồng"],
    ["siêu anh hùng Việt Nam cưỡi rồng phun lửa"],
    ["cà phê sữa đá Sài Gòn buổi sáng"],
    ["vịnh Hạ Long bình minh sương mù"],
    ["cậu bé chăn trâu trên cánh đồng quê"]
]

demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(
        label="Mô tả bằng tiếng Việt",
        placeholder="Ví dụ: cô gái mặc áo dài đứng giữa cánh đồng lúa",
        lines=3
    ),
    outputs=gr.Image(label="Kết quả (1024×1024)", type="pil"),
    title=title,
    description=description,
    examples=examples,
    cache_examples=True,           # tự động lưu ảnh ví dụ → load nhanh hơn
    allow_flagging="never"
)

# ====================== CHẠY ======================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # cần khi deploy RunPod / Railway
        server_port=7860
    )
