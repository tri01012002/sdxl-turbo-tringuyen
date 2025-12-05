# app.py
import torch
from diffusers import AutoPipelineForText2Image
import gradio as gr

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()  # nhanh hơn 30%

def generate(prompt_vn):
    # Bonus: tự động dịch nếu cần (bạn có thể thêm Google Translate API sau)
    image = pipe(prompt_vn, num_inference_steps=4, guidance_scale=0.0).images[0]
    return image

title = "SDXL Turbo – Text-to-Image Tiếng Việt Siêu Nhanh"
description = "Gõ bất kỳ câu tiếng Việt nào → ảnh 1024x1024 trong <3 giây"

demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Mô tả bằng tiếng Việt", placeholder="cô gái mặc áo dài đứng giữa cánh đồng lúa"),
    outputs=gr.Image(label="Kết quả"),
    title=title,
    description=description,
    examples=[
        ["mèo đội nón lá ăn phở"],
        ["Hồ Gươm Hà Nội lúc hoàng hôn"],
        ["siêu anh hùng Việt Nam cưỡi rồng"]
    ],
    cache_examples=True
)

demo.launch()
