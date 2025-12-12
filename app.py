
import torch
from diffusers import AutoPipelineForText2Image
import gradio as gr

# Load model gốc + LoRA đã train
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")


pipe.load_lora_weights("lora_viet_turbo_super")  # Folder lưu từ train.py

pipe.enable_xformers_memory_efficient_attention()

def generate(prompt):
    image = pipe(
        prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=768,
        width=768
    ).images[0]
    return image

demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Prompt tiếng Việt", placeholder="cô gái mặc áo dài cánh đồng lúa"),
    outputs=gr.Image(),
    title="SDXL-Turbo Việt Nam – Model Tự Train",
    examples=[
        "cô gái mặc áo dài đứng giữa cánh đồng lúa vàng",
        "mèo đội nón lá ăn phở Hà Nội",
        "Hội An về đêm đèn lồng"
    ]
)

demo.launch(share=True)
