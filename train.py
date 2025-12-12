

# pip install diffusers transformers accelerate peft datasets torch icrawler deep-translator pillow --upgrade
# pip install bitsandbytes  
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from accelerate import Accelerator
from transformers import CLIPTokenizer
from icrawler.builtin import GoogleImageCrawler
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
from PIL import Image
import os
import random

# Crawl ảnh 
os.makedirs("viet_dataset", exist_ok=True)

keywords = [
    "áo dài Việt Nam", "cánh đồng lúa Việt Nam", "nón lá Việt Nam", "phở Hà Nội", "Hội An đèn lồng", "Hạ Long vịnh", "Sapa ruộng bậc thang", "cà phê sữa đá Sài Gòn", "cô gái Việt Nam xinh đẹp", "bãi biển Phú Quốc", "chợ Bến Thành", "lễ hội Tết Nguyên Đán Việt Nam", "tranh Đông Hồ", "nhà sàn Tây Nguyên", "chùa Một Cột Hà Nội"
]

total_crawled = 0
for kw in keywords:
    print(f"Crawl: {kw} (100 ảnh)")
    crawler = GoogleImageCrawler(storage={'root_dir': 'viet_dataset'})
    crawler.crawl(keyword=kw, max_num=100, filters={'size': 'large'})

total_images = len([f for f in os.listdir("viet_dataset") if f.endswith(('.jpg', '.png', '.jpeg'))])
print(f"Tổng ảnh crawl: {total_images}")

# Sinh Caption Tiếng Việt Cho Ảnh (dùng BLIP + Dịch)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

def generate_viet_caption(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=50)
    en_cap = processor.decode(out[0], skip_special_tokens=True)
    vi_cap = GoogleTranslator(source='en', target='vi').translate(en_cap)
    return vi_cap

for file in os.listdir("viet_dataset"):
    if file.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join("viet_dataset", file)
        caption = generate_viet_caption(img_path)
        txt_path = img_path.rsplit('.', 1)[0] + '.txt'
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption)
        print(f"Caption cho {file}: {caption}")

# Tạo Dataset Từ Folder
image_files = [f for f in os.listdir("viet_dataset") if f.endswith(('.jpg', '.png', '.jpeg'))]
dataset = Dataset.from_dict({
    "image": [os.path.join("viet_dataset", f) for f in image_files],
    "text": [open(os.path.join("viet_dataset", f.rsplit('.', 1)[0] + '.txt'), "r", encoding="utf-8").read().strip() for f in image_files]
})

#  Fine-Tune LoRA Trên GPU batch 16, số steps = dataset/batch size
accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=4)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None
).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

lora_config = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.1,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"]
)
pipe.unet = get_peft_model(pipe.unet, lora_config)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

pipe.unet, dataloader = accelerator.prepare(pipe.unet, dataloader)

optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)

pipe.unet.train()
for epoch in range(5):  # 5 epochs 
    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(pipe.unet):
            # Preprocess image
            images = [Image.open(img_path).convert("RGB").resize((512, 512)) for img_path in batch["image"]]
            pixel_values = pipe.image_processor(images, return_tensors="pt").pixel_values.to(device, torch.float16)
            latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor

            # Text
            text_inputs = pipe.tokenizer(batch["text"], padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)

            # Noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Predict
            encoder_hidden_states = pipe.text_encoder(text_inputs)[0]
            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Loss
            loss = F.mse_loss(noise_pred, noise, reduction="mean")

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        if step % 50 == 0:
            print(f"Epoch {epoch}, Step {step} | Loss: {loss.item():.4f}")

pipe.save_pretrained("lora_viet_turbo_super")

# Model lưu tại lora_viet_turbo_super – dùng trong app.py để test.
