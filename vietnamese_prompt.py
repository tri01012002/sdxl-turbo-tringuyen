# vietnamese_prompt.py
# File này giúp ảnh có chất Việt Nam rõ rệt (áo dài bay, nón lá, ánh sáng vàng cam, phong cách phim Việt…)

VIETNAM_BOOST = [
    ", highly detailed, cinematic lighting, golden hour, made by Vietnam filmmaker",
    ", beautiful Vietnamese girl, traditional áo dài, flowing in the wind, 8k, masterpiece",
    ", Hanoi old quarter, motorbikes, street food, warm sunlight, realistic",
    ", Sapa rice terraces, mist, morning light, national geographic style",
    ", Hội An ancient town, lanterns, night scene, warm tones, dreamy",
    ", Phở bò Hà Nội, steam rising, delicious food photography",
    ", conical hat (nón lá), Vietnamese culture, sharp details",
    ", mekong delta, boat on river, coconut trees, lush green",
    ", halong bay emerald water, limestone mountains, dramatic sky",
    ", Vietnamese coffee dripping, morning vibe, cozy cafe"
]

def enhance_prompt(prompt_vn: str) -> str:
    """
    Nhận prompt tiếng Việt → trả về prompt tiếng Anh cực mạnh + chất Việt
    """
    # Danh sách từ khóa Việt → thêm boost tương ứng
    boost = ""
    prompt_lower = prompt_vn.lower()

    if any(x in prompt_lower for x in ["áo dài", "aodai", "áo"]):
        boost += VIETNAM_BOOST[1]
    if any(x in prompt_lower for x in ["hà nội", "hanoi", "phố cổ"]):
        boost += VIETNAM_BOOST[2]
    if any(x in prompt_lower for x in ["sapa", "ruộng bậc thang"]):
        boost += VIETNAM_BOOST[3]
    if any(x in prompt_lower for x in ["hội an", "hoi an"]):
        boost += VIETNAM_BOOST[4]
    if any(x in prompt_lower for x in ["phở", "pho"]):
        boost += VIETNAM_BOOST[5]
    if any(x in prompt_lower for x in ["nón lá", "non la"]):
        boost += VIETNAM_BOOST[6]
    if any(x in prompt_lower for x in ["cà phê", "coffee"]):
        boost += VIETNAM_BOOST[9]

    # Dịch thô sang tiếng Anh (bạn có thể thay bằng Google Translate API sau)
    translations = {
        "cô gái": "beautiful young Vietnamese woman",
        "con mèo": "cute cat",
        "con chó": "cute dog",
        "siêu anh hùng": "superhero",
        "hoàng hôn": "golden hour sunset",
        "biển": "beach, turquoise water",
        "núi": "mountain, dramatic landscape"
    }

    prompt_en = prompt_vn
    for vi, en in translations.items():
        prompt_en = prompt_en.replace(vi, en)

    final_prompt = f"{prompt_en}, best quality, ultra detailed, 8k, sharp focus{boost}"
    return final_prompt
