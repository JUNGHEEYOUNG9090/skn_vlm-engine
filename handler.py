import os
import torch
import runpod
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# [핵심] 우리가 직접 만들고 관리할 모델 창고 주소
SAVE_PATH = "/runpod-volume/clip_model"
MODEL_ID = "openai/clip-vit-base-patch32"

def load_model():
    # 1. 만약 지정한 폴더에 모델이 없으면? 여기서 직접 다운로드합니다.
    if not os.path.exists(SAVE_PATH):
        print(f"--- 모델이 없어서 직접 다운로드합니다: {MODEL_ID} ---")
        # 이때는 인터넷 연결이 필요하므로 local_files_only를 쓰지 않습니다.
        model = CLIPModel.from_pretrained(MODEL_ID)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        
        # 다운로드 완료 후 우리만의 창고(SAVE_PATH)에 저장합니다.
        model.save_pretrained(SAVE_PATH)
        processor.save_pretrained(SAVE_PATH)
        print(f"--- 모델 저장 완료: {SAVE_PATH} ---")
    
    # 2. 이제 폴더가 확실히 있으니, 거기서 로컬로 불러옵니다.
    print(f"--- 로컬 경로({SAVE_PATH})에서 모델을 로드합니다 ---")
    model = CLIPModel.from_pretrained(SAVE_PATH).to(device)
    processor = CLIPProcessor.from_pretrained(SAVE_PATH)
    return model, processor

# 모델 로드 (최초 실행 시 다운로드 혹은 로드 수행)
model, processor = load_model()

def handler(job):
    try:
        job_input = job["input"]
        image_url = job_input.get("image_url")
        
        if not image_url:
            return {"error": "image_url이 필요합니다."}

        # 이미지 처리
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        embedding = image_features.cpu().numpy().tolist()[0]
        return {"embedding": embedding}

    except Exception as e:
        return {"error": str(e)}

# 런팟 서버리스 시작
runpod.serverless.start({"handler": handler})