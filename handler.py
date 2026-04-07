import os
import torch
import runpod
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# [핵심] 런팟의 영구 보관소 경로입니다.
# 나중에 내 컴퓨터에 있는 파일들을 이 위치로 옮길 겁니다.
RUNPOD_VOLUME_PATH = "/runpod-volume/clip_model"

def load_model():
    print(f"--- 모델 로드 시작 (경로: {RUNPOD_VOLUME_PATH}) ---")
    
    # 인터넷 연결을 완전히 차단하고 로컬 파일만 사용하도록 설정
    try:
        model = CLIPModel.from_pretrained(
            RUNPOD_VOLUME_PATH, 
            local_files_only=True # 인터넷 접속 금지 설정
        ).to(device)
        
        processor = CLIPProcessor.from_pretrained(
            RUNPOD_VOLUME_PATH,
            local_files_only=True
        )
        print("--- 로컬 모델 로드 성공! ---")
        return model, processor
    except Exception as e:
        print(f"--- 로컬 로드 실패: {e} ---")
        print("팁: 아직 /runpod-volume/clip_model에 파일이 업로드되지 않은 것 같습니다.")
        raise e

model, processor = load_model()

def handler(job):
    try:
        image_url = job["input"].get("image_url")
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        return {"embedding": image_features.cpu().numpy().tolist()[0]}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})