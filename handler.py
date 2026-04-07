import os
import torch
import runpod
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# [핵심] 모델을 저장할 절대 경로 (런팟 볼륨 주소)
MODEL_NAME = "openai/clip-vit-base-patch32"
SAVE_PATH = "/runpod-volume/clip_model"

def load_model():
    # 저장된 폴더가 없으면 새로 다운로드
    if not os.path.exists(SAVE_PATH):
        print(f"--- 모델이 없습니다. 최초 1회 다운로드를 시작합니다: {MODEL_NAME} ---")
        temp_model = CLIPModel.from_pretrained(MODEL_NAME)
        temp_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        
        # 지정된 경로에 모델 저장
        temp_model.save_pretrained(SAVE_PATH)
        temp_processor.save_pretrained(SAVE_PATH)
        print(f"--- 모델 저장 완료: {SAVE_PATH} ---")
    
    # 저장된 경로(로컬)에서 모델 불러오기 (인터넷 안 씀)
    print("--- 로컬 볼륨에서 모델을 로드합니다 ---")
    model = CLIPModel.from_pretrained(SAVE_PATH).to(device)
    processor = CLIPProcessor.from_pretrained(SAVE_PATH)
    return model, processor

# 모델 로드 (전역 변수)
model, processor = load_model()

def handler(job):
    """
    런팟 서버리스 호출 시 실행되는 메인 함수
    """
    job_input = job["input"]
    image_url = job_input.get("image_url")
    
    if not image_url:
        return {"error": "image_url is required"}

    try:
        # 이미지 다운로드 및 처리
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # 결과를 리스트로 변환하여 반환
        embedding = image_features.cpu().numpy().tolist()[0]
        return {"embedding": embedding}

    except Exception as e:
        return {"error": str(e)}

# 런팟 서버리스 시작
runpod.serverless.start({"handler": handler})