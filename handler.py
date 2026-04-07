import os
import torch
import runpod
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# [가장 중요] 모델을 저장할 우리만의 확실한 창고 주소
SAVE_PATH = "/runpod-volume/clip_model"
MODEL_ID = "openai/clip-vit-base-patch32"

def load_model():
    print(f"--- 모델 로드 프로세스 시작 ---")
    
    # 1. 창고에 모델이 있는지 확인
    if not os.path.exists(SAVE_PATH):
        print(f"--- [최초 1회] 창고가 비어있습니다. 모델 다운로드를 시작합니다... ---")
        # 폴더 생성
        os.makedirs(SAVE_PATH, exist_ok=True)
        
        # 인터넷에서 모델 가져오기
        model = CLIPModel.from_pretrained(MODEL_ID)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        
        # 가져온 모델을 우리 창고(SAVE_PATH)에 영구 저장
        model.save_pretrained(SAVE_PATH)
        processor.save_pretrained(SAVE_PATH)
        print(f"--- 다운로드 및 창고 저장 완료! (위치: {SAVE_PATH}) ---")
    else:
        print(f"--- 창고에 이미 모델이 있습니다. 바로 꺼내서 쓰겠습니다. ---")

    # 2. 이제 창고(로컬)에 있는 모델을 읽어서 GPU에 올림
    print("--- GPU로 모델 로딩 중... ---")
    model = CLIPModel.from_pretrained(SAVE_PATH).to(device)
    processor = CLIPProcessor.from_pretrained(SAVE_PATH)
    return model, processor

# 서버 시작 전 모델 로드 실행
model, processor = load_model()

def handler(job):
    try:
        job_input = job["input"]
        image_url = job_input.get("image_url")
        
        if not image_url:
            return {"error": "image_url이 필요합니다."}

        # 이미지 다운로드 및 전처리
        response = requests.get(image_url, timeout=15)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # 모델 추론
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # 결과 반환
        embedding = image_features.cpu().numpy().tolist()[0]
        return {"embedding": embedding}

    except Exception as e:
        return {"error": f"실행 중 에러 발생: {str(e)}"}

# 런팟 서버리스 엔진 시작
runpod.serverless.start({"handler": handler})