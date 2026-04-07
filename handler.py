import runpod
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel

# 1. 모델 로드 (서버 시작 시 한 번만 실행됨)
device = "cuda" if torch.cuda.is_available() else "cpu"

# local_files_only=True 추가 및 .to(device)로 메모리에 올리기
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)

def handler(job):
    """
    main.py에서 보낸 {'images': [url1, url2, ...]} 데이터를 처리합니다.
    """
    job_input = job['input']
    image_urls = job_input.get("images", [])
    
    if not image_urls:
        return {"error": "No image URLs provided"}

    results = []
    
    for url in image_urls:
        try:
            # 2. S3 URL(Presigned)에서 이미지 다운로드
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # 3. CLIP 모델로 임베딩 추출
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            
            # 4. 텐서를 리스트로 변환 (JSON 응답용)
            embedding = image_features.cpu().numpy().tolist()[0]
            results.append(embedding)
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
            results.append(None) 

    return results

# 런팟 서버리스 시작
runpod.serverless.start({"handler": handler})