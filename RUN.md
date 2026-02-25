# Startup

```bash
cd sdxl_dog_editor
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7000
```

# Example request

```bash
curl -X POST http://localhost:7000/generate \
  -F "file=@/path/to/dog.jpg" \
  -F "prompt=Change the dog's clothes into a hoodie"
```

Response:
```json
{"seed": 1234567890, "image_path": "/abs/path/...", "image_url": "/outputs/out_1234567890_1734567890.png"}
```

# Frontend

API 주소 `http://210.91.154.131:20443/95ce287337c3ad9f` 로 요청하는 프론트엔드:

```bash
# 프론트엔드만 로컬에서 열기 (파일 열기)
open sdxl_dog_editor/frontend/index.html

# 또는 간이 서버로 제공 (CORS 없이 같은 도메인처럼 쓰려면)
cd sdxl_dog_editor/frontend && npx serve -l 3000
# 브라우저: http://localhost:3000
```

프론트엔드에서 이미지 업로드 + 프롬프트 입력 후 생성하면, 해당 API로 POST `/generate` 요청을 보내고 결과 이미지를 표시한다.
