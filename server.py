import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from predict import analyze_content  # Gọi hàm dự đoán từ file predict.py của bạn

app = FastAPI()

# Cho phép trình duyệt gửi dữ liệu xuống server không bị chặn CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 1. Mở giao diện index.html khi vào link http://127.0.0.1:8000
@app.get("/")
async def serve_ui():
    return FileResponse("brand_safety_moderation_guard_dashboard.html")

# 2. Nhận video người dùng tải lên từ HTML và đưa vào predict.py xử lý
@app.post("/api/moderate")
async def moderate_video(
    video: UploadFile = File(None),
    caption: str = Form("")
):
    video_path = None
    if video and video.filename:
        video_path = os.path.join(UPLOAD_DIR, video.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

    # Chạy hàm analyze_content trong predict.py và trả Output về cho HTML
    result = analyze_content(video_path=video_path, caption=caption)
    return result