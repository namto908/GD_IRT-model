# gemini_interface.py
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Tự động load .env dù chạy từ đâu
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Lấy API Key
api_key = os.getenv("GOOGLE_API_KEY")
print("🔑 Đang dùng Gemini API Key:", api_key[:10], "...")  # Kiểm tra nhanh

if not api_key:
    raise RuntimeError("❌ GOOGLE_API_KEY không tìm thấy trong .env")

# Cấu hình Gemini API
genai.configure(api_key=api_key)

# Dùng model Gemini đã test hoạt động tốt
model = genai.GenerativeModel("gemini-2.0-flash")  # ✅ bạn có quyền dùng model này

def query_gemini(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text.strip()
