import google.generativeai as genai

# 🔑 Dán API KEY LẤY TỪ MAKERSUITE (https://makersuite.google.com/app/apikey)
API_KEY = ""  # ⬅️ Dán key của bạn vào đây

# ✅ Cấu hình Gemini
genai.configure(api_key=API_KEY)

# 🧠 Tạo model chat
model = genai.GenerativeModel("gemini-2.0-flash")

# 💬 Vòng lặp chatbot đơn giản
print("🤖 Chatbot Gemini Flash – Nhập 'exit' để thoát")
while True:
    user_input = input("👤 Bạn: ")
    if user_input.strip().lower() == "exit":
        break

    try:
        response = model.generate_content(user_input)
        print("🤖 Gemini:", response.text.strip())
    except Exception as e:
        print("❌ Lỗi:", e)
