import streamlit as st
from edu_agent import EduAgent

agent = EduAgent()

st.set_page_config(page_title="AI Giáo Viên", layout="centered")
st.title("🎓 AI Giáo Viên – Gemini-2.0-flash+ Deep-IRT")

with st.form("input_form"):
    student_id = st.number_input("ID học sinh", 0, 100)
    item_id = st.number_input("ID câu hỏi", 0, 100)
    response = st.selectbox("Kết quả", [0, 1])
    time_spent = st.slider("Thời gian làm bài (giây)", 0, 600, 60)
    submit = st.form_submit_button("Gửi")

if submit:
    st.info("⏳ Đang xử lý...")
    reply = agent.respond(int(student_id), int(item_id), int(response), int(time_spent))
    st.success(reply)

    # ⬇️ Ghi log dữ liệu vào CSV
    import csv
    with open("data.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([int(student_id), int(item_id), int(response), int(time_spent)])
    st.toast("✅ Dữ liệu đã được lưu để huấn luyện tiếp theo", icon="💾")
