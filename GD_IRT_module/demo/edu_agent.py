import torch
from GDeepIRT import GDeepIRT
from gemini_interface import query_gemini
from utils import recommend, diagnose
from behavior import analyze_behavior


class EduAgent:
    def __init__(self):
        self.model = GDeepIRT(10, 10)  # ví dụ
        try:
            self.model.load_state_dict(torch.load("model.pt"))
            print("✅ Loaded model")
        except:
            print("⚠️ Chưa có model, dùng model trống")
        self.model.eval()

    def respond(self, student_id, item_id, response, time_spent):
        s_id = torch.tensor([student_id])
        i_id = torch.tensor([item_id])
        time_spent = torch.tensor([time_spent])
        prob = self.model(s_id, i_id, time_spent).item()

        rec = recommend(prob)
        diag = diagnose([response])
        behav = analyze_behavior(time_spent)

        prompt = f"""Học sinh làm bài {item_id}, xác suất đúng là {prob:.2f}.
    Lỗi: {diag}
    Hành vi: {behav}
    Đề xuất học: {rec}
    Hãy viết phản hồi một cách cục súc, nhưng đầy đủ thông tin."""

        reply = query_gemini(prompt)
        return reply
