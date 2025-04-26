def recommend(theta):
    if theta < 0.4:
        return "Nên ôn lại kiến thức cơ bản."
    elif theta < 0.8:
        return "Tiếp tục luyện bài trung bình."
    else:
        return "Sẵn sàng thử bài nâng cao."

def diagnose(response_history):
    if response_history.count(0) > 0:
        return "Thường sai ở câu khó hoặc chưa nắm vững."
    return "Làm bài ổn định."
