def analyze_behavior(time_spent):
    if time_spent < 15:
        return "Làm bài quá nhanh, có thể chưa đọc kỹ đề."
    elif time_spent > 180:
        return "Làm bài quá lâu, có thể gặp khó khăn."
    else:
        return "Thời gian làm bài hợp lý."
