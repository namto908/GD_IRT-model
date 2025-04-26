import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Giả sử data.csv đã có các cột: student_id, item_id, response, time_spent
df = pd.read_csv("databehavior.csv")

# ⚙️ Sinh nhãn hành vi (dạng rule trước, dùng để train)
def label_behavior(row):
    t = row["time_spent"]
    if t < 10:
        return "guessing"
    elif t > 90:
        return "hesitating"
    else:
        return "focused"

df["behavior"] = df.apply(label_behavior, axis=1)

# 🧠 Huấn luyện mô hình đơn giản
X = df[["time_spent", "response"]]
y = df["behavior"]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y_enc)

# 💾 Lưu model và encoder
joblib.dump(model, "behavior_model.pkl")
joblib.dump(le, "behavior_label_encoder.pkl")
print("✅ Đã huấn luyện xong mô hình hành vi")
