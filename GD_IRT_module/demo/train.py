import torch
import torch.nn as nn
import torch.optim as optim
import csv
from GDeepIRT import GDeepIRT

# ⚙️ Cấu hình
NUM_STUDENTS = 10
NUM_ITEMS = 10
EPOCHS = 50
LR = 0.01
MODEL_PATH = "model.pt"


# 📥 Load dữ liệu (thêm time_spent)
def load_data(filename="data.csv"):
    student_ids, item_ids, responses, time_spents = [], [], [], []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            student_ids.append(int(row["student_id"]))
            item_ids.append(int(row["item_id"]))
            responses.append(float(row["response"]))
            time_spents.append(int(row["time_spent"]))
    return (
        torch.tensor(student_ids),
        torch.tensor(item_ids),
        torch.tensor(time_spents),
        torch.tensor(responses),
    )


# 🚀 Huấn luyện
def train():
    model = GDeepIRT(NUM_STUDENTS, NUM_ITEMS)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()

    stu_ids, item_ids, time_spents, labels = load_data()

    for epoch in range(EPOCHS):
        model.train()
        preds = model(stu_ids, item_ids, time_spents).squeeze()
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"📊 Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")

    # 💾 Lưu mô hình
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Đã lưu model vào {MODEL_PATH}")


if __name__ == "__main__":
    train()
