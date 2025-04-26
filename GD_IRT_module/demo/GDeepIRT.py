import torch
import torch.nn as nn
import torch.optim as optim
import csv

# 🔧 Mô hình lõi GDeepIRT
class GDeepIRT(nn.Module):
    def __init__(self, student_dim, item_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.student_embed = nn.Embedding(student_dim, hidden_dim)
        self.item_embed = nn.Embedding(item_dim, hidden_dim)
        self.time_embed = nn.Embedding(10, hidden_dim // 2)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, 128),  # = 64*2 + 32 = 160
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )



    def forward(self, s_ids, i_ids, time_spent=None):
        s = self.student_embed(s_ids)
        q = self.item_embed(i_ids.clamp(min=0))  # xử lý missing item

        if time_spent is not None:
            binned = torch.clamp(time_spent // 60, 0, 9)
            t = self.time_embed(binned)
            x = torch.cat([s, q, t], dim=-1)
        else:
            x = torch.cat([s, q], dim=-1)

        return self.out(x)


# ✅ Focal loss chống mất cân bằng
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# 📥 Load dữ liệu từ CSV
def load_data(filename="data.csv"):
    student_ids, item_ids, responses, time_spents = [], [], [], []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            student_ids.append(int(row["student_id"]))
            item_ids.append(int(row["item_id"]) if row["item_id"].isdigit() else -1)
            responses.append(float(row["response"]))
            time_spents.append(int(row["time_spent"]))
    return (
        torch.tensor(student_ids),
        torch.tensor(item_ids),
        torch.tensor(time_spents),
        torch.tensor(responses),
    )


# 🚀 Huấn luyện model
def train(
    student_dim=10,
    item_dim=10,
    epochs=50,
    lr=0.01,
    model_path="model.pt",
    use_focal_loss=True,
    l2_lambda=1e-4
):
    model = GDeepIRT(student_dim, item_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = FocalLoss() if use_focal_loss else nn.BCELoss()

    stu_ids, item_ids, time_spents, labels = load_data()

    for epoch in range(epochs):
        model.train()
        preds = model(stu_ids, item_ids, time_spents).squeeze()
        loss = loss_fn(preds, labels)

        # ➕ L2 regularization
        l2 = torch.norm(model.student_embed.weight) + torch.norm(model.item_embed.weight)
        loss += l2_lambda * l2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"📊 Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ Đã lưu model vào {model_path}")


if __name__ == "__main__":
    train()
