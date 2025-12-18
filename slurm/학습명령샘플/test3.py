import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import sys
import socket


# 더 복잡한 Residual MLP 모델 정의
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return self.relu(out + residual)


class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.input_layer = nn.Linear(100, 256)
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        self.output_layer = nn.Linear(256, 10)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)


def train(model, data_loader, optimizer, device, num_epochs):
    hostname = socket.gethostname()

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[{hostname}] Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader):.4f}")
        sys.stdout.flush()


def main():
    model = ComplexModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(model)
    model.to(device)

    x = torch.randn(400000, 100)
    y = torch.randint(0, 10, (400000,))
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    print("Training started...")

    train(model, data_loader, optimizer, device, num_epochs=100)

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    torch.save(model.state_dict(), "complex_model.pth")
    print("Model saved as complex_model.pth")


if __name__ == "__main__":
    main()

