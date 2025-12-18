import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import sys
import socket  # 노드 이름을 가져오기 위해 추가


# 예시 모델 정의 (간단한 MLP)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        return self.net(x)


# 학습 함수
def train(model, data_loader, optimizer, device, num_epochs):
    hostname = socket.gethostname()  # 현재 노드 이름 얻기

    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 노드 이름 포함한 로그 출력
        print(f"[{hostname}] Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader):.4f}")
        sys.stdout.flush()


# 메인 함수
def main():
    # 모델과 데이터 준비
    model = SimpleModel()

    # CUDA 사용 가능한 경우 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델을 병렬 처리로 구성
    model = nn.DataParallel(model)
    model.to(device)

    # 예시 데이터 생성 (랜덤 데이터, 실제 학습 데이터로 대체 가능)
    x = torch.randn(400000, 100)  # 데이터 포인트를 10만으로 줄임
    y = torch.randint(0, 10, (400000,))  # 10만개의 라벨 (0~9)

    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 시간 추적 시작
    start_time = time.time()

    print("Training started...")

    # 모델 학습 (에폭 수를 100으로 설정)
    train(model, data_loader, optimizer, device, num_epochs=100)

    # 학습 시간 추적 종료
    end_time = time.time()

    print(f"Training completed in {end_time - start_time:.2f} seconds")

    # 모델 저장 (선택 사항)
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")


# 실행
if __name__ == "__main__":
    main()

