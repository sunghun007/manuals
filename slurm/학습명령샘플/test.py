import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import sys

# 간단한 신경망 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, data_loader, optimizer, device, num_epochs=100):
    model.train()
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

        # 학습 상태 출력 (에폭마다)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader):.4f}")
        sys.stdout.flush()

def main():
    # 모델과 데이터 준비
    model = SimpleModel()
    
    # CUDA 사용 가능한 경우 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델을 병렬 처리로 구성
    model = nn.DataParallel(model)
    model.to(device)
    
    # 예시 데이터 생성 (랜덤 데이터, 실제 학습 데이터로 대체 가능)
    x = torch.randn(100000, 100)  # 데이터 포인트를 10만으로 늘림
    y = torch.randint(0, 10, (100000,))  # 10만개의 라벨 (0~9)
    
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)  # 배치 사이즈를 64로 설정
    
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

if __name__ == "__main__":
    main()

