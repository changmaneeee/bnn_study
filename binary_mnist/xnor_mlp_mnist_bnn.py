import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, transform=transforms.ToTensor()),
    batch_size=1024)
class BinActive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        return a.sign()
    
    @staticmethod
    def backward(ctx, g):
        return g
    
def binarize_activation(a):
        return BinActive.apply(a)
    

class BinWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_real): #가중치를 위한 이진화화
        alpha = w_real.abs().mean(dim=list(range(1,w_real.ndim)), keepdim=True)#w_real은 실제 실수형 가중치치
                                                                               #각 출력채널별로 alpha를 계산
        if w_real.ndim > 1:
             alpha = w_real.abs().mean(dim=1, keepdim=True)
        
        else:
            alpha = w_real.abs().mean()
        
        w_bin = w_real.sign()
        ctx.save_for_backward(w_real)
        return alpha * w_bin
    
    @staticmethod
    def backward(ctx, g_alpha_b):
        w_real = ctx.saved_tensors[0]
    
        return g_alpha_b
    
def binarize_weight_xnor(w_real):
     return BinWeight.apply(w_real)


#신경망 정의 / XNOR-Net 스타일 MLP

class XNOR_MLP(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.fc1_features =1024 # 은닉층의 뉴런 수 
        self.fc1 = nn.Linear(28 * 28, self.fc1_features, bias=False)
        self.bn1 = nn.BatchNorm1d(self.fc1_features)
        # BatchNorm1d는 1차원 데이터에 대한 배치 정규화

        self.fc2 = nn.Linear(self.fc1_features, num_classes, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = binarize_activation(self.fc1(x))  # Binarize activation
        x = self.fc2(binarize_weight_xnor(x))  # Binarize weights
        return F.log_softmax(x, dim=1)
    
    def forward(self,x):

        x = x.view(x.size(0), -1)  # 입력 평탄화
        real_w_fc1 = self.fc1.weight  # 실제 가중치
        bin_w_fc1_xnor = binarize_weight_xnor(real_w_fc1)  # XNOR 이진화 가중치
        x = F.linear(x, bin_w_fc1_xnor)

        x = self.bn1(x)  # 배치 정규화
        x = binarize_activation(x)  # 활성화 함수 이진화

        real_w_fc2 = self.fc2.weight  # 두 번째 층의 실제 가중치
        bin_w_fc2_xnor = binarize_weight_xnor(real_w_fc2)
        x = F.linear(x, bin_w_fc2_xnor)


        return x
    
# ───────────────────────────────────────────────────────────────

model = XNOR_MLP().to(device)  # 모델 인스턴스 생성 및 장치로 이동
opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam 옵티마이저 설정
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)  # 학습률 스케줄러 설정

num_epochs = 20  # 학습 에폭 수
best_test_acc = 0.0  # 최상의 테스트 정확도 초기화

# ───────────────────────────────────────────────────────────────

for epoch in range(num_epochs):
    # ---------- 학습 단계 ----------
    model.train()
    train_correct = 0
    train_loss_sum = 0

    loop_train = tqdm(train_loader, desc=f'E{epoch+1:02d}/{num_epochs} Train | LR: {opt.param_groups[0]["lr"]:.1e}')

    for data, target in loop_train:
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        opt.step()

        train_loss_sum += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        loop_train.set_postfix(loss=loss.item()) # tqdm에 현재 loss 표시

    train_acc = train_correct / len(train_loader.dataset) * 100
    avg_train_loss = train_loss_sum / len(train_loader)
    print(f'Epoch {epoch+1:02d}/{num_epochs} ▶ Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')

    # ---------- 검증 단계 ----------
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)  # 평균 손실
    test_acc = test_correct / len(test_loader.dataset) * 100
    print(f'                └ Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n')
    # 학습률 스케줄러 업데이트
    scheduler.step()
    # 최상의 테스트 정확도 갱신

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        print(f'New best test accuracy: {best_test_acc:.2f}%\n')
    else:
        print(f'No improvement in test accuracy. Best so far: {best_test_acc:.2f}%\n')

print(f"\n최종 목표: MNIST Test Accuracy ≥ 96 %")
print(f"달성한 최고 Test Accuracy: {best_test_acc:.2f}%")
# ───────────────────────────────────────────────────────────────

    