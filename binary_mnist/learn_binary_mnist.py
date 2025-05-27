# ───────────────────────────────────────────────────────────────
# 0. 라이브러리 임포트
# ───────────────────────────────────────────────────────────────
import torch                                # PyTorch 핵심 패키지 (텐서 + 자동미분)
import torch.nn as nn                       # torch.nn : 레이어(층) 클래스 모음
import torch.nn.functional as F             # torch.nn.functional : 레이어가 아닌 함수형 API
from torchvision import datasets, transforms# 이미지 데이터셋/전처리 툴 모음
from tqdm import tqdm                       # tqdm : for 문 돌 때 진행률 바를 보여줌

# ───────────────────────────────────────────────────────────────
# 1. 학습에 사용할 ‘장치(Device)’ 설정
#    - GPU(엔비디아 카드)가 있으면 속도가 훨씬 빠르다.
#    - 없으면 CPU도 가능. 단지 시간이 오래 걸릴 뿐.
# torch.cuda.is_available() → CUDA 사용 가능 여부 True / False
# ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────────────────────────
# 2. 데이터 준비 : MNIST 손글씨 (28×28 흑백, 0~9 숫자 10종)
#    DataLoader : (데이터셋, 배치크기, 셔플여부)를 묶어
#                 for 루프로 돌릴 수 있게 해 주는 클래스.
# transforms.ToTensor() : [0,255] uint8 이미지를 [0,1] 실수 Tensor로 변환
# ───────────────────────────────────────────────────────────────
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True,  download=True,
                   transform=transforms.ToTensor()),
    batch_size=128, shuffle=True)           # 학습용은 섞어서 랜덤 배치

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, transform=transforms.ToTensor()),
    batch_size=1024)                        # 테스트용은 섞을 필요 없음

# ───────────────────────────────────────────────────────────────
# 3. “이진화 가중치 + STE” 정의
#    PyTorch는 torch.autograd.Function을 상속해
#    ★ forward(순전파)와 backward(역전파)★ 를 직접 구현할 수 있다.
#    여기서는 sign() 함수로 가중치를 ±1 로 만든 뒤,
#    역전파 때는 gradient를 그대로 통과시켜 학습이 가능하게 한다(STE).
# ───────────────────────────────────────────────────────────────
class BinW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        return w.sign()          # sign : 음수 -1, 0포함 양수 +1

    @staticmethod
    def backward(ctx, g):
        return g                 # STE : 그래디언트를 ‘있는 그대로’ 전달

# 래퍼 함수 – 코드 짧게 쓰려고
def binarize(w):
    return BinW.apply(w)

# ───────────────────────────────────────────────────────────────
# 4. 신경망(Model) 정의 – 완전연결(FC) 2층 구조
#    (28×28=784 차원 → 은닉 1024 → 출력 10)
#    bias=False : BNN 연구에서 bias 없이도 충분히 학습됨 & 파라미터 절약
# ───────────────────────────────────────────────────────────────
class BinaryFC(nn.Module):       # nn.Module 은 ‘모든 네트워크’의 부모 클래스
    def __init__(self):
        super().__init__()       # 부모 초기화
        self.fc1 = nn.Linear(28*28, 1024, bias=False)  # 1층
        self.fc2 = nn.Linear(1024, 10,  bias=False)    # 2층

    def forward(self, x):        # 순전파 정의 – 입력 x → 출력 y
        x = x.view(x.size(0), -1) # (N,1,28,28) → (N,784) 평탄화
        x = F.linear(x, binarize(self.fc1.weight))  # 이진 가중치로 FC
        x = F.relu(x)            # ReLU : 0 이하 0, 양수 유지 (비선형성)
        x = F.linear(x, binarize(self.fc2.weight))  # 두 번째 FC
        return x                 # 소프트맥스 없이 로짓 그대로 반환

# ───────────────────────────────────────────────────────────────
# 5. 모델 인스턴스 & 옵티마이저(Adam) 설정
#    model.to(device) : CPU/GPU 중 선택한 장치로 올림
#    Adam : 학습률을 자동 보정하는 인기 옵티마이저
# ───────────────────────────────────────────────────────────────
model = BinaryFC().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

# ───────────────────────────────────────────────────────────────
# 6. 학습(Training) + 검증(Testing) 루프
#    Epoch = 전체 데이터를 1회 다 학습하는 단위 (여기서는 50회)
# ───────────────────────────────────────────────────────────────
for epoch in range(50):
    # ---------- 학습 단계 ----------
    model.train()               # drop-out, BN 등 ‘학습모드’로 전환
    correct = 0
    for data, target in tqdm(train_loader, desc=f'E{epoch}'):
        # ① 데이터를 ▶ GPU/CPU 장치로 복사
        data, target = data.to(device), target.to(device)
        opt.zero_grad()         # 이전 mini-batch Gradient 초기화
        output = model(data)    # 순전파 : y_hat 계산
        loss = F.cross_entropy(output, target) # 소프트맥스 + NLL
        loss.backward()         # 역전파 : dLoss/dW 계산 (STE 포함)
        opt.step()              # 가중치 갱신 (Adam 규칙)

        # 미니배치 정답 개수 카운트
        correct += output.argmax(1).eq(target).sum().item()

    train_acc = correct / len(train_loader.dataset) * 100
    print(f'Epoch {epoch:02d} ▶ Train Acc  {train_acc:.2f}%')

    # ---------- 검증 단계 ----------
    model.eval()                # 평가 모드 (drop-out OFF 등)
    correct = 0
    with torch.no_grad():       # 그래디언트 계산 OFF → 메모리/속도 절약
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(1)        # 예측 숫자 (0~9 중 최대값 인덱스)
            correct += pred.eq(target).sum().item()

    test_acc = correct / len(test_loader.dataset) * 100
    print(f'                └ Test Acc  {test_acc:.2f}%\n')


