import time, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import psutil, os

# ------------------------------------------------------------------
# 1. 데이터 (테스트만, 배치 1024)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, download=True, transform=transform),
    batch_size=1024, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------
# 2. 공통: 이진 가중치 함수 & Sign+α activation
class BinW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w): return w.sign()
    @staticmethod
    def backward(ctx, g): return g

def binarize(w): return BinW.apply(w)

class SignAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.sign()
    @staticmethod
    def backward(ctx, g): return g * (abs(ctx.saved_tensors[0]) <= 1)
def sign_activation(x): return SignAct.apply(x) * x.abs().mean(dim=1, keepdim=True)

# ------------------------------------------------------------------
# 3. 모델 정의 (flag에 따라 가중치 bin 여부 선택)
class FC_BNN(nn.Module):
    def __init__(self, binary_weight=True, binary_act=False):
        super().__init__()
        self.bw, self.ba = binary_weight, binary_act
        self.fc1 = nn.Linear(784, 2048, bias=False)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 10,  bias=False)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        w1 = binarize(self.fc1.weight) if self.bw else self.fc1.weight
        x  = F.linear(x, w1)
        x  = self.bn1(x)
        x  = sign_activation(x) if self.ba else F.relu(x)
        w2 = binarize(self.fc2.weight) if self.bw else self.fc2.weight
        return F.linear(x, w2)

# ------------------------------------------------------------------
# 4. 측정 함수
def measure(model, tag, bit_weight=32, bit_act=32):
    model.to(device).eval()
    param_bits = sum(p.numel()*bit_weight for p in model.parameters()) if bit_act else \
                 sum(p.numel()*bit_weight for p in model.parameters())
    param_MB = param_bits/8/1024/1024
    print(f'\n【{tag}】추론용 파라미터 메모리 ≈ {param_MB:.2f} MB')

    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for data,_ in loader:
            data = data.to(device)
            _ = model(data)
    torch.cuda.synchronize() if device.type=='cuda' else None
    t1 = time.perf_counter()
    print(f'【{tag}】총 추론 시간 {t1-t0:.3f}s  (데이터 10,000장)')

    if device.type=='cuda':
        mem = torch.cuda.memory_allocated()/1024/1024
        print(f'【{tag}】실사용 GPU 메모리 {mem:.1f} MB')
    else:
        rss = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'【{tag}】프로세스 RSS 메모리 {rss:.1f} MB')

# ------------------------------------------------------------------
# 5. 두 버전 실행
fp32_model = FC_BNN(binary_weight=False, binary_act=False)
bnn_model  = FC_BNN(binary_weight=True,  binary_act=True)

measure(fp32_model, 'FP32 DNN', bit_weight=32)
measure(bnn_model,  '1-bit BNN', bit_weight=1)
