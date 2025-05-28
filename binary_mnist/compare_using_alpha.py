import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import time # 학습 시간 측정을 위해 추가

# ───────────────────────────────────────────────────────────────
# 0. 기본 설정
# ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 하이퍼파라미터
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20 # 비교를 위해 동일한 에폭 사용
LR_STEP_SIZE = 20 # 이전 코드와 동일하게 유지 (20 에폭이면 스케줄러는 1번만 동작)
LR_GAMMA = 0.1

# ───────────────────────────────────────────────────────────────
# 1. 데이터 준비
# ───────────────────────────────────────────────────────────────
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, transform=transforms.ToTensor()),
    batch_size=1024) # 테스트 시에는 더 큰 배치 사용 가능

# ───────────────────────────────────────────────────────────────
# 2. 이진화 함수 정의
# ───────────────────────────────────────────────────────────────

# 2.1. 활성화 이진화 (공통 사용)
class BinActive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        return a.sign()

    @staticmethod
    def backward(ctx, g):
        return g # STE

def binarize_activation(a):
    return BinActive.apply(a)

# 2.2. 가중치 이진화 - Alpha 미사용 (단순 Sign)
class BinWeightSimple(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_real):
        w_bin = w_real.sign()
        ctx.save_for_backward(w_real)
        return w_bin # 알파 없이 이진 가중치만 반환

    @staticmethod
    def backward(ctx, g_bin):
        w_real, = ctx.saved_tensors
        return g_bin # STE

def binarize_weights_simple(w_real):
    return BinWeightSimple.apply(w_real)

# 2.3. 가중치 이진화 - Alpha 사용 (XNOR-Net 스타일)
class BinWeightXNOR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_real):
        # XNOR-Net 논문에서는 필터별 α를 계산합니다.
        # FC 레이어의 경우, 각 출력 뉴런에 연결된 가중치 그룹별로 α를 계산.
        if w_real.ndim > 1: # FC 레이어 (out_features, in_features)
            alpha = w_real.abs().mean(dim=1, keepdim=True) # 각 출력 뉴런(행)에 대한 α
        else: # 혹시 1D 가중치라면
            alpha = w_real.abs().mean()

        w_bin = w_real.sign()
        ctx.save_for_backward(w_real)
        return alpha * w_bin # 이진화된 가중치에 스케일링 팩터 적용

    @staticmethod
    def backward(ctx, g_alpha_b):
        w_real, = ctx.saved_tensors
        # 단순 STE: g_alpha_b를 w_real의 그래디언트로 전달
        return g_alpha_b

def binarize_weights_xnor(w_real):
    return BinWeightXNOR.apply(w_real)

# ───────────────────────────────────────────────────────────────
# 3. 신경망 모델 정의
# ───────────────────────────────────────────────────────────────
class MLP_Binary(nn.Module):
    def __init__(self, num_classes=10, use_alpha_in_weights=False):
        super().__init__()
        self.fc1_features = 1024
        self.use_alpha_in_weights = use_alpha_in_weights

        self.fc1 = nn.Linear(28*28, self.fc1_features, bias=False)
        self.bn1 = nn.BatchNorm1d(self.fc1_features)
        self.fc2 = nn.Linear(self.fc1_features, num_classes, bias=False)
        # 두 번째 레이어 다음에도 BN을 추가해볼 수 있으나, XNOR-Net에서는 보통 활성화 전에 BN
        # self.bn2 = nn.BatchNorm1d(num_classes) # 필요시 추가

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # --- 첫 번째 블록 ---
        # 가중치 이진화 선택
        if self.use_alpha_in_weights:
            bin_w_fc1 = binarize_weights_xnor(self.fc1.weight)
        else:
            bin_w_fc1 = binarize_weights_simple(self.fc1.weight)
        x = F.linear(x, bin_w_fc1)

        x = self.bn1(x)
        x = binarize_activation(x) # 활성화 이진화

        # --- 두 번째 블록 ---
        # 가중치 이진화 선택
        if self.use_alpha_in_weights:
            bin_w_fc2 = binarize_weights_xnor(self.fc2.weight)
        else:
            bin_w_fc2 = binarize_weights_simple(self.fc2.weight)
        x = F.linear(x, bin_w_fc2)

        # 마지막 레이어는 일반적으로 로짓을 반환 (CrossEntropyLoss가 Softmax 포함)
        # F.log_softmax(x, dim=1) 대신 x를 반환하고 loss 함수로 F.cross_entropy 사용
        return x

# ───────────────────────────────────────────────────────────────
# 4. 학습 및 검증 함수
# ───────────────────────────────────────────────────────────────
def train_and_evaluate(model_name, model, train_loader, test_loader, optimizer, scheduler, num_epochs):
    print(f"\n--- Training: {model_name} ---")
    best_test_acc_model = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_loss_sum = 0
        current_lr = optimizer.param_groups[0]['lr']
        loop_train = tqdm(train_loader, desc=f'E{epoch+1:02d}/{num_epochs} Train | LR: {current_lr:.1e}', leave=False)

        for data, target in loop_train:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            loop_train.set_postfix(loss=loss.item())

        train_acc = 100. * train_correct / len(train_loader.dataset)
        avg_train_loss = train_loss_sum / len(train_loader)
        # print(f'E{epoch+1:02d} Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%') # tqdm 사용 시 중복될 수 있음

        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = 100. * test_correct / len(test_loader.dataset)

        print(f'E{epoch+1:02d}/{num_epochs} ▶ Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        if scheduler:
            scheduler.step()

        if test_acc > best_test_acc_model:
            best_test_acc_model = test_acc
            # print(f'  🎉 New Best Test Accuracy for {model_name}: {best_test_acc_model:.2f}%')

    end_time = time.time()
    training_time = end_time - start_time
    print(f"--- Training Finished for {model_name} ---")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Best Test Accuracy for {model_name} after {num_epochs} epochs: {best_test_acc_model:.2f}%\n")
    return best_test_acc_model

# ───────────────────────────────────────────────────────────────
# 5. 실험 실행
# ───────────────────────────────────────────────────────────────

# 실험 1: Alpha 미사용 모델
print("*"*30 + " Experiment 1: MLP without Alpha in Weights " + "*"*30)
model_no_alpha = MLP_Binary(use_alpha_in_weights=False).to(device)
optimizer_no_alpha = torch.optim.Adam(model_no_alpha.parameters(), lr=LEARNING_RATE)
scheduler_no_alpha = torch.optim.lr_scheduler.StepLR(optimizer_no_alpha, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
acc_no_alpha = train_and_evaluate("MLP (No Alpha)", model_no_alpha, train_loader, test_loader, optimizer_no_alpha, scheduler_no_alpha, NUM_EPOCHS)


# 실험 2: Alpha 사용 모델 (XNOR-Net 스타일)
print("*"*30 + " Experiment 2: MLP with Alpha in Weights (XNOR-style) " + "*"*30)
# 모델 가중치 초기화를 위해 새로 생성 (중요!)
# 이전 실험의 학습된 가중치를 사용하지 않도록 주의
torch.manual_seed(42) # 재현성을 위해 시드 고정 (선택 사항)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

model_with_alpha = MLP_Binary(use_alpha_in_weights=True).to(device)
optimizer_with_alpha = torch.optim.Adam(model_with_alpha.parameters(), lr=LEARNING_RATE)
scheduler_with_alpha = torch.optim.lr_scheduler.StepLR(optimizer_with_alpha, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
acc_with_alpha = train_and_evaluate("MLP (With Alpha)", model_with_alpha, train_loader, test_loader, optimizer_with_alpha, scheduler_with_alpha, NUM_EPOCHS)


# ───────────────────────────────────────────────────────────────
# 6. 결과 비교
# ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(" 최종 비교 결과 ".center(70, "="))
print("="*70)
print(f"MLP (No Alpha) 최종 Test Accuracy ({NUM_EPOCHS} epochs): {acc_no_alpha:.2f}%")
print(f"MLP (With Alpha) 최종 Test Accuracy ({NUM_EPOCHS} epochs): {acc_with_alpha:.2f}%")
print("="*70)

if acc_with_alpha > acc_no_alpha:
    print("🎉 Alpha를 사용한 모델의 정확도가 더 높습니다!")
elif acc_no_alpha > acc_with_alpha:
    print("🤔 Alpha를 사용하지 않은 모델의 정확도가 더 높습니다. (이론과 다를 수 있음)")
else:
    print("🙂 두 모델의 정확도가 동일합니다.")

print("\n목표: MNIST Test Accuracy ≥ 96 %")