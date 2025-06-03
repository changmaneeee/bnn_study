# kd_bnn_cifar10.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os # 파일/디렉토리 관리를 위해 추가

# ───────────────────────────────────────────────────────────────
# 0. 하이퍼파라미터 및 기본 설정
# ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} for KD Training")

# 학생 모델용 기본 하이퍼파라미터
BATCH_SIZE_STUDENT = 128
LEARNING_RATE_STUDENT_BASE = 1e-3 # 기본 학습률
NUM_EPOCHS_KD = 100 # 각 실험당 에폭 수 (시간 관계상 줄여서 테스트 가능)
WEIGHT_DECAY_STUDENT = 1e-4

SEED = 42
# SEED는 루프 밖에서 한 번만 설정하거나, 각 실험 시작 시 동일하게 설정하여 재현성 확보

print("--- Knowledge Distillation for BNN Student Model on CIFAR-10 (Hyperparameter Search) ---")

# 결과 저장 디렉토리 생성
results_dir = "kd_results"
os.makedirs(results_dir, exist_ok=True) # exist_ok=True는 디렉토리가 이미 있어도 에러 발생 안 함

# ───────────────────────────────────────────────────────────────
# 1. CIFAR-10 데이터셋 준비 및 전처리 (이전과 동일)
# ───────────────────────────────────────────────────────────────
# ... (데이터셋 준비 코드는 그대로) ...
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_STUDENT, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_STUDENT*2, shuffle=False, num_workers=2, pin_memory=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# ───────────────────────────────────────────────────────────────
# 2. 모델 정의 (BinaryActivation, BinaryWeight, BinaryConv2d, BiRealBasicBlock, StudentBNNResNet, get_teacher_model_for_kd)
#    - 이 부분은 이전 코드와 동일하게 유지 (길어서 생략, 실제 코드에는 포함되어야 함)
# ───────────────────────────────────────────────────────────────
# ... (이전 답변의 학생 BNN 모델 및 교사 모델 정의 코드 전체를 여기에 복사) ...
# --- 활성화 이진화 ---
class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # ctx.save_for_backward(x) # 클리핑 STE 시 필요
        return x.sign()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output # 단순 STE

def binary_activation_fn(x):
    return BinaryActivation.apply(x)

# --- 가중치 이진화 (Alpha 스케일링, τ 없음) ---
class BinaryWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight_real):
        if weight_real.ndim == 4:
            alpha = weight_real.abs().mean(dim=[1,2,3], keepdim=True)
        elif weight_real.ndim == 2:
            alpha = weight_real.abs().mean(dim=1, keepdim=True)
        else:
            alpha = weight_real.abs().mean()
        weight_binary = weight_real.sign()
        ctx.save_for_backward(weight_real)
        return alpha * weight_binary
    @staticmethod
    def backward(ctx, grad_alpha_b):
        weight_real, = ctx.saved_tensors
        return grad_alpha_b # 단순 STE

def binary_weight_fn(weight_real):
    return BinaryWeight.apply(weight_real)

# --- 이진화 컨볼루션 레이어 (τ 없음) ---
class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    def forward(self, x):
        binarized_weight = binary_weight_fn(self.weight)
        return F.conv2d(x, binarized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# --- 이진화 ResNet 기본 블록 (τ 없음) ---
class BiRealBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = BinaryConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinaryConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = binary_activation_fn(self.bn1(self.conv1(x))) # 수정된 부분: bn1 적용
        out_main = self.bn2(self.conv2(out))
        shortcut_out = self.shortcut(x)
        out = out_main + shortcut_out
        out = binary_activation_fn(out)
        return out

# --- 전체 학생 BNN 모델 (τ 없음) ---
class StudentBNNResNet(nn.Module):
    def __init__(self, block, num_blocks_list, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks_list[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out); out = self.layer4(out)
        out = self.avgpool(out); out = out.view(out.size(0), -1); out = self.linear(out)
        return out

def StudentBNNResNet18(num_classes=10):
    return StudentBNNResNet(BiRealBasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# --- 교사 모델 정의 ---
def get_teacher_model_for_kd(num_classes=10):
    model = models.resnet18(weights=None, num_classes=num_classes) # 또는 pretrained=False
    model.conv1 = nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
# ───────────────────────────────────────────────────────────────
# 3. KD 손실 함수 및 학습/평가 루프 정의 (이전과 동일)
# ───────────────────────────────────────────────────────────────
# ... (distillation_loss, train_kd_epoch, evaluate_student_model 함수 정의는 이전과 동일하게 여기에 복사) ...
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha_kd):
    loss_kd = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1)
    ) * (temperature ** 2)
    loss_ce = F.cross_entropy(student_logits, labels)
    total_loss = alpha_kd * loss_kd + (1.0 - alpha_kd) * loss_ce
    return total_loss

def train_kd_epoch(student_model, teacher_model, train_loader, criterion_kd, optimizer, epoch_num, total_epochs, device, temp, alpha_kd_val):
    student_model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num+1:02d}/{total_epochs} [KD Train]", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        student_logits = student_model(inputs)
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
        loss = criterion_kd(student_logits, teacher_logits, labels, temp, alpha_kd_val)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(student_logits.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.3f}")
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples * 100
    return epoch_loss, epoch_acc

def evaluate_student_model(model, test_loader, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    criterion_eval = nn.CrossEntropyLoss()
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="[Student Eval]", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion_eval(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.3f}")
    if total_samples == 0: return 0.0, 0.0
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples * 100
    return epoch_loss, epoch_acc
# ───────────────────────────────────────────────────────────────
# 4. 메인 실행 부분 (하이퍼파라미터 루프 추가)
# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # --- 교사 모델 로드 ---
    teacher_model_path = 'teacher_resnet18_cifar10_best1.pth' # 학습된 교사 모델 경로
    teacher_model = get_teacher_model_for_kd(num_classes=len(classes))
    teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=DEVICE))
    teacher_model = teacher_model.to(DEVICE)
    teacher_model.eval()
    print(f"Teacher Model (ResNet-18) loaded from {teacher_model_path} for Knowledge Distillation.")

    # --- 실험할 하이퍼파라미터 조합 정의 ---
    # 예시: TEMPERATURE와 ALPHA_KD 값을 변경하며 실험
    hyperparameter_sets = [
        {'temp': 2.0, 'alpha': 0.3, 'lr': 1e-3}, # 조합 1
        {'temp': 2.0, 'alpha': 0.5, 'lr': 1e-3},
        {'temp': 2.0, 'alpha': 0.7, 'lr': 1e-3},
        {'temp': 2.0, 'alpha': 0.9, 'lr': 1e-3},
        {'temp': 4.0, 'alpha': 0.3, 'lr': 1e-3}, # 조합 1
        {'temp': 4.0, 'alpha': 0.5, 'lr': 1e-3},
        {'temp': 4.0, 'alpha': 0.7, 'lr': 1e-3},
        {'temp': 4.0, 'alpha': 0.9, 'lr': 1e-3},
        {'temp': 7.0, 'alpha': 0.3, 'lr': 1e-3}, # 조합 1
        {'temp': 7.0, 'alpha': 0.5, 'lr': 1e-3},
        {'temp': 7.0, 'alpha': 0.7, 'lr': 1e-3},
        {'temp': 7.0, 'alpha': 0.9, 'lr': 1e-3},
        {'temp': 4.0, 'alpha': 0.5, 'lr': 5e-4},
        {'temp': 4.0, 'alpha': 0.9, 'lr': 5e-4},
        {'temp': 7.0, 'alpha': 0.5, 'lr': 5e-4},
        {'temp': 2.0, 'alpha': 0.5, 'lr': 5e-4}, # 조합 5 (학습률 변경)
        # 필요에 따라 더 많은 조합 추가
    ]
    
    # (선택적) 학생 모델 단독 성능 테스트 (ALPHA_KD = 0.0)
    hyperparameter_sets.insert(0, {'temp': 1.0, 'alpha': 0.0, 'lr': LEARNING_RATE_STUDENT_BASE})


    all_experiments_results = [] # 모든 실험 결과를 저장할 리스트

    for i, params in enumerate(hyperparameter_sets):
        current_temp = params['temp']
        current_alpha_kd = params['alpha']
        current_lr_student = params['lr']

        print(f"\n\n===== Experiment {i+1}/{len(hyperparameter_sets)} =====")
        print(f"Current Hyperparameters: Temperature={current_temp}, Alpha_KD={current_alpha_kd}, LR_Student={current_lr_student}")

        # --- 각 실험 시작 전 학생 모델 및 옵티마이저, 스케줄러 재생성 (중요!) ---
        # 시드 고정을 통해 동일한 초기 가중치로 시작 (선택적이지만 권장)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
            torch.cuda.synchronize() # 이전 GPU 작업 완료 대기 (혹시 모를 간섭 방지)

        student_model = StudentBNNResNet18(num_classes=len(classes)).to(DEVICE)
        # 주의: StudentBNNResNet18 내부에서 _initialize_weights()가 호출되어 매번 동일하게 초기화됨

        optimizer = optim.AdamW(student_model.parameters(), lr=current_lr_student, weight_decay=WEIGHT_DECAY_STUDENT)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(NUM_EPOCHS_KD*0.5), int(NUM_EPOCHS_KD*0.75)], gamma=0.1)
        
        print(f"Re-initialized Student BNN Model and Optimizer for new experiment.")

        best_student_acc_kd_for_this_run = 0.0
        kd_history_for_this_run = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        start_time_this_run = time.time()

        for epoch in range(NUM_EPOCHS_KD):
            epoch_start_time = time.time()
            train_loss, train_acc = train_kd_epoch(student_model, teacher_model, train_loader, distillation_loss,
                                                   optimizer, epoch, NUM_EPOCHS_KD, DEVICE,
                                                   current_temp, current_alpha_kd)
            test_loss, test_acc = evaluate_student_model(student_model, test_loader, DEVICE)

            kd_history_for_this_run['train_loss'].append(train_loss)
            kd_history_for_this_run['train_acc'].append(train_acc)
            kd_history_for_this_run['test_loss'].append(test_loss)
            kd_history_for_this_run['test_acc'].append(test_acc)
            scheduler.step()

            epoch_duration = time.time() - epoch_start_time
            current_lr_optim = optimizer.param_groups[0]['lr']
            print(f"Exp {i+1} - Epoch {epoch+1:02d}/{NUM_EPOCHS_KD} | "
                  f"Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {test_loss:.4f}, Val Acc: {test_acc:.2f}% | "
                  f"LR: {current_lr_optim:.1e} | Time: {epoch_duration:.2f}s")

            if test_acc > best_student_acc_kd_for_this_run:
                best_student_acc_kd_for_this_run = test_acc
                print(f"🎉 Exp {i+1} - New Best Acc: {best_student_acc_kd_for_this_run:.2f}% (Epoch {epoch+1})")
                # 각 실험별 최고 모델 저장 (파일 이름에 파라미터 값 포함)
                model_save_name = f"student_bnn_T{current_temp}_A{current_alpha_kd}_LR{current_lr_student}_best.pth"
                torch.save(student_model.state_dict(), os.path.join(results_dir, model_save_name))
                # print(f"Saved best model for this run to {os.path.join(results_dir, model_save_name)}")


        run_duration_total = time.time() - start_time_this_run
        print(f"\n--- Experiment {i+1} Finished ---")
        print(f"Hyperparameters: T={current_temp}, Alpha={current_alpha_kd}, LR={current_lr_student}")
        print(f"Total training time for this run: {run_duration_total/60:.2f} minutes")
        print(f"Best Test Accuracy for this run: {best_student_acc_kd_for_this_run:.2f}%")

        # 결과 저장
        current_experiment_result = {
            'params': {'T': current_temp, 'alpha_kd': current_alpha_kd, 'lr_student': current_lr_student},
            'best_accuracy': best_student_acc_kd_for_this_run,
            'training_time_minutes': run_duration_total/60,
            'history': kd_history_for_this_run
        }
        all_experiments_results.append(current_experiment_result)

        # 그래프 저장 (각 실험별)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(kd_history_for_this_run['train_loss'], label='Train Loss')
        plt.plot(kd_history_for_this_run['test_loss'], label='Test Loss (CE)')
        plt.title(f'Loss (T={current_temp}, A={current_alpha_kd}, LR={current_lr_student})')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(kd_history_for_this_run['train_acc'], label='Train Accuracy')
        plt.plot(kd_history_for_this_run['test_acc'], label='Test Accuracy')
        plt.title(f'Accuracy (T={current_temp}, A={current_alpha_kd}, LR={current_lr_student})')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True)
        
        graph_save_name = f"curves_T{current_temp}_A{current_alpha_kd}_LR{current_lr_student}.png"
        plt.savefig(os.path.join(results_dir, graph_save_name))
        # plt.show() # 자동으로 돌릴 때는 show()를 주석 처리하는 것이 좋음
        plt.close() # 다음 그래프를 위해 현재 figure 닫기
        print(f"Saved learning curves for this run to {os.path.join(results_dir, graph_save_name)}")

    # --- 모든 실험 완료 후 결과 요약 ---
    print("\n\n===== All Experiments Finished =====")
    for idx, result in enumerate(all_experiments_results):
        print(f"Experiment {idx+1}: Params={result['params']}, Best Acc={result['best_accuracy']:.2f}%, Time={result['training_time_minutes']:.2f}min")

    # 최고 성능 실험 찾기
    if all_experiments_results:
        best_overall_experiment = max(all_experiments_results, key=lambda x: x['best_accuracy'])
        print("\n--- Best Overall Experiment ---")
        print(f"Params: {best_overall_experiment['params']}")
        print(f"Best Accuracy: {best_overall_experiment['best_accuracy']:.2f}%")
        print(f"Training Time: {best_overall_experiment['training_time_minutes']:.2f} minutes")