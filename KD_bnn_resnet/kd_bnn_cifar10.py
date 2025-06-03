import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models # models 추가 (교사 모델 로드 시 필요할 수 있음)
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt # 필요시

# 0. 하이퍼파라미터 및 기본 설정

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} for KD Training")

# 학생 모델용 하이퍼파라미터 (기존 BNN 설정 유지 또는 조정)
BATCH_SIZE_STUDENT = 128
LEARNING_RATE_STUDENT = 1e-3 # 또는 이전 BNN 최적값
NUM_EPOCHS_KD = 100        # KD 학습 에폭
WEIGHT_DECAY_STUDENT = 1e-4 # 또는 이전 BNN 최적값

# KD 관련 하이퍼파라미터
TEMPERATURE = 4.0   # 소프트 타겟 생성 시 온도 (보통 1보다 큰 값, 3~10 사이)
ALPHA_KD = 0.7      # KD 손실의 가중치 (0.0 ~ 1.0 사이, 나머지는 CE 손실 가중치)
                    # 예: 총손실 = ALPHA_KD * L_KD + (1 - ALPHA_KD) * L_CE

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("--- Knowledge Distillation for BNN Student Model on CIFAR-10 ---")


# 1. CIFAR-10 데이터셋 준비 및 전처리 (이전과 동일)

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


class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()
    
    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output
    
def binary_activation_fn(x):
    return BinaryActivation.apply(x)

class BinaryWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight_real):
        if weight_real.ndim == 4:
            alpha = weight_real.abs().mean(dim=[1,2,3], keepdim = True)
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
        return grad_alpha_b
    
def binary_weight_fn(weight_real):
    return BinaryWeight.apply(weight_real)

class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        binarized_weight = binary_weight_fn(self.weight)
        return F.conv2d(x, binarized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

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
        out = binary_activation_fn(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = binary_activation_fn(out)
        return out
        
# 전체 학생 BNN 모델 정의

class StudentBNNResNet(nn.Module):
    def __init__(self, block, num_blocks_list, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks_list[0], stride=1)
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def StudentBNNResNet18(num_classes=10):
    return StudentBNNResNet(BiRealBasicBlock, [2, 2, 2, 2], num_classes)

student_model = StudentBNNResNet18(num_classes=len(classes)).to(DEVICE)
print("Student BNN Model (ResNet-18 based) created.")

def get_teacher_model_for_kd(num_classes=10):
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1, bias=False)  # CIFAR-10에 맞게 수정
    model.maxpool = nn.Identity()  # CIFAR-10에 맞게 수정
    return model

teacher_model_path = 'teacher_resnet18_cifar10_best1.pth'
teacher_model = get_teacher_model_for_kd(num_classes=len(classes))
teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=DEVICE))
teacher_model = teacher_model.to(DEVICE)
teacher_model.eval()  # 평가 모드로 설정
print("Teacher Model (ResNet-18) loaded for Knowledge Distillation.")


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

        # 학생 모델의 출력 (로짓)
        student_logits = student_model(inputs)

        # 교사 모델의 출력 (로짓) - 그래디언트 흐름 차단!
        with torch.no_grad(): # 교사 모델은 학습되지 않도록!
            teacher_logits = teacher_model(inputs)
        
        # KD 손실 함수를 사용하여 총 손실 계산
        loss = criterion_kd(student_logits, teacher_logits, labels, temp, alpha_kd_val)

        loss.backward() # 학생 모델의 가중치에 대한 그래디언트 계산
        optimizer.step()  # 학생 모델의 가중치 업데이트

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(student_logits.data, 1) # 학생 모델의 예측으로 정확도 계산
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.3f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples * 100
    return epoch_loss, epoch_acc

# evaluate_student_model 함수는 이전 BNN 코드의 evaluate_model과 동일하게 사용 가능
# (학생 모델만 평가하므로 KD 관련 로직 불필요, 일반적인 CE 손실로 평가)
def evaluate_student_model(model, test_loader, device): # criterion은 내부에서 CE 사용
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    criterion_eval = nn.CrossEntropyLoss() # 평가 시에는 CE 손실 사용
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="[Student Eval]", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion_eval(outputs, labels) # CE 손실
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.3f}")
    if total_samples == 0: return 0.0, 0.0
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples * 100
    return epoch_loss, epoch_acc

# 6. 메인 실행 부분 (KD 적용)
# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 학생 모델, 교사 모델은 위에서 이미 생성 및 로드됨
    
    # 옵티마이저 및 스케줄러는 학생 모델에 대해 설정
    optimizer = optim.AdamW(student_model.parameters(), lr=LEARNING_RATE_STUDENT, weight_decay=WEIGHT_DECAY_STUDENT)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_KD, eta_min=0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(NUM_EPOCHS_KD*0.5), int(NUM_EPOCHS_KD*0.75)], gamma=0.1)
    

    print(f"\nStarting Knowledge Distillation for student BNN model for {NUM_EPOCHS_KD} epochs...")
    print(f"KD Hyperparameters: Temperature={TEMPERATURE}, Alpha_KD={ALPHA_KD}")
    
    best_student_acc_kd = 0.0
    kd_history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    start_total_time = time.time()

    for epoch in range(NUM_EPOCHS_KD):
        epoch_start_time = time.time()

        # train_kd_epoch 함수 호출
        # distillation_loss 함수를 criterion_kd 인자로 전달 (함수 자체를 전달)
        train_loss, train_acc = train_kd_epoch(student_model, teacher_model, train_loader, distillation_loss, 
                                               optimizer, epoch, NUM_EPOCHS_KD, DEVICE, 
                                               TEMPERATURE, ALPHA_KD)
        
        # evaluate_student_model 함수 호출 (일반 CE 손실로 평가)
        test_loss, test_acc = evaluate_student_model(student_model, test_loader, DEVICE)

        kd_history['train_loss'].append(train_loss)
        kd_history['train_acc'].append(train_acc)
        kd_history['test_loss'].append(test_loss) # 평가 시 CE 손실
        kd_history['test_acc'].append(test_acc)
        
        scheduler.step()

        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS_KD} | "
              f"Train Loss (Total): {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss (CE): {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
              f"LR: {current_lr:.1e} | Time: {epoch_duration:.2f}s")

        if test_acc > best_student_acc_kd:
            best_student_acc_kd = test_acc
            print(f"🎉 New Best Student Accuracy with KD: {best_student_acc_kd:.2f}% (Epoch {epoch+1})")
            torch.save(student_model.state_dict(), 'student_bnn_resnet18_kd_best1.pth')
            print(f"Saved best student model (KD) to student_bnn_resnet18_kd_best1.pth")

    total_training_time = time.time() - start_total_time
    print("\n--- Student BNN Model KD Training Finished ---")
    print(f"Total training time: {total_training_time/60:.2f} minutes")
    print(f"Best Test Accuracy achieved by Student BNN with KD: {best_student_acc_kd:.2f}%")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1); plt.plot(kd_history['train_loss'], label='Train Loss'); plt.plot(kd_history['test_loss'], label='Test Loss'); plt.legend(); plt.title('Teacher Loss')
plt.subplot(1, 2, 2); plt.plot(kd_history['train_acc'], label='Train Acc'); plt.plot(kd_history['test_acc'], label='Test Acc'); plt.legend(); plt.title('Teacher Accuracy')
plt.savefig('Using_KD_resnet18_BNN_cifar10_curves.png')
plt.show()    
