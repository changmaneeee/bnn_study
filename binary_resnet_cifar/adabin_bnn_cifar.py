import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# 0. 하이퍼파라미터 및 기본 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"---------Adabin + BNN CIFAR-10---------")


# 1. 데이터 셋 준비 및 전처리

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# 2. 활성화 이진화 함수

class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        return x.sign()
    
    @staticmethod
    def backward(ctx, grad_output):

        return grad_output
    
def binary_activation(x):
    return BinaryActivation.apply(x)


# 2.2 Adabin 가중치 이진화 함수
class AdaBinaryWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight_real, Htanh_thresholds):
        if weight_real.ndim == 4:
            alpha = weight_real.abs().mean(dim=[1,2,3], keepdim=True)
        elif weight_real.ndim == 2:
            alpha = weight_real.abs().mean(dim=1, keepdim=True)
        else:
            alpha = weight_real.abs().mean()

    

#해당 code에서는 가장 간단한 형태의 임계값 구현을 진행함
# 즉, W>τ이면 1, W<=τ이면 -1로 이진화

        #weight_binary = (weight_real > Htanh_thresholds).sign()
        # W > τ 이면 1.0을, 그렇지 않으면(W <= τ 이면) -1.0을 선택
        weight_binary = torch.where(
            weight_real > Htanh_thresholds,  # 조건: W가 τ보다 큰가?
            torch.tensor(1.0, device=weight_real.device),  # 조건이 True일 때 선택할 값: +1.0
            torch.tensor(-1.0, device=weight_real.device)   # 조건이 False일 때 선택할 값: -1.0
        )   
        ctx.save_for_backward(weight_real, Htanh_thresholds, weight_binary)

        return alpha * weight_binary
    
    @staticmethod
    def backward(ctx, grad_alpha_b_prime):
        weight_real, Htanh_thresholds, weight_binary = ctx.saved_tensors
    
        grad_w_real = grad_alpha_b_prime.clone()
        #grad_thresholds = -grad_alpha_b_prime.clone() #기존 코드

        #upgrade코드
        if weight_real.ndim == 4:
            alpha_val = weight_real.abs().mean(dim=[1,2,3], keepdim=True)
        elif weight_real.ndim == 2:
            alpha_val = weight_real.abs().mean(dim=1, keepdim=True)
        else:
            alpha_val = weight_real.abs().mean()
        
        input_to_binarization = weight_real - Htanh_thresholds
        clipping_value = 1.0

        ste_derivative_of_b_prime = (input_to_binarization.abs() < clipping_value).float()

        grad_contributions_to_tau = grad_alpha_b_prime * alpha_val * ste_derivative_of_b_prime * (-1.0)

        if grad_contributions_to_tau.ndim == 4 and Htanh_thresholds.ndim == 4: # Conv2d
            # 각 출력 채널(dim=0)에 대해, 나머지 차원(dim=1,2,3: 입력채널, 커널높이, 커널너비)의
            # 그래디언트 값들을 모두 더합니다. keepdim=True로 차원 유지.
            grad_thresholds = grad_contributions_to_tau.sum(dim=[1,2,3], keepdim=True)
        elif grad_contributions_to_tau.ndim == 2 and Htanh_thresholds.ndim == 2: # Linear
            # 각 출력 뉴런(dim=0)에 대해, 입력 특징(dim=1)으로부터 오는 그래디언트 값들을 더합니다.
            grad_thresholds = grad_contributions_to_tau.sum(dim=1, keepdim=True)
        else:
            # 예외적인 상황이거나, 모양이 맞지 않으면 일단 0으로 채워서라도 에러는 피하게 합니다.
            # 하지만 이 경우는 로직을 다시 점검해야 합니다.
            print("Warning: Dimension mismatch for grad_thresholds in AdaBinaryWeight.backward. Filling with zeros.")
            grad_thresholds = torch.zeros_like(Htanh_thresholds)

        return grad_w_real, grad_thresholds

def adabinary_weight_fn(weight_real, Htanh_thresholds_param):
    return AdaBinaryWeight.apply(weight_real, Htanh_thresholds_param)


# 3. AdaBin 이진화 컨볼루션 레이어 정의

class AdaBinConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(AdaBinConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.Htanh_thresholds = nn.Parameter(torch.zeros(out_channels, 1, 1, 1))

    def forward(self, x):
        binarized_weight = adabinary_weight_fn(self.weight, self.Htanh_thresholds)
        return F.conv2d(x, binarized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class AdaBinaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(AdaBinaryLinear, self).__init__(in_features, out_features, bias)
        self.Htanh_thresholds = nn.Parameter(torch.zeros(out_features, 1))
        
    def forward(self, x):
        binarized_weight = adabinary_weight_fn(self.weight, self.Htanh_thresholds)
        return F.linear(x, binarized_weight, self.bias)
    

# 4. Adabin BNN 모델 정의
class AdaBiRealBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_plane, planes, stride=1):
        super(AdaBiRealBasicBlock, self).__init__()
        
        self.conv1 = AdaBinConv2d(in_plane, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = AdaBinConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_plane != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_plane, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        
    def forward(self, x):
        out = binary_activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = binary_activation(out)
        return out
        
# 5. Adabin BNN 모델 정의(ResNet18)
class AdaBinResNet(nn.Module):
    def __init__(self, block, num_blocks_list, num_classes=10, first_conv_binary=False, last_fc_binary_type="real"):
        super(AdaBinResNet, self).__init__()
        self.in_planes = 64

        if first_conv_binary:
            print("Using binary first convolution")
            self.conv1 = AdaBinConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        else:
            print("Using real first convolution")
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks_list[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks_list[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if last_fc_binary_type == "adabinary":
            print("Using AdaBinary last fully connected layer")
            self.linear = AdaBinaryLinear(512 * block.expansion, num_classes)

        elif last_fc_binary_type == "binary":
            print("Using Binary Linear for the last layer (Non-AdaBin).")
            print("Warning: BinaryLinear (Non-AdaBin) for last layer is not fully defined here. Using Real for now.")
            self.linear = nn.Linear(512 * block.expansion, num_classes)
            
        else:
            print("Using Real Linear for the last layer")
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
                if hasattr(m, 'Htanh_thresholds') and m.Htanh_thresholds is not None:
                    nn.init.zeros_(m.Htanh_thresholds)

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

def AdaBiRealResNet18(num_classes=10, first_conv_binary=False, last_fc_binary_type="real"):
    return AdaBinResNet(AdaBiRealBasicBlock, [2, 2, 2, 2], num_classes, first_conv_binary, last_fc_binary_type)

# 6. 학습 및 평가 루프

def train_one_epoch(model, train_loader, criterion, optimizer, epoch_num, total_epochs, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_samples = 0
    correct_predictions = 0

    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num+1:02d}/{total_epochs} [Train]", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # 여기서 AdaBinaryWeight의 backward가 호출됨
        optimizer.step() # 가중치와 Htanh_thresholds 파라미터가 업데이트됨
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.3f}")
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples * 100
    return epoch_loss, epoch_acc

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.3f}")
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples * 100
    return epoch_loss, epoch_acc


# ───────────────────────────────────────────────────────────────
# 7. 메인 실행 부분
# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*30 + " AdaBin + Bi-Real Net (ResNet-18) on CIFAR-10 " + "="*30)

    # 모델 인스턴스 생성
    # first_conv_binary=False, last_fc_binary_type="real" (Bi-Real Net과 유사한 설정)
    model = AdaBiRealResNet18(num_classes=10, first_conv_binary=False, last_fc_binary_type="real").to(DEVICE)
    print(f"Model: AdaBiRealResNet18 (AdaBin applied to Conv layers in BasicBlocks)")

    # 옵티마이저는 모델의 모든 학습 가능한 파라미터 (가중치, BN 파라미터, Htanh_thresholds)를 포함
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0) # 또는 MultiStepLR

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    best_test_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    start_total_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, NUM_EPOCHS, DEVICE)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        scheduler.step()

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e} | Time: {epoch_duration:.2f}s")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"🎉 New Best Test Accuracy: {best_test_acc:.2f}% (Epoch {epoch+1})")
            # torch.save(model.state_dict(), 'adabin_bireal_resnet18_cifar10_best.pth')

        # (선택적) 학습된 Htanh_thresholds 값 확인 (매 에폭 또는 주기적으로)
        # for name, param in model.named_parameters():
        #     if "Htanh_thresholds" in name:
        #         print(f"{name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")


    total_training_time = time.time() - start_total_time
    print("\n--- Training Finished ---")
    print(f"Total training time: {total_training_time/60:.2f} minutes")
    print(f"Best Test Accuracy achieved with AdaBin: {best_test_acc:.2f}%")

    # 학습 곡선 및 층별 β (Htanh_thresholds) 그래프 그리기 (생략, 필요시 추가)
    # ... (matplotlib 코드 추가) ...
    # 예시: 층별 Htanh_thresholds 평균값 시각화
    thresholds_means = []
    threshold_names = []
    for name, param in model.named_parameters():
        if "Htanh_thresholds" in name and param.ndim == 4: # Conv layer thresholds
            thresholds_means.append(param.data.mean().item())
            threshold_names.append(name.replace('.Htanh_thresholds',''))
    
    if thresholds_means:
        
        plt.figure(figsize=(10, 5))
        plt.bar(threshold_names, thresholds_means)
        plt.xlabel("Layer Name")
        plt.ylabel("Mean Htanh_threshold (τ)")
        plt.title("Layer-wise Mean of Learned Thresholds (τ) in AdaBinConv2d")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('adabin_layer_thresholds.png')
        print("Saved layer-wise thresholds plot to adabin_layer_thresholds.png")
        plt.show()
        plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig('bireal_resnet18_cifar10_curves.png')
    plt.show()
