import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4 #가중치 감쇠 (RESNET에서 주로 사용됨)

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# CIFAR-10 데이터셋 로드
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), #4픽셀 패딩을 진행한 후 32x32로 랜덤 자름
    transforms.RandomHorizontalFlip(), # 수평 뒤집기
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #정규화
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #정규화
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

#데이터 셋 클래스 이름 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Number of classes: {len(classes)}")


# Binary Neural Network 모델 정의

# 2.1 활성화 이진화 함수 (STE 사용)
class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        #x, = ctx.saved_tensors
        #grad_input = grad_output.clone()
        #grad_input[x.abs() < 1.001] = 0  # STE: 기울기를 0으로 설정

        #return grad_input
        return grad_output

def binary_activation_fn(x):
    return BinaryActivation.apply(x)


#2.2 가중치 이진화 함수 (STE 사용)
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

        return grad_alpha_b
    
def binary_weight_fn(weight_real):
    return BinaryWeight.apply(weight_real)


# 3. 이진화된 컨볼루션 레이어 정의

class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,dilation=1, groups=1, bias=False):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        binarized_weight = binary_weight_fn(self.weight)

        return F.conv2d(x, binarized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class BinaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(BinaryLinear, self).__init__(in_features, out_features, bias)
    
    def forward(self, x):
        binarized_weight = binary_weight_fn(self.weight)
        
        return F.linear(x, binarized_weight, self.bias)


# Bi-Real Net의 이진화된 Resnet 기본 블록 정의

class BiRealBasicBlock(nn.Module):
    expansion = 1 #basic block의 경우 출력 채널 배수가 1

    def __init__(self, in_planes, planes, stride=1):
        super(BiRealBasicBlock, self).__init__()

        #첫 번째 컨볼루션 레이어
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

        shortcut_x = self.shortcut(x)

        out += shortcut_x
        out = binary_activation_fn(out)

        return out
    
# 5. Bi-Real Net 모델 정의

class BiRealResNet(nn.Module):
    def __init__(self, block_class, num_blocks_list, num_classes=10, first_conv_binary=False, last_fc_binary=False):
        super(BiRealResNet, self).__init__()

        self.in_planes = 64 #첫 번째 스테이지의 입력 채널 수 (초기 컨볼루션 출력 후)

        #초기 컨볼루션 레이어

        if first_conv_binary:
            print("Using binary first convolution layer")
            self.conv1 = BinaryConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            print("Using standard first convolution layer")
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block_class, 64, num_blocks_list[0], stride=1)
        self.layer2 = self._make_layer(block_class, 128, num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(block_class, 256, num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(block_class, 512, num_blocks_list[3], stride=2)

        #분류기
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if last_fc_binary:
            print("Using binary last fully connected layer")
            self.linear = BinaryLinear(512 * block_class.expansion, num_classes, bias=False)
        else:
            print("Using standard last fully connected layer")
            self.linear = nn.Linear(512 * block_class.expansion, num_classes, bias=False)

        self._initialize_weights()

    def _make_layer(self, block_class, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block_class(self.in_planes, planes, s))
            self.in_planes = planes * block_class.expansion
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
        out = out.view(out.size(0), -1)  # Flatten
        out = self.linear(out)

        return out
    
def BiRealResNet18(num_classes=10, first_conv_binary=False, last_fc_binary=False):
    return BiRealResNet(BiRealBasicBlock, [2, 2, 2, 2], num_classes, first_conv_binary, last_fc_binary)


# bi_real_net_cifar10.py (이어서)

# ───────────────────────────────────────────────────────────────
# 6. 학습 및 평가 루프
# ───────────────────────────────────────────────────────────────

def train_one_epoch(model, train_loader, criterion, optimizer, epoch_num, total_epochs, device):
    model.train() # 모델을 학습 모드로 설정
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # tqdm을 사용하여 진행 상황 표시
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num+1:02d}/{total_epochs} [Train]", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # 그래디언트 초기화

        outputs = model(inputs) # 순전파
        loss = criterion(outputs, labels) # 손실 계산

        loss.backward() # 역전파
        optimizer.step() # 가중치 업데이트

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # tqdm 진행률 바에 현재 손실과 정확도 표시
        progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.3f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples * 100
    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device):
    model.eval() # 모델을 평가 모드로 설정
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # 그래디언트 계산 비활성화
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
    print("\n" + "="*30 + " Bi-Real Net (ResNet-18) on CIFAR-10 " + "="*30)

    # 모델 인스턴스 생성
    # Bi-Real Net 논문에서는 첫 번째 conv와 마지막 fc는 실수형을 사용
    # first_conv_binary=False, last_fc_binary=False 가 기본 Bi-Real Net 설정
    model = BiRealResNet18(num_classes=10, first_conv_binary=False, last_fc_binary=False).to(DEVICE)
    print(f"Model: BiRealResNet18")
    # 모델 파라미터 수 확인 (옵션)
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {total_params:,}")


    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    # AdamW는 Adam에 weight decay를 좀 더 잘 적용하는 옵티마이저
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 학습률 스케줄러 (예: CosineAnnealingLR 또는 MultiStepLR)
    # Bi-Real Net 논문에서는 cosine annealing을 자주 사용
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0)
    # 또는 MultiStepLR
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(NUM_EPOCHS*0.5), int(NUM_EPOCHS*0.75)], gamma=0.1)


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

        scheduler.step() # 스케줄러 업데이트

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e} | Time: {epoch_duration:.2f}s")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"🎉 New Best Test Accuracy: {best_test_acc:.2f}% (Epoch {epoch+1})")
            # 최고 성능 모델 저장 (옵션)
            # torch.save(model.state_dict(), 'bireal_resnet18_cifar10_best.pth')

    total_training_time = time.time() - start_total_time
    print("\n--- Training Finished ---")
    print(f"Total training time: {total_training_time/60:.2f} minutes")
    print(f"Best Test Accuracy achieved: {best_test_acc:.2f}%")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.1e}")

    # 학습 곡선 그리기 (옵션)

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