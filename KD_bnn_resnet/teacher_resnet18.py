import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BATCH_SIZE = 128
Learning_Rate = 0.1

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_EPOCHS_TEACHER = 100

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print("--- Training Teacher Model (ResNet-18) on CIFAR-10 ---")

# CIFAR-10 dataset준비

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")



#Teacher Model (ResNet-18) 정의

def get_teacher_model(num_classes=10):

    model = models.resnet18(pretrained=None, num_classes=num_classes)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    model.maxpool = nn.Identity()

    return model

teacher_model = get_teacher_model(num_classes=len(classes)).to(DEVICE)
print("Teacher Model (ResNet-18 for CIFAR-10) created.")
print(teacher_model)


# 학습 및 평가 함수

def train_teacher_epoch(model, train_loader, criterion, optimizer, epoch_num, total_epochs,device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num+1:02d}/{total_epochs} [TRAIN]", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.3f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples * 100

    return epoch_loss, epoch_acc

def evaluate_teacher_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="[VALIDATE]", leave=False)
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

if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(teacher_model.parameters(), lr = Learning_Rate, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY)

    milestones = [int(NUM_EPOCHS_TEACHER * 0.5), int(NUM_EPOCHS_TEACHER * 0.75)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    print(f"\nStarting training teacher model for {NUM_EPOCHS_TEACHER} epochs...")

    best_teacher_acc = 0.0
    teacher_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    start_total_time = time.time()

    for epoch in range(NUM_EPOCHS_TEACHER):
        epoch_start_time = time.time()

        train_loss, train_acc = train_teacher_epoch(teacher_model, train_loader, criterion, optimizer, epoch, NUM_EPOCHS_TEACHER, DEVICE)
        test_loss, test_acc = evaluate_teacher_model(teacher_model, test_loader, criterion, DEVICE)

        teacher_history['train_loss'].append(train_loss)
        teacher_history['train_acc'].append(train_acc)
        teacher_history['test_loss'].append(test_loss)  
        teacher_history['test_acc'].append(test_acc)
        scheduler.step()

        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr'] # 현재 학습률 확인
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS_TEACHER} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
              f"LR: {current_lr:.1e} | Time: {epoch_duration:.2f}s")

        if test_acc > best_teacher_acc:
            best_teacher_acc = test_acc
            print(f"🎉 New Best Teacher Accuracy: {best_teacher_acc:.2f}% (Epoch {epoch+1})")
            # 최고 성능 교사 모델 저장
            torch.save(teacher_model.state_dict(), 'teacher_resnet18_cifar10_best1.pth')
            print(f"Saved best teacher model to teacher_resnet18_cifar10_best.pth")

    total_training_time = time.time() - start_total_time
    print("\n--- Teacher Model Training Finished ---")
    print(f"Total training time: {total_training_time/60:.2f} minutes")
    print(f"Best Test Accuracy achieved by Teacher Model: {best_teacher_acc:.2f}%")


# Teacher Model Learning curve plot

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1); plt.plot(teacher_history['train_loss'], label='Train Loss'); plt.plot(teacher_history['test_loss'], label='Test Loss'); plt.legend(); plt.title('Teacher Loss')
plt.subplot(1, 2, 2); plt.plot(teacher_history['train_acc'], label='Train Acc'); plt.plot(teacher_history['test_acc'], label='Test Acc'); plt.legend(); plt.title('Teacher Accuracy')
plt.savefig('teacher_resnet18_cifar10_curves.png')
plt.show()       