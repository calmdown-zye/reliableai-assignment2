
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os


from tqdm import tqdm
from torchvision.models import resnet50
import torchvision.transforms as transforms

# 1. data loda function
# 2. model load function
# 3. train function
# main 


def compute_mean_std():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=2)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for images, _ in loader:
        for c in range(3):
            mean[c] += images[:, c, :, :].mean()
            std[c] += images[:, c, :, :].std()
    mean /= len(loader)
    std /= len(loader)
    print(f'Computed Mean: {mean}')
    print(f'Computed Std: {std}')
    return mean.tolist(), std.tolist()


# 1. data loda function
def get_dataloaders(mean, std, batch_size=64):
    # train: augmentation yes
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    


    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


# model load
def build_resnet50(num_classes=10):
    model = resnet50(weights=None)  
    # ResNet50 마지막 fc 레이어를 CIFAR-10 클래스 수에 맞게 교체
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model



def train_model(model, train_loader, test_loader, epochs, lr, weight_decay, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs+1):
        # 학습
        model.train()
        correct, total = 0, 0
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += inputs.size(0)

        # 검증
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_correct += outputs.argmax(1).eq(targets).sum().item()
                test_total += inputs.size(0)

        scheduler.step()
        print(f'Epoch {epoch}: Train Acc: {correct/total*100:.2f}% | Test Acc: {test_correct/test_total*100:.2f}%')

    return model



if __name__ == '__main__':
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')
    
    os.makedirs('models', exist_ok=True)
    mean, std = compute_mean_std()
    train_loader, test_loader = get_dataloaders(mean, std, batch_size=64)

    # 모델 1: seed=42, lr=0.01, weight_decay=5e-4
    torch.manual_seed(42)
    model1 = build_resnet50()
    model1 = train_model(model1, train_loader, test_loader, 
                         epochs=10, lr=0.01, weight_decay=5e-4, device=device)
    torch.save(model1.state_dict(), 'models/model1.pth')
    print('model1')

    # 모델 2: seed=123, lr=0.05, weight_decay=1e-4
    torch.manual_seed(123)
    model2 = build_resnet50()
    model2 = train_model(model2, train_loader, test_loader,
                         epochs=10, lr=0.05, weight_decay=1e-4, device=device)
    torch.save(model2.state_dict(), 'models/model2.pth')
    print('model2')