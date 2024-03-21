import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import matplotlib.pyplot as plt



num_epochs = 50  # Define the number of epochs

# 检测并输出当前使用的计算设备
device_info = torch.cuda.is_available()  # True if CUDA is available, else False
if device_info:
    device_name = torch.cuda.get_device_name(0)  # Get the name of the CUDA device
    print(f"Training with CUDA device: {device_name}")
else:
    print("Training on CPU.")

# 加载预训练的ResNet-18模型
resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# 如需微调，则将模型置于训练模式
resnet18.train()

# 更改全连接层以适应二分类任务
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 2)

# 打印模型结构
# print(resnet18)
train_losses = []
val_losses = []

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): CSV文件的路径，含图像的标注。
            root_dir (string): 含所有图像的目录。
            transform (callable, optional): 需要应用于样本的可选变换。
        """
        self.image_labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        # 构造图像的完整路径
        img_name = os.path.join(self.root_dir, self.image_labels.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.image_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# 这里定义您的图像预处理和数据增强
transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # 如果所有图片都是224x224，则不需要这步
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的均值和标准差
])


# 指定CSV文件和根目录路径
csv_file = 'image_labels.csv'
root_dir = 'data\\224Data\\1'

# 实例化数据集
dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

criterion = nn.CrossEntropyLoss() # 如果是二分类问题，使用BCEWithLogitsLoss可能需要在模型输出层前添加一个Sigmoid激活函数。
optimizer = torch.optim.Adam(resnet18.parameters(), lr=0.001) # 也可以尝试使用不同的学习率或优化器。



# 训练集和验证集的划分
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 训练数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 验证数据加载器
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 每过7个周期，学习率乘以0.1。
scheduler = StepLR(optimizer, step_size=7, gamma=0.1) 



for epoch in range(num_epochs):
  # Set the model to training mode
    resnet18.train()  
    train_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = resnet18(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss = train_loss / len(train_loader.dataset)

    # Validation loop
    resnet18.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = resnet18(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
    val_loss = val_loss / len(val_loader.dataset)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

    # 在每个epoch后追加损失值
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Instantiate the test dataset
test_dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

# Create the test data loader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 在训练循环结束后，绘制损失趋势图
def plot_loss_curve(train_losses, val_losses, num_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, marker='x', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

correct = 0
total = 0
with torch.no_grad():
  for images, labels in test_loader:
    outputs = resnet18(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print(f'Accuracy of the model on test images: {100 * correct / total}%')
plot_loss_curve(train_losses, val_losses, num_epochs)

#保存模型
# torch.save(resnet18.state_dict(), 'model_weights.pth')


# 定义测试图像的路径
test_image_dir = 'data\\224Data\\1'

# 创建测试数据集实例
test_image_dataset = CustomDataset(csv_file='image_labels.csv', root_dir=test_image_dir, transform=transform)

# 创建测试数据加载器
test_image_loader = DataLoader(test_image_dataset, batch_size=32, shuffle=False)

# 在训练循环结束后，进行测试
def test_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 确保模型处于评估模式
resnet18.eval()

# 使用测试数据加载器进行测试
test_accuracy = test_model(resnet18, test_image_loader)
print(f'Model accuracy on test images: {test_accuracy}%')

# 如果您还想保存模型权重
# torch.save(resnet18.state_dict(), 'model_weights.pth')