import os
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

# 定义模型


class ResNet18ForPrediction(nn.Module):
  def __init__(self, num_classes):
    super(ResNet18ForPrediction, self).__init__()
    # 这里创建预训练的ResNet-18模型实例
    self.resnet18 = models.resnet18(pretrained=False)
    # 替换最后的全连接层以适应您的分类任务
    num_ftrs = self.resnet18.fc.in_features
    self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

  def forward(self, x):
    x = self.resnet18(x)
    return x

# 加载模型权重
def load_model(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    return model

# 准备数据预处理
def prepare_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    return image_tensor

# 预测函数
def predict_image(model, image_tensor):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# 实例化模型
num_classes = 2  # 根据您的分类任务调整类别数
model = ResNet18ForPrediction(num_classes)
model = load_model(model, 'model_weights.pth')

# 定义图像预处理转换
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 图像预测目录
image_dir = 'C:\\Users\\14097\\Desktop\\cnn\\data\\224Data\\1'

# 预测并打印结果
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image_tensor = prepare_image(image_path, transform)
    prediction = predict_image(model, image_tensor)
    print(f'Image: {image_name}, Predicted Class: {prediction}')