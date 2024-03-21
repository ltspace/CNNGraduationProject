import os
import cv2
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, GaussNoise, Normalize,
    Rotate, ShiftScaleRotate, RandomGamma, RandomCrop, RandomResizedCrop, Resize, ElasticTransform
)
from albumentations.pytorch import ToTensorV2

# 定义图像增强序列
augmentations = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    RandomGamma(p=0.5),
    RandomCrop(height=224, width=224, p=0.5),
    RandomResizedCrop(height=224, width=224, p=0.5),
    Resize(height=224, width=224),
    Rotate(limit=15, p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    ElasticTransform(alpha=1, sigma=0.5, alpha_affine=1, p=0.5),
    ToTensorV2(),
])

# 输入和输出文件夹路径
input_folder_path = 'data/Processed224x224data/3'
output_folder_path = 'data/Albumentations/3'

# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 定义要生成的增强图像数量
num_augmented_images = 9

# 遍历输入文件夹中的图像，增强它们，并保存
for filename in os.listdir(input_folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_file_path = os.path.join(input_folder_path, filename)
        
        # 读取图像
        image = cv2.imread(input_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        
        # 应用图像增强
        for i in range(num_augmented_images):
            # 准备增强管道的输入数据
            image_input = {"image": image}
            
            # 应用增强
            augmented = augmentations(**image_input)
            
            # 获取增强后的图像
            image_aug = augmented['image']
            
            # 保存增强后的图像
            output_filename = f"{filename.split('.')[0]}_aug_{i}.png"
            output_file_path = os.path.join(output_folder_path, output_filename)
            image_aug = image_aug.cpu().numpy().transpose(1, 2, 0)  # 从Tensor转换为NumPy数组，并调整通道顺序
            image_aug = cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR)  # 转换为BGR格式
            cv2.imwrite(output_file_path, image_aug)
            
            print(f"Processed and saved: {output_file_path}")
        
print("All images have been processed and saved with multiple augmentations.")