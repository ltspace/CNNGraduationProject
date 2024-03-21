import os
import imageio
import imgaug.augmenters as iaa

# 定义一个图像增强序列
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # 对50%的图像进行水平翻转
    iaa.LinearContrast((0.75, 1.5)), # 改变图像对比度
    iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)), # 添加高斯噪声
    iaa.Multiply((0.8, 1.2)), # 改变图像亮度
])

# 输入和输出文件夹路径
input_folder_path = 'data\\Processed224x224data\\1'
output_folder_path = 'data\\Augmented224x224data'

# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 遍历输入文件夹内的图像文件进行增强和保存
for filename in os.listdir(input_folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 检查文件扩展名
        input_file_path = os.path.join(input_folder_path, filename)
        output_file_path = os.path.join(output_folder_path, filename)
        
        # 读取图像
        image = imageio.imread(input_file_path)
        
        # 应用增强序列
        image_aug = seq(image=image)
        
        # 保存增强后的图像
        imageio.imwrite(output_file_path, image_aug)
        
        print(f"Processed and saved: {output_file_path}")
        
print("All images have been processed and saved.")