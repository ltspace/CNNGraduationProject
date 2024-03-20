import os
from PIL import Image

# 替换为你的文件夹路径
folder_path = 'D:\\QQ\\QQ数据\\1409757351\\FileRecv\\test(1)\\test\\11\\vott-json-export'

# 列出文件夹中的所有文件
files = os.listdir(folder_path)

# 遍历文件检查每个图像
for file in files:
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, file)
    # 确保是文件且符合图像格式
    if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 打开图像
        image = Image.open(file_path)
        
        # 检查图像模式
        if image.mode == 'RGB':
            print(f'图像 "{file}" 是三通道RGB格式。')
        else:
            print(f'图像 "{file}" 不是三通道RGB格式。当前格式为: {image.mode}')
            # 如果需要将非RGB图像转换为RGB
            image = image.convert('RGB')
            image.save(file_path)  # 保存转换后的图像