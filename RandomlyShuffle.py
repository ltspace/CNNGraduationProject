import os
import random
import shutil

# 原始图片所在的目录
original_image_folder = 'data/DoneData/3'
# 新图片目标目录
new_image_folder = 'data\\DoneData\\3new'

# 确保新的目标目录存在
os.makedirs(new_image_folder, exist_ok=True)

# 获取所有图片文件名
files = [f for f in os.listdir(original_image_folder) if f.endswith('.png')]

# 随机打乱文件顺序
random.shuffle(files)

# 为每个打乱后的图片分配一个新的编号
for i, file in enumerate(files, start=1):
    new_name = f'{i}.png'  # 新编号从1开始
    
    # 构建完整的源文件路径和目标文件路径
    src_path = os.path.join(original_image_folder, file)
    dest_path = os.path.join(new_image_folder, new_name)

    # 复制文件到新目录
    shutil.copy(src_path, dest_path)
    
print("Files have been copied and renamed successfully in the new directory.")
