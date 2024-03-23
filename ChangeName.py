import os
import shutil

# 指定原图片所在的文件夹路径
old_dir_path = r'data\DoneData\7'
# 指定新图片将要移到的文件夹路径
new_dir_path = r'data\DoneData\8'

# 如果新的文件夹不存在，就创建它
if not os.path.exists(new_dir_path):
    os.makedirs(new_dir_path)

# 获取原路径下的所有文件列表
files = os.listdir(old_dir_path)

# 初始化计数器，从1开始
counter = 1

# 遍历所有文件
for file in files:
    # 拼接完整的文件路径
    file_path = os.path.join(old_dir_path, file)
    # 分离文件名和扩展名
    _, ext = os.path.splitext(file_path)
    # 只处理图片文件
    if ext.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        # 构造新的文件名
        new_file_name = f"{counter}{ext}"
        # 构造新的文件完整路径
        new_file_path = os.path.join(new_dir_path, new_file_name)
        # 将文件移动到新路径
        shutil.move(file_path, new_file_path)
        print(f"Moved and renamed '{file}' to '{new_file_path}'")
        # 更新计数器
        counter += 1