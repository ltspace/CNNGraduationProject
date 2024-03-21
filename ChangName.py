import os

# 指定存放图片的文件夹路径
folder_path = r'C:\Users\14097\Desktop\cnn\data\DoneData'

# 获取文件夹下所有文件名
files = os.listdir(folder_path)

# 遍历文件，并重命名为数字序号的格式
for index, file in enumerate(files):
    if file.endswith('.jpg') or file.endswith('.png'):  # 只处理图片文件
        file_extension = os.path.splitext(file)[1]  # 获取文件扩展名
        new_file_name = str(index + 1) + file_extension  # 新文件名，如1.jpg、2.png等

        # 处理可能重名的情况
        while os.path.exists(os.path.join(folder_path, new_file_name)):
            index += 1
            new_file_name = str(index) + file_extension

        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file_name))
