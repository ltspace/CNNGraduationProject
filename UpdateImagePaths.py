import os
import re
import shutil

# 设置目标目录
target_directory = '.'

# 配置你的Markdown文件的路径
markdown_directory = '.'
# 配置存放图片的文件夹名称，相对于markdown_directory
images_directory = 'images\\'


# 替换绝对路径为相对路径的函数
def replace_image_path(file_path, image_dir):
    # 建立一个正则表达式模式匹配Markdown图片链接
    regex = r'!\[.*?\]\((.*?)\)'

    # 读取Markdown文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 使用正则表达式匹配并替换路径
    new_content = content
    matches = re.findall(regex, content)
    for match in matches:
        if not match.startswith('http'):  # 只替换本地文件路径
            new_image_path = os.path.join('/', image_dir, os.path.basename(match)).replace('\\', '/')
            new_content = new_content.replace(match, new_image_path)

    # 写回新的内容到文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)


# 写一个函数来处理文件的移动，并替换Markdown文件中图片的引用
def process_markdown_files(md_directory, target_dir):
    for root, dirs, files in os.walk(md_directory):
        for file in files:
            if file.lower().endswith('.md'):
                full_path = os.path.join(root, file)
                move_images(full_path, target_dir)
                replace_image_path(full_path, images_directory)


# 处理文件移动的函数
def move_images(md_file_path, target_dir):
    with open(md_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 正则表达式查找所有 Markdown 图片链接
    image_paths = re.findall(r'!\[.*?\]\((.*?)\)', content)

    if not image_paths:
        print(f"No images found in {md_file_path}")

    for image_path in image_paths:
        # 检查是否是绝对路径
        if os.path.isabs(image_path):
            # 获取图片名和目标路径
            image_name = os.path.basename(image_path)
            destination = os.path.join(target_dir, image_name)

            # 显示正在处理的图片
            print(f"Processing image: {image_name}")

            # 如果目标路径下没有这个文件再进行移动
            if not os.path.exists(destination):
                # 尝试移动图片，如果原始图片路径不存在则会抛出异常
                try:
                    shutil.move(image_path, destination)
                    print(f'Moved image {image_name} to {target_dir}')
                except FileNotFoundError:
                    print(f'Image not found at {image_path}, skipped moving')
            else:
                print(f'Image {image_name} already exists in target directory; no action taken')


# 开始处理Markdown文件和图片
print(f"Processing Markdown files and images in {markdown_directory}")
process_markdown_files(markdown_directory, target_directory)

print("All Markdown file image paths and images have been processed.")