import csv

# 设定文件数量和文件名模式
num_images = 75
image_pattern = "{}.png"

# 创建要写入的CSV文件名
csv_filename = "image_labels.csv"

# 使用写入模式打开文件，创建一个CSV写入对象
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    # 写入标题行（如果需要）
    writer.writerow(['filename', 'label'])
    # 循环写入图像文件名
    for i in range(1, num_images + 1):
        # 写入图片文件名，这里留下标签空白，需要你之后填写
        writer.writerow([image_pattern.format(i), ''])

print(f"CSV文件 '{csv_filename}' 创建完成，已填入图像文件名。")