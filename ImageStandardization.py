import os
from PIL import Image, ImageOps

def resize_and_save_images(input_dir, output_dir, target_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    with Image.open(image_path) as image:
                        # 更新部分：使用Image.Resampling.LANCZOS代替Image.ANTIALIAS
                        image = image.resize(target_size, Image.Resampling.LANCZOS)
                        if image.mode == 'RGBA':
                            image = image.convert('RGBA')
                            datas = image.getdata()

                            new_data = []
                            # 把透明部分的所有颜色替换成黑色
                            for item in datas:
                                if item[3] == 0:  # 透明度为0
                                    new_data.append((0, 0, 0, 255))  # 完全不透明的黑色
                                else:
                                    new_data.append(item)
                            image.putdata(new_data)
                            image = image.convert('RGB')

                        output_image_path = os.path.join(output_class_path, image_name)
                        image.save(output_image_path, 'JPEG')

# Set your actual paths here
image_directory = 'data/DoneData/'
output_directory = 'data/224Data/'

# Process images
resize_and_save_images(image_directory, output_directory)