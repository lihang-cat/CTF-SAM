from pathlib import Path


def count_images_in_folder(folder_path, image_extensions=None):
    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp','.npy',".dcm"}

    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"路径 '{folder_path}' 不是一个有效的文件夹。")

    count = 0
    for file in folder.rglob('*'):
        if file.is_file() and file.suffix.lower() in image_extensions:
            count += 1
    return count


# 使用示例
folder_path = r"E:\code\Med-CLIP-SAM\data\processed\PH2_256_Expert_DullRazor\imgs"  # 替换为你的实际路径，例如：r'C:\Users\YourName\Pictures'
total_images = count_images_in_folder(folder_path)
print(f"文件夹中共有 {total_images} 张图片。")