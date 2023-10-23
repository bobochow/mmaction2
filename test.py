import os
import shutil

def rename_and_copy_images(input_directory, output_directory):
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.startswith("img_") and file.endswith(".jpg"):
                relative_path = os.path.relpath(root, input_directory)
                new_root = os.path.join(output_directory, relative_path)
                if not os.path.exists(new_root):
                    os.makedirs(new_root)

                old_path = os.path.join(root, file)
                filename, file_extension = os.path.splitext(file)
                parts = filename.split('_')
                if len(parts) == 2 and parts[0] == "img" and parts[1].isdigit():
                    i = int(parts[1])
                    new_filename = f"img_{i+1:05d}.jpg"
                    new_path = os.path.join(new_root, new_filename)
                    shutil.copy(old_path, new_path)

if __name__ == "__main__":
    input_directory = "data/hmdb51/rawframes"  # 替换为实际的文件夹路径
    output_directory = "data/hmdb51/rawframes_new"  # 替换为实际的输出文件夹路径

    rename_and_copy_images(input_directory, output_directory)
