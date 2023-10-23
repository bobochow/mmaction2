


# 打开第一个txt文件
with open('data/hmdb51/hmdb51_val_split_1_videos.txt', 'r', encoding='utf-8') as file1:
    content1 = file1.read()

# 打开第二个txt文件
with open('data/hmdb51/hmdb51_val_split_2_videos.txt', 'r', encoding='utf-8') as file2:
    content2 = file2.read()

# 打开第三个txt文件
with open('data/hmdb51/hmdb51_val_split_3_videos.txt', 'r', encoding='utf-8') as file3:
    content3 = file3.read()

# 合并三个文件的内容
merged_content = content1 + content2  + content3

# 创建一个新的txt文件来存储合并后的内容
with open('data/hmdb51/val_videos.txt', 'w', encoding='utf-8') as merged_file:
    merged_file.write(merged_content)

print("合并完成并保存到merged_file.txt")
