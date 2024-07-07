import os
import random

# 定义一个列表，用于txt文件内存放路径及标签数据
data_list = []
# 初始化类别标签
class_label = -1
# 加载dataset图片数据
dataset_path = './images'
# 遍历文件，依次将文件名存入上述定义列表当中
for root, _, filenames in os.walk(dataset_path):
    for i in filenames:
        data = root + "/" + i + "\t" + str(class_label) + "\n"
        print(data)
        data_list.append(data)  # 依次添加，不清空
    class_label += 1
# 打乱txt文件中的数据，保证下面分类进行测试集与训练集每个标签都有涉及
random.shuffle(data_list)

# 定义训练文本数据列表
train_list = []
# 将打乱后的总数据列表中的80%的数据用于训练集
for i in range(int(len(data_list) * 0.8)):
    train_list.append(data_list[i])
# 创建并以“写”方式打开train.txt
with open('train.txt', 'w', encoding='UTF-8') as f:
    for train_img in train_list:
        f.write(str(train_img))  # 将训练数据集数据写入train.txt
    print(train_img)
# 定义测试文本数据列表
eval_list = []

# 将打乱后的总数据列表中的20%的数据用于训练集
for i in range(int(len(data_list) * 0.8), len(data_list)):
    eval_list.append(data_list[i])
# 创建并以“写”方式打开eval.txt
with open('./eval.txt', 'w', encoding='UTF-8') as f:
    print(eval_list)
    for eval_img in eval_list:
        f.write(eval_img)  # 将测试数据集数据写入eval.txt

print(len(data_list))