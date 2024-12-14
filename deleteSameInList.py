input_str = input()
# 将输入的字符串分割成列表，假设列表元素由逗号分隔，并且没有引号
input_list = input_str.split(',')

appeared = set()  # 用于存储已经出现过的元素
output_list = []  # 用于存储结果的列表
# 遍历输入列表中的每个元素
for item in input_list:
    # 去除元素两端的空白字符
    item = item.strip()
    if item not in appeared:  # 如果元素不在seen集合中，说明是第一次出现
        appeared.add(item)  # 将元素添加到seen集合中
        output_list.append(item)  # 将元素添加到输出列表中

# 输出去除重复元素后的新列表
print(output_list)