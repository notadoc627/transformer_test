fruits = {"apple": 10, "mango": 12, "durian": 20, "banana": 5}

# 初始化最大值的键
m = "apple"

# 遍历字典中的所有键值对
for key in fruits.keys():
    # 如果当前键对应的值大于已知的最大值，则更新最大值和对应的键
    if fruits[key] > fruits[m]:
        max_value = fruits[key]
        m = key  # 更新最大值对应的键

print('{}:{}'.format(m, fruits[m]))