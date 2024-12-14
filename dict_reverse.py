str = input()
try:
    # 尝试将输入的字符串转换为字典
    original_dict = eval(str)

    # 检查转换结果是否为字典类型
    if not isinstance(original_dict, dict):
        raise ValueError("Input Error")

    # 初始化翻转后的字典
    reversed_dict = {}

    # 遍历原始字典的键值对
    for key, value in original_dict.items():
        # 将值作为键，键作为值添加到翻转后的字典中
        # 如果值不是唯一的，最后一个键值对将覆盖之前的值
        reversed_dict[value] = key

    print(reversed_dict)
except:
    # 如果转换过程中出现错误，返回错误信息
    print("Input Error")