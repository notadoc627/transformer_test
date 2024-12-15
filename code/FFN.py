import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, inputs, hiddens, outputs, **kwargs):
        super(FNN, self).__init__(**kwargs)
        self.fcn1 = nn.Linear(inputs, hiddens)
        self.relu = nn.ReLU()
        self.fcn2 = nn.Linear(hiddens, outputs)

    def forward(self, X):
        # X input shape (batchsize, lens, features)
        # X output shape (batchsize, lens, outputs)
        return self.fcn2(self.relu(self.fcn1(X)))

# ffn = FNN(4,4,8)
# ffn.eval()
# # 获取batch = 1的输出
# print(ffn(torch.ones((2, 3, 4)))[0])