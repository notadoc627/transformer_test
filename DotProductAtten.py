import torch
import torch.nn as nn
import math
class DotProductAtten(nn.Module):
    def __init__(self, dropout, **kwags):
        super(DotProductAtten, self).__init__(**kwags)
        self.dropout = nn.Dropout(dropout)

    def sequence_mask(self, X, valid_lens, value = 0):
        maxlen = X.size(1)
        mask = torch.arange(maxlen,device=
                            X.device).expand(len(valid_lens), maxlen) < valid_lens.unsqueeze(1)
        X[~mask] = 0

    def masked_softmax(self, X, valid_lens):
        # X：3d valid_lens: 1d/2d
        # 如果是 2D，表示每个序列中每个位置的有效长度，需要将其展平。
        if valid_lens is None:
            return nn.Softmax(X, dim=-1)
        else:
            shape = X.shape
            if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            else:
                valid_lens = valid_lens.reshape(-1)
            X = self.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=1e-6)
            return nn.Softmax(X.reshape(shape), dim=-1)


    def forward(self, q, k, v, valid_lens = None):
        d = q.shape[-1]
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = self.masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), v)
