import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    # 包含解码器自注意力，encoder-decoder注意力，ffn
    def __init__(self, i, inputs, hiddens, q, k, v, heads, num_hiddens, dropout, normalized_shape, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(q, k, v, heads, num_hiddens, heads, dropout)
        self.addnorm1 = AddNorm(normalized_shape, dropout)
        self.attention2 = MultiHeadAttention(q, k, v, heads, num_hiddens, heads, dropout)
        self.addnorm2 = AddNorm(normalized_shape, dropout)
        self.ffn = FFN(inputs, hiddens, num_hiddens)
        self.addnorm3 = AddNorm(normalized_shape, dropout)

    def forward(self, X, state):
        encoder_outs, encoder_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
            state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device.repeat(batch_size, 1)
                                          )
        else:
            dec_valid_lens = None
        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # en-de atten
        Y2 = self.attention2(Y, encoder_outs, encoder_outs, encoder_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state




# X = torch.ones((2, 100, 24))
# encoder_blk = EncoderBlock(24, 24, 24, 8, 24, 0.5, 24, 48, [100, 24])
# state = [encoder_blk(X, valid_lens), valid_lens, [None]]
