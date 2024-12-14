import math

import torch
import torch.nn as nn

class TransformerDecoder(DecoderBlock):
    def __init__(self, vocab_size, q, k, v, num_hiddens, normalized_shape, inputs, hiddens, heads, num_layers, dropout, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = positionEncoder(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(i, inputs, hiddens, q, k, v, heads, num_hiddens, dropout, normalized_shape))

        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embeddings(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights

        return self.dense(X), state

    # 允许外部代码访问_attention_weights
    def attention_weights(self):
        return self._attention_weights


