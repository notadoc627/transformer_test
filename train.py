import torch
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epoch = 0.005, 200
inputs, hiddens, heads = 32, 64, 4
q, k, v = 32, 32, 32
normalized_shape = [32]

encoder = TransformerEncoder(len())
decoder = TransformerDecoder()
net = EncoderDecoder(encoder, decoder)
