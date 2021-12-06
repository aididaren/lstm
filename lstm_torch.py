# 来源 https://blog.csdn.net/jining11/article/details/90675276
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
import torch.nn.init as init

from typing import Tuple

import math
import numpy as np
import random
 
from matplotlib import pyplot as plt
# %matplotlib inline



class NaiveLSTM(nn.Module):
    """Naive LSTM like nn.LSTM"""
    def __init__(self, input_size: int, hidden_size: int):
        super(NaiveLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入门的权重矩阵和bias矩阵
        self.w_ii = Parameter(Tensor(hidden_size, input_size))
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(Tensor(hidden_size, 1))
        self.b_hi = Parameter(Tensor(hidden_size, 1))

        # 遗忘门的权重矩阵和bias矩阵
        self.w_if = Parameter(Tensor(hidden_size, input_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(Tensor(hidden_size, 1))
        self.b_hf = Parameter(Tensor(hidden_size, 1))

        # 输出门的权重矩阵和bias矩阵
        self.w_io = Parameter(Tensor(hidden_size, input_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(Tensor(hidden_size, 1))
        self.b_ho = Parameter(Tensor(hidden_size, 1))
        
        # cell的的权重矩阵和bias矩阵
        self.w_ig = Parameter(Tensor(hidden_size, input_size))
        self.w_hg = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(Tensor(hidden_size, 1))
        self.b_hg = Parameter(Tensor(hidden_size, 1))

        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs: Tensor, state: Tuple[Tensor]) \
        -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
#       ->用来提示该函数返回值的数据类型
        """Forward
        Args:
            inputs: [1, 1, input_size]
            state: ([1, 1, hidden_size], [1, 1, hidden_size])
        """

#         batch_size, seq_size , _ = inputs.size()

        if state is None:
            h_t = torch.zeros(1, self.hidden_size).t()
            c_t = torch.zeros(1, self.hidden_size).t()
        else:
            (h, c) = state
            h_t = h.squeeze(0).t()
            c_t = c.squeeze(0).t()

        hidden_seq = []

        seq_size = 1
        for t in range(seq_size):
            x = inputs[:, t, :].t()
            # input gate
            i = torch.sigmoid(self.w_ii @ x + self.b_ii + self.w_hi @ h_t +
                              self.b_hi)
            # forget gate
            f = torch.sigmoid(self.w_if @ x + self.b_if + self.w_hf @ h_t +
                              self.b_hf)
            # cell
            g = torch.tanh(self.w_ig @ x + self.b_ig + self.w_hg @ h_t
                           + self.b_hg)
            # output gate
            o = torch.sigmoid(self.w_io @ x + self.b_io + self.w_ho @ h_t +
                              self.b_ho)
            
            c_next = f * c_t + i * g
            h_next = o * torch.tanh(c_next)
            c_next_t = c_next.t().unsqueeze(0)
            h_next_t = h_next.t().unsqueeze(0)
            hidden_seq.append(h_next_t)

        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq, (h_next_t, c_next_t)

def reset_weigths(model):
    """reset weights
    """
    for weight in model.parameters():
        init.constant_(weight, 0.5)

# print("1111111111")
# inputs = torch.ones(1, 1, 10)
# h0 = torch.ones(1, 1, 20)
# c0 = torch.ones(1, 1, 20)
# print(h0.shape, h0)
# print(c0.shape, c0)
# print(inputs.shape, inputs)


# print("\n\n\n222222222")
# # test naive_lstm with input_size=10, hidden_size=20
# naive_lstm = NaiveLSTM(10, 20)
# reset_weigths(naive_lstm)
# output1, (hn1, cn1) = naive_lstm(inputs, (h0, c0))
# print(hn1.shape, cn1.shape, output1.shape)
# print(hn1)
# print(cn1)
# print(output1)

# print("\n\n\n333333333")
# # Use official lstm with input_size=10, hidden_size=20
# lstm = nn.LSTM(10, 20)
# reset_weigths(lstm)
# output2, (hn2, cn2) = lstm(inputs, (h0, c0))
# print(hn2.shape, cn2.shape, output2.shape)
# print(hn2)
# print(cn2)
# print(output2)


########################################
# Grad for LSTM
def setup_seed(seed):
    """
    确保每次都会生成相同的结果
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def show_gates(i_s, o_s, f_s):
    """Show input gate, output gate, forget gate for LSTM
    """
    plt.plot(i_s, "r", label="input gate")
    plt.plot(o_s, "b", label="output gate")
    plt.plot(f_s, "g", label="forget gate")
    plt.title('Input gate, output gate and forget gate of LSTM')
    plt.xlabel('t', color='#1C2833')
    plt.ylabel('Mean Value', color='#1C2833')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def lstm_step(x, h, c, w_ii, b_ii, w_hi, b_hi,
                  w_if, b_if, w_hf, b_hf,
                  w_ig, b_ig, w_hg, b_hg,
                  w_io, b_io, w_ho, b_ho, use_forget_gate=True):
    """run lstm a step
    """
    x_t = x.t()
    h_t = h.t()
    c_t = c.t()
    i = torch.sigmoid(w_ii @ x_t + b_ii + w_hi @ h_t + b_hi)
    o = torch.sigmoid(w_io @ x_t + b_io + w_ho @ h_t + b_ho)
    g = torch.tanh(w_ig @ x_t + b_ig + w_hg @ h_t + b_hg)
    f = torch.sigmoid(w_if @ x_t + b_if + w_hf @ h_t + b_hf)
    if use_forget_gate:
        c_next = f * c_t + i * g
    else:
        c_next = c_t + i * g
    h_next = o * torch.tanh(c_next)
    c_next_t = c_next.t()
    h_next_t = h_next.t()
    
    i_avg = torch.mean(i).detach()
    o_avg = torch.mean(o).detach()
    f_avg = torch.mean(f).detach()
    
    return h_next_t, c_next_t, f_avg, i_avg, o_avg

device='cpu'
hidden_size = 50
input_size = 100
sequence_len = 100
high = 1000000
test_idx = torch.randint(high=high, size=(1, sequence_len)).to(device)
setup_seed(45)
embeddings = nn.Embedding(high, input_size).to(device)
test_embeddings = embeddings(test_idx).to(device)
h_0 = torch.zeros(1, hidden_size, requires_grad=True).to(device)
c_0 = torch.zeros(1, hidden_size, requires_grad=True).to(device)
h_t = h_0
c_t = c_0
# print(test_embeddings)
# print(h_0)
# print(c_0)

# 不用遗忘门
lstm = NaiveLSTM(input_size, hidden_size).to(device)
iters = test_embeddings.size(1)
lstm_grads = []
i_s = []
o_s = []
f_s = []
for t in range(iters):
    h_t, c_t, f, i, o = lstm_step(test_embeddings[: , t, :], h_t, c_t, 
                               lstm.w_ii, lstm.b_ii, lstm.w_hi, lstm.b_hi,
                               lstm.w_if, lstm.b_if, lstm.w_hf, lstm.b_hf,
                               lstm.w_ig, lstm.b_ig, lstm.w_hg, lstm.b_hg,
                               lstm.w_io, lstm.b_io, lstm.w_ho, lstm.b_ho,
                               use_forget_gate=False)
    loss = h_t.abs().sum()
    h_0.retain_grad()
    loss.backward(retain_graph=True)
    lstm_grads.append(torch.norm(h_0.grad).item())
    i_s.append(i)
    o_s.append(o)
    f_s.append(f)
    h_0.grad.zero_()
    lstm.zero_grad()
plt.plot(lstm_grads)




show_gates(i_s, o_s, f_s)


#使用遗忘门
setup_seed(45)
embeddings = nn.Embedding(high, input_size).to(device)
test_embeddings = embeddings(test_idx).to(device)
h_0 = torch.zeros(1, hidden_size, requires_grad=True).to(device)
c_0 = torch.zeros(1, hidden_size, requires_grad=True).to(device)
h_t = h_0
c_t = c_0
print(test_embeddings)
print(h_0)
print(c_0)

lstm = NaiveLSTM(input_size, hidden_size).to(device)
## BIG CHANGE!!
lstm.b_hf.data = torch.ones_like(lstm.b_hf) * 1/2
lstm.b_if.data = torch.ones_like(lstm.b_if) * 1/2
iters = test_embeddings.size(1)
lstm_grads = []
i_s = []
o_s = []
f_s = []
for t in range(iters):
    h_t, c_t, f, i, o = lstm_step(test_embeddings[: , t, :], h_t, c_t, 
                               lstm.w_ii, lstm.b_ii, lstm.w_hi, lstm.b_hi,
                               lstm.w_if, lstm.b_if, lstm.w_hf, lstm.b_hf,
                               lstm.w_ig, lstm.b_ig, lstm.w_hg, lstm.b_hg,
                               lstm.w_io, lstm.b_io, lstm.w_ho, lstm.b_ho,
                               use_forget_gate=True)
    loss = h_t.abs().sum()
    h_0.retain_grad()
    loss.backward(retain_graph=True)
    lstm_grads.append(torch.norm(h_0.grad).item())
    i_s.append(i)
    o_s.append(o)
    f_s.append(f)
    h_0.grad.zero_()
    lstm.zero_grad()

# plt.plot(lstm_grads)

# show_gates(i_s, o_s, f_s)