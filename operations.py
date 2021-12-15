import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from scipy.sparse.linalg import eigs
import scipy.sparse as sp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


OPS = {
    'none': lambda C: Zero(),
    'skip_connect': lambda C: Identity(),
    # 'dcc_1': lambda C: DCCLayer(C, C, (1, 2), dilation=1),
    'dcc_2': lambda C: DCCLayer(C, C, (1, 2), dilation=2),
    'trans': lambda C: InformerLayer(C),
    's_trans': lambda C: SpatialInformerLayer(C),
    'diff_gcn': lambda C, supports, nodevec1, nodevec2: DiffusionConvLayer(
        2, supports, nodevec1, nodevec2, C, C),
    # 'cheb_gcn': lambda C, cheb, nodevec1, nodevec2, alpha: Cheb_gcn(2, cheb, C, C, nodevec1, nodevec2, alpha),
    # 'cnn': lambda C: CNN(C, C, (1, 2), dilation=1),
    # 'att1': lambda C: TransformerLayer(C),
    # 'att2': lambda C: SpatialTransformerLayer(C),
    # 'lstm': lambda C: LSTM(C, C),
    # 'gru': lambda C: GRU(C, C),
}


######################################################################
# RNN layer
######################################################################
class GRU(nn.Module):
    def __init__(self, c_in, c_out,):
        super(GRU, self).__init__()
        self.gru = nn.GRU(c_in, c_out, batch_first=True)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [b, N, T, f_in]
        x = x.reshape(-1, T, C)  # [bN, T, f_in]
        output, state = self.gru(x)
        output = output.reshape(b, N, T, C)
        output = output.permute(0, 3, 1, 2)

        return output


class LSTM(nn.Module):
    def __init__(self, c_in, c_out,):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(c_in, c_out, batch_first=True)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [b, N, T, f_in]
        x = x.reshape(-1, T, C)  # [bN, T, f_in]
        output, state = self.lstm(x)
        output = output.reshape(b, N, T, C)
        output = output.permute(0, 3, 1, 2)

        return output


class CNN(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride=1, dilation=1):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.filter_conv = CausalConv2d(c_in, c_out, kernel_size, stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(c_out, affine=False)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        x = self.relu(x)
        output = (self.filter_conv(x))
        output = self.bn(output)

        return output


class Cheb_gcn(nn.Module):
    """
    K-order chebyshev graph convolution layer
    """

    def __init__(self, K, cheb_polynomials, c_in, c_out, nodevec1, nodevec2, alpha):
        """
        :param K: K-order
        :param cheb_polynomials: laplacian matrix？
        :param c_in: size of input channel
        :param c_out: size of output channel
        """
        super(Cheb_gcn, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.c_in = c_in
        self.c_out = c_out
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.alpha = alpha
        self.DEVICE = cheb_polynomials[0].device
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out, affine=False)
        self.Theta = nn.ParameterList(  # weight matrices
            [nn.Parameter(torch.FloatTensor(c_in, c_out).to(self.DEVICE)) for _ in range(K)])

        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            self.theta_k = self.Theta[k]
            nn.init.xavier_uniform_(self.theta_k)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return: [batch_size, f_out, N, T]
        """
        x = self.relu(x)
        x = x.transpose(1, 2)  # [batch_size, N, f_in, T]

        batch_size, num_nodes, c_in, timesteps = x.shape

        adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)

        outputs = []
        for step in range(timesteps):
            graph_signal = x[:, :, :, step]  # [b, N, f_in]
            output = torch.zeros(
                batch_size, num_nodes, self.c_out).to(self.DEVICE)  # [b, N, f_out]

            for k in range(self.K):
                alpha, beta = F.softmax(self.alpha[k])
                T_k = alpha * self.cheb_polynomials[k] + beta * adp

                # T_k = self.cheb_polynomials[k]  # [N, N]
                self.theta_k = self.Theta[k]  # [c_in, c_out]
                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                output = output + rhs.matmul(self.theta_k)
            outputs.append(output.unsqueeze(-1))
        outputs = F.relu(torch.cat(outputs, dim=-1)).transpose(1, 2)
        outputs = self.bn(outputs)

        return outputs


class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ReLUConvBN(nn.Module):
    """
    ReLu -> Conv2d -> BatchNorm2d
    """
    def __init__(self, C_in, C_out, kernel_size, stride):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        return self.op(x)


class linear(nn.Module):
    """
    Linear for 2d feature map
    """
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))  # bias=True

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        return self.mlp(x)


class nconv(nn.Module):
    """
    张量运算
    """

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl, vw->ncwl', (x, A))
        return x.contiguous()


######################################################################
# Dilated causal convolution with GTU
######################################################################
class CausalConv2d(nn.Conv2d):
    """
    单向padding
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self._padding = (kernel_size[-1] - 1) * dilation
        super(CausalConv2d, self).__init__(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=(0, self._padding),
                                           dilation=dilation,
                                           groups=groups,
                                           bias=bias)

    def forward(self, input):
        result = super(CausalConv2d, self).forward(input)
        if self._padding != 0:
            return result[:, :, :, :-self._padding]
        return result


class DCCLayer(nn.Module):
    """
    dilated causal convolution layer with GLU function
    暂时用GTU代替
    """

    def __init__(self, c_in, c_out, kernel_size, stride=1, dilation=1):
        super(DCCLayer, self).__init__()
        self.relu = nn.ReLU()
        self.filter_conv = CausalConv2d(c_in, c_out, kernel_size, stride, dilation=dilation)
        self.gate_conv = CausalConv2d(c_in, c_out, kernel_size, stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(c_out, affine=False)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]这些block的input必须具有相同的shape？
        :return:
        """
        x = self.relu(x)
        filter = torch.tanh(self.filter_conv(x))
        # filter = self.filter_conv(x)
        gate = torch.sigmoid(self.gate_conv(x))
        output = filter * gate
        output = self.bn(output)
        # output = F.dropout(output, 0.5, training=self.training)

        return output


######################################################################
# node-wise Gated dilation causal convolution layer with GTU
######################################################################
# class dcc_nodewise(nn.Module):
#     """
#     modeling node-wise temporal dependency.
#     """
#
#     def __init__(self, c_in, c_out, kernel_size, dilation=1, num_of_vertices=207):
#         super(dcc_nodewise, self).__init__()
#         self.dilation = dilation
#         self.kernel_size = kernel_size[-1]
#         self.c_in = c_in
#         self.c_out = c_out
#         self._padding = (kernel_size[-1] - 1) * dilation
#         self.relu = nn.ReLU()
#         self.bn = nn.BatchNorm2d(c_out, affine=False)
#
#         self.lw1 = nn.Parameter(torch.FloatTensor(num_of_vertices, 4).to(DEVICE))
#         self.lw2 = nn.Parameter(torch.FloatTensor(num_of_vertices, 4).to(DEVICE))
#         self.rw1 = nn.Parameter(torch.FloatTensor(4, 2*c_in*c_out).to(DEVICE))
#         self.rw2 = nn.Parameter(torch.FloatTensor(4, 2*c_in*c_out).to(DEVICE))
#         self.lb1 = nn.Parameter(torch.FloatTensor(num_of_vertices, 4).to(DEVICE))
#         self.lb2 = nn.Parameter(torch.FloatTensor(num_of_vertices, 4).to(DEVICE))
#         self.rb1 = nn.Parameter(torch.FloatTensor(4, c_out).to(DEVICE))
#         self.rb2 = nn.Parameter(torch.FloatTensor(4, c_out).to(DEVICE))
#
#         # self.meta_weights = MLP([16, 3] + [c_in * c_out * kernel_size[-1]], 16,
#         #                         activation_function=nn.ReLU(), out_act=False)
#         # self.meta_bias = MLP([16, 3] + [c_out], 16,
#         #                      activation_function=nn.ReLU(), out_act=False)
#         # self.w1_memory = nn.Parameter(torch.randn(
#         #                     num_of_vertices, 16).to(DEVICE), requires_grad=True).to(DEVICE)
#         # self.w2_memory = nn.Parameter(torch.randn(
#         #                     num_of_vertices, 16).to(DEVICE), requires_grad=True).to(DEVICE)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.lw1)
#         nn.init.xavier_uniform_(self.lw2)
#         nn.init.xavier_uniform_(self.rw1)
#         nn.init.xavier_uniform_(self.rw2)
#         nn.init.xavier_uniform_(self.lb1)
#         nn.init.xavier_uniform_(self.lb2)
#         nn.init.xavier_uniform_(self.rb1)
#         nn.init.xavier_uniform_(self.rb2)
#
#     def forward(self, x):
#         """
#         :param x: [batch_size, f_in, N, T]
#         :return:
#         """
#         x = self.relu(x)
#         b, f_in, N, T = x.shape
#         x = x.permute(0, 2, 1, 3).reshape(b, f_in * N, T)
#         weights1 = torch.matmul(self.lw1, self.rw1).reshape(
#             N * self.c_out, self.c_in, self.kernel_size)  # (N*f_out, f_in, 2)
#         weights2 = torch.matmul(self.lw2, self.rw2).reshape(
#             N * self.c_out, self.c_in, self.kernel_size)
#         bias1 = torch.matmul(self.lb1, self.rb1).reshape(-1)  # (N, f_out)->(N*f_out)
#         bias2 = torch.matmul(self.lb2, self.rb2).reshape(-1)
#         # weights1 = self.meta_weights(self.w1_memory).reshape(N * self.c_out, f_in, self.kernel_size)
#         # weights2 = self.meta_weights(self.w2_memory).reshape(N * self.c_out, f_in, self.kernel_size)
#         # bias1 = self.meta_bias(self.w1_memory).reshape(-1)
#         # bias2 = self.meta_bias(self.w2_memory).reshape(-1)
#
#         filter = F.conv1d(x, weights1, bias1, padding=self._padding, dilation=self.dilation, groups=N)
#         filter = filter.view(b, N, self.c_out, -1).permute(0, 2, 1, 3)
#         gate = F.conv1d(x, weights2, bias2, padding=self._padding, dilation=self.dilation, groups=N)
#         gate = gate.view(b, N, self.c_out, -1).permute(0, 2, 1, 3)
#         if self._padding != 0:
#             filter = filter[:, :, :, :-self._padding]
#             gate = F.sigmoid(gate[:, :, :, :-self._padding])
#
#         output = filter * gate
#         output = self.bn(output)
#
#         return output


######################################################################
# Temporal attention layer
######################################################################
# class Temporal_Attention_layer(nn.Module):
#     def __init__(self, c_in, c_out, get_u2, num_of_vertices=207, num_of_timesteps=12):
#         super(Temporal_Attention_layer, self).__init__()
#         self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
#         # self.u2_memory = nn.Parameter(torch.FloatTensor(c_in, 8).to(DEVICE))
#         # self.get_u2 = get_u2
#         self.U2 = nn.Parameter(torch.FloatTensor(c_in, num_of_vertices).to(DEVICE))
#         self.U3 = nn.Parameter(torch.FloatTensor(c_in).to(DEVICE))
#         self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
#         self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))
#         self.relu = nn.ReLU()
#         self.bn = nn.BatchNorm2d(c_out, affine=False)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.U2)
#         nn.init.xavier_uniform_(self.be)
#         nn.init.xavier_uniform_(self.Ve)
#         nn.init.uniform_(self.U1)
#         nn.init.uniform_(self.U3)
#
#     def forward(self, x):
#         '''
#         :param x: [batch_size, f_in, N, T]
#         :return: [batch_size, f_in, N, T]
#         参数量有点大
#         '''
#         # self.U2 = self.get_u2(self.u2_memory)
#
#         x = self.relu(x)
#         x = x.permute(0, 2, 1, 3)  # [batch_size, f_in, N, T] -> (batch_size, N, F_in, T)
#         batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
#
#         lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
#         # x:(B, N, F_in, T) -> (B, T, F_in, N)
#         # (B, T, F_in, N)(N) -> (B,T,F_in)
#         # (B,T,F_in)(F_in,N)->(B,T,N)
#
#         rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)
#
#         product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
#
#         E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
#         E_normalized = F.softmax(E, dim=1)  # 没问题
#         output = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), E_normalized).reshape(
#             batch_size, num_of_vertices, num_of_features, num_of_timesteps)
#
#         output = output.permute(0, 2, 1, 3)
#         output = self.bn(output)
#
#         return output


######################################################################
# Transformer layer
######################################################################
class LayerNorm(nn.Module):
    """
    Layer normalization.
    """

    def __init__(self, d_model, epsilon=1e-8):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.size = d_model

        self.gamma = nn.Parameter(torch.ones(self.size))
        self.beta = nn.Parameter(torch.ones(self.size))

    def forward(self, x):
        """
        :param x: [bs, T, d_model]
        :return:
        """
        normalized = (x - x.mean(dim=-1, keepdim=True)) \
                     / ((x.std(dim=-1, keepdim=True) + self.epsilon) ** .5)
        output = self.gamma * normalized + self.beta

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    """

    :param q: [bs, heads, seq_len, d_k]
    :param k:
    :param v:
    :param d_k: dim
    :param mask:
    :param dropout:
    :return:
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    seq_len = q.size(2)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads=4, dropout=0.1):
        """

        :param d_model: input feature dimension?
        :param heads:
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads  # narrow multi-head
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, attn_mask=None):
        """
        :param q: [bs, T, d_model]
        :param k:
        :param v:
        :param attn_mask:
        :return: [bs, T, d_model]
        """
        batch_size = q.size(0)

        # perform linear operation and split into N heads
        q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k)
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k)

        # transpose to get dimensions batch_size * heads * seq_len * d_k
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention
        scores = attention(q, k, v, self.d_k, attn_mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(concat)

        return output


class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.1):
        super(Feedforward, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        :param x: [bs, T, d_model]
        :return: [bs, T, d_model]
        """
        x = self.dropout(F.relu(self.linear_1(x)))
        output = self.linear_2(x)

        return output


class PositionalEncoder(nn.Module):
    """
    add position embedding
    hyper-network包含多个transformer layer，所以需要采用固定pe
    """

    def __init__(self, d_model, max_seq_len=12, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.learnable_pe = False
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        if self.learnable_pe:
            self.pe = nn.Parameter(torch.zeros(1, max_seq_len, self.d_model))
        else:
            # create constant 'pe' matrix with values dependent on pos and i
            pe = torch.zeros(max_seq_len, d_model)
            for pos in range(max_seq_len):
                for i in range(0, d_model, 2):
                    pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                    pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * (i + 1) / d_model)))
            pe = pe.unsqueeze(0)  # [bs, T, dim]
            self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [bs, T, dim]
        :return:
        """
        if self.learnable_pe:
            pe = Variable(self.pe[:, :x.size(1)], requires_grad=True)  # learnable position embedding
        else:
            # # make embeddings relatively larger
            # x = x * math.sqrt(self.d_model)  # why
            pe = Variable(self.pe[:, :x.size(1)], requires_grad=False)  # fixed position embedding

        if x.is_cuda:
            pe = pe.cuda()
        x = x + pe

        return self.dropout(x)


# class TransformerLayer(nn.Module):
#     """
#     transformer layer with 4 heads
#     """
#
#     def __init__(self, d_model, heads=4, dropout=0.1, max_seq_len=12):
#         super(TransformerLayer, self).__init__()
#         self.use_library_function = False
#         self.block_size = max_seq_len  # block_size是这个意思吧？
#
#         self.norm_1 = LayerNorm(d_model)
#         self.norm_2 = LayerNorm(d_model)
#         self.norm_3 = LayerNorm(d_model)  # 测试
#         self.dropout_1 = nn.Dropout(dropout)
#         self.dropout_2 = nn.Dropout(dropout)
#         self.ff = Feedforward(d_model, dropout=dropout)
#         self.pe = PositionalEncoder(d_model, max_seq_len, dropout)
#
#         if self.use_library_function:
#             self.attn = nn.MultiheadAttention(d_model, heads, dropout)
#         else:
#             self.attn = MultiHeadAttention(d_model, heads, dropout)
#
#         # causal mask
#         self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size))
#                              .view(1, self.block_size, self.block_size))  # 下三角
#
#     def forward(self, x):
#         """
#         输入多了207这一维，需要reshape为64*207，最后再reshape回来
#         :param x: [batch_size, f_in, N, T]
#         :return: [batch_size, f_in, N, T]
#         """
#         batch_size = x.size(0)
#
#         x = x.permute(0, 2, 3, 1)  # [64, 207, 12/1, 32]
#         x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [64*207, 12/1, 32]
#
#         x = self.pe(x)  # plus position embedding
#
#         # # standard transformer
#         # if self.use_library_function:  # 使用torch自带的multi-head attention
#         #     x = x.transpose(0, 1)  # [12/1, 64*207, 32]
#         #     mask = torch.tril(torch.ones(x.size(0), x.size(0)))
#         #     if x.is_cuda:
#         #         mask = mask.cuda()
#         #     mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         #     x = x + self.dropout_1(self.attn(x, x, x, attn_mask=mask)[0])
#         #     x = x.transpose(0, 1)
#         # else:
#         #     x = x + self.dropout_1(self.attn(x, x, x, attn_mask=self.mask))  # dropout有必要吗？
#         # x = self.norm_1(x)
#         # x = x + self.dropout_2(self.ff(x))
#         # x = self.norm_2(x)  # [64*207, T, d_model]
#
#         # variants
#         x2 = self.norm_1(x)
#         x = x + self.dropout_1(self.attn(x2, x2, x2, attn_mask=self.mask))
#         x2 = self.norm_2(x)
#         x = x + self.dropout_2(self.ff(x2))
#         x = self.norm_3(x)
#
#         output = x.reshape(batch_size, -1, x.size(-2), x.size(-1))
#         output = output.permute(0, 3, 1, 2)
#
#         return output


######################################################################
# Informer encoder layer
######################################################################
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(DEVICE)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=3, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        :param Q: [b, heads, T, d_k]
        :param K: 采样的K? 长度为Ln(L_K)?
        :param sample_k: c*ln(L_k), set c=3 for now
        :param n_top: top_u queries?
        :return: Q_K and Top_k query index
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

        # kernel_size = 3
        # pad = (kernel_size - 1) // 2
        # self.query_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        # self.key_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        # self.value_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # queries = queries.transpose(-1, 1)
        # keys = keys.transpose(-1, 1)
        # values = values.transpose(-1, 1)
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class InformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(InformerLayer, self).__init__()
        # d_ff = d_ff or 4*d_model
        self.attention = AttentionLayer(
            ProbAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.pe = PositionalEmbedding(d_model)
        self.d_model = d_model

    def forward(self, x, attn_mask=None):
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [64, 207, 12, 32]
        x = x.reshape(-1, T, C)  # [64*207, 12, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        # x = x * math.sqrt(self.d_model)
        # x = x + self.pe(x)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, T, C)
        output = output.permute(0, 3, 1, 2)

        return output


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(TransformerLayer, self).__init__()
        # d_ff = d_ff or 4*d_model
        self.attention = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.pe = PositionalEmbedding(d_model)
        self.d_model = d_model

    def forward(self, x, attn_mask=None):
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [64, 207, 12, 32]
        x = x.reshape(-1, T, C)  # [64*207, 12, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        # x = x * math.sqrt(self.d_model)
        # x = x + self.pe(x)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, T, C)
        output = output.permute(0, 3, 1, 2)

        return output


######################################################################
# GCN
######################################################################
def scaled_Laplacian(W):
    """
    compute \tilde{L}
    :param W: adj_mx
    :return: scaled laplacian matrix
    """
    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real  # k largest real part of eigenvalues

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    """
    compute a list of chebyshev polynomials from T_0 to T{K-1}
    :param L_tilde: scaled laplacian matrix
    :param K: the maximum order of chebyshev polynomials
    :return: list(np.ndarray), length: K, from T_0 to T_{K-1}
    """
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i-1] - cheb_polynomials[i-2])

    return cheb_polynomials


class GCNLayer(nn.Module):
    """
    K-order chebyshev graph convolution layer
    """

    def __init__(self, K, cheb_polynomials, c_in, c_out):
        """
        :param K: K-order
        :param adj_mx: original Adjacency matrix
        :param c_in: size of input channel
        :param c_out: size of output channel
        """
        super(GCNLayer, self).__init__()
        self.K = K
        self.c_in = c_in
        self.c_out = c_out
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out, affine=False)

        self.cheb_polynomials = cheb_polynomials
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList(  # weight matrices
            [nn.Parameter(torch.randn(c_in, c_out).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return: [batch_size, f_out, N, T]
        """
        x = self.relu(x)
        x = x.transpose(1, 2)  # [batch_size, N, f_in, T]

        batch_size, num_nodes, c_in, timesteps = x.shape

        outputs = []
        for step in range(timesteps):
            graph_signal = x[:, :, :, step]  # [b, N, f_in]
            output = torch.zeros(
                batch_size, num_nodes, self.c_out).to(self.DEVICE)  # [b, N, f_out]

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # [N, N]
                theta_k = self.Theta[k]
                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        outputs = torch.cat(outputs, dim=-1).transpose(1, 2)
        outputs = self.bn(outputs)
        # outputs = F.relu(torch.cat(outputs, dim=-1)).transpose(1, 2)
        # outputs = F.dropout(outputs, 0.5, training=self.training)
        return outputs


######################################################################
# 1-order gcn
######################################################################
def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, A_wave, c_in, c_out, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = c_in
        self.out_features = c_out
        A_wave = torch.from_numpy(A_wave).to(DEVICE)
        self.A_wave = A_wave
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        :param input: [batch_size, f_in, N, T]
        :param A_wave: Normalized adjacency matrix.
        :return:
        """
        x = input.permute(0, 2, 3, 1)  # [B, N, T, F]
        # x = self.relu(x)
        lfs = torch.einsum("ij,jklm->kilm", [self.A_wave, x.permute(1, 0, 2, 3)])
        output = F.relu(torch.matmul(lfs, self.weight))
        # output = (torch.matmul(lfs, self.weight))

        if self.bias is not None:
            output = output + self.bias

        output = output.permute(0, 3, 1, 2)
        # output = self.bn(output)

        return output


######################################################################
# Diffusion convolution layer
######################################################################
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


class DiffusionConvLayer(nn.Module):
    """
    K-order diffusion convolution layer with self-adaptive adjacency matrix (N, N)
    """

    def __init__(self, K, supports, nodevec1, nodevec2, c_in, c_out):
        super(DiffusionConvLayer, self).__init__()
        c_in = (K * (len(supports) + 1) + 1) * c_in
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.mlp = linear(c_in, c_out).to(DEVICE)  # 7 * 32 * 32
        self.c_out = c_out
        self.K = K
        self.supports = supports
        self.nconv = nconv()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out, affine=False)

    def forward(self, x):
        x = self.relu(x)

        # adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # bug?
        adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)
        new_supports = self.supports + [adp]

        out = [x]
        for a in new_supports:
            # x.shape [b, dim, N, seq_len]
            # a.shape [b, N, N]
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.K + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = self.bn(h)
        # h = F.dropout(h, 0.2, training=self.training)

        return h


######################################################################
# Diffusion convolution layer with
# dynamic self-adaptive adjacency matrix in temporal dimension
######################################################################
class Diff_gcn(nn.Module):
    def __init__(self, K, supports, nodevec1, nodevec2, alpha, c_in, c_out):
        """
        diffusion gcn with self-adaptive adjacency matrix (T, N, N)
        :param K:
        :param supports:
        :param c_in:
        :param c_out:
        """
        super(Diff_gcn, self).__init__()
        self.nconv = nconv()
        c_in = (K * len(supports) + 1) * c_in
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.mlp = linear(c_in, c_out)
        self.K = K
        self.supports = [s.unsqueeze(0).repeat(12, 1, 1) for s in supports]
        self.alpha = alpha
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out, affine=False)

    def forward(self, x):
        """
        :param x: [b, C, N, T]
        :return:
        """
        x = self.relu(x)

        adp = F.relu(torch.matmul(self.nodevec1, self.nodevec2))
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp==0, mask, adp)
        adp = F.softmax(adp, dim=-1)  # [T, N, N]


        out = [x]
        x = x.permute(0, 3, 1, 2)  # (b, T, C, N)
        for i in range(len(self.supports)):
            alpha, beta = F.softmax(self.alpha[i])
            a = alpha * self.supports[i] + beta * adp
            x1 = torch.matmul(x, a)  # (b, T, C, N)(T, N, N)->
            out.append(x1.permute(0, 2, 3, 1))
            for k in range(2, self.K + 1):
                x2 = torch.matmul(x1, a)
                out.append(x2.permute(0, 2, 3, 1))
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = self.bn(h)
        # h = F.dropout(h, 0.3, training=self.training)  # necessary?

        return h


######################################################################
# Spatial Transformer
######################################################################
class SpatialTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(SpatialTransformerLayer, self).__init__()
        self.attention = SpatialAttentionLayer(
            SpatialFullAttention(attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.d_model = d_model

    def forward(self, x):
        b, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1)  # [64, 12, 207, 32]
        x = x.reshape(-1, N, C)  # [64*12, 207, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, N, C)
        output = output.permute(0, 3, 2, 1)

        return output


class SpatialInformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(SpatialInformerLayer, self).__init__()
        self.attention = SpatialAttentionLayer(
            SpatialProbAttention(attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.pe = PositionalEmbedding(d_model)
        self.d_model = d_model

    def forward(self, x):
        b, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1)  # [64, 12, 207, 32]
        x = x.reshape(-1, N, C)  # [64*12, 207, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, N, C)
        output = output.permute(0, 3, 2, 1)

        return output


class SpatialAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(SpatialAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        # shape=[b*T, N, C]
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class SpatialFullAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(SpatialFullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # 在这里加上fixed邻接矩阵？
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class SpatialProbAttention(nn.Module):
    def __init__(self, factor=3, scale=None, attention_dropout=0.1, output_attention=False):
        super(SpatialProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        :param Q: [b, heads, T, d_k]
        :param K: 采样的K? 长度为Ln(L_K)?
        :param sample_k: c*ln(L_k), set c=3 for now
        :param n_top: top_u queries?
        :return: Q_K and Top_k query index
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        # V_sum = V.sum(dim=-2)
        V_sum = V.mean(dim=-2)  # # [256*12, 4, 8]
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()  # [256*12, 4, 207, 8]
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores) [256*12, 4, 18, 207]

        # print(context_in.shape)  # [256*12, 4, 207, 8]
        # print(torch.matmul(attn, V).shape)  # [256*12, 4, 18, 8]
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        # print(index.shape)  # [256*12, 4, 18]

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale  # [256*12, 4, 18, 207] 18=sqrt(207)*3
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        return context.transpose(2, 1).contiguous(), attn
