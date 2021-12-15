import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Gumbel

from genotypes import PRIMITIVES, Genotype
from operations import OPS, linear, ReLUConvBN, Identity, asym_adj, \
    scaled_Laplacian, cheb_polynomial, PositionalEmbedding
from utils import masked_mae

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def channel_shuffle(x, groups):
    bs, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.contiguous().view(bs, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(bs, -1, height, width)

    return x


class MixedOp(nn.Module):
    """
    Compute the weighted output of candidate operations on an edge
    """

    def __init__(self, C, supports, nodevec1, nodevec2, alpha, cheb):
        """
        :param C: hid_dim
        :param supports: supports for diff_gcn
        """
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.k = 4  # channel proportion

        for primitive in PRIMITIVES:
            if 'diff_gcn' in primitive:
                op = OPS[primitive](C // self.k, supports, nodevec1, nodevec2)
            elif 'cheb_gcn' in primitive:
                op = OPS[primitive](C // self.k, cheb, nodevec1, nodevec2, alpha)
            else:
                op = OPS[primitive](C // self.k)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        :param x: [bsize, in_dim, num_nodes, seq_len]?
        :param weights: alpha, [num_edges, num_ops]
        """
        dim_2 = x.shape[1]
        xtemp = x[:, :dim_2 // self.k, :, :]
        xtemp2 = x[:, dim_2 // self.k:, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        ans = torch.cat([temp1, xtemp2], dim=1)
        ans = channel_shuffle(ans, self.k)

        return ans


class Cell(nn.Module):

    def __init__(self, supports, nodevec1, nodevec2, alpha, cheb, C, steps):
        """
                :param supports: supports for diff_gcn
                :param C: hidden units number
                :param steps: nodes number of a cell
                """
        super(Cell, self).__init__()

        # self.preprocess = ReLUConvBN(C_prev, C, (1, 1), 1)
        self.preprocess = Identity()  # 先试试Identity
        self._steps = steps

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(1 + i):
                op = MixedOp(C, supports, nodevec1, nodevec2, alpha, cheb)
                self._ops.append(op)

    def forward(self, s_prev, weights, weights2):
        """

        :param s_prev:
        :param weights: for operations
        :param weights2: for edge normalization
        :return:
        """
        s0 = self.preprocess(s_prev)

        states = [s0]
        offset = 0
        for i in range(self._steps):
            s = sum(weights2[offset + j] * self._ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return states[-1]


class Network(nn.Module):

    def __init__(self, adj_mx, scaler, args):
        super(Network, self).__init__()
        self._adj_mx = adj_mx
        self._args = args
        self._scaler = scaler
        self._layers = args.layers
        self._steps = args.steps
        self._temp = args.temp

        # for diff_gcn
        new_adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
        supports = [torch.tensor(i).to(DEVICE) for i in new_adj]

        # for cheb_gcn
        L_tilde = scaled_Laplacian(adj_mx)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, 2)]

        # adaptive adjacency matrix
        self.alpha = nn.Parameter(torch.randn(len(supports), len(supports)).to(DEVICE), requires_grad=True).to(DEVICE)
        if args.randomadj:
            aptinit = None
        else:
            aptinit = supports[0]
        if aptinit is None:
            self.nodevec1 = nn.Parameter(torch.randn(args.num_nodes, 10).to(DEVICE), requires_grad=True).to(DEVICE)
            self.nodevec2 = nn.Parameter(torch.randn(10, args.num_nodes).to(DEVICE), requires_grad=True).to(DEVICE)
        else:
            m, p, n = torch.svd(aptinit)
            initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
            self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(DEVICE)
            self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(DEVICE)

        C_prev, C_curr = args.hid_dim, args.hid_dim

        self.cells = nn.ModuleList()
        self.skip_connect = nn.ModuleList()
        for i in range(self._layers):
            cell = Cell(supports, self.nodevec1, self.nodevec2, self.alpha, cheb_polynomials, C_curr, self._steps)
            self.cells += [cell]
            # C_prev = C_curr * self._steps
            self.skip_connect.append(nn.Conv1d(C_prev, args.hid_dim * 8, (1, 1)))

        # input layer
        self.start_linear = linear(args.in_dim, args.hid_dim)

        # output layer
        self.end_linear_1 = linear(c_in=args.hid_dim * 8, c_out=args.hid_dim * 16)
        self.end_linear_2 = linear(c_in=args.hid_dim * 16, c_out=args.seq_len)

        # position encoding
        self.pe = PositionalEmbedding(args.hid_dim)

        self._initialize_alphas()

    def forward(self, input, count):
        x = self.start_linear(input)

        b, D, N, T = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, T, D)  # [64, 207, 12, 32]
        x = x * math.sqrt(self._args.hid_dim)
        x = x + self.pe(x)
        x = x.reshape(b, -1, T, D).permute(0, 3, 1, 2)

        skip = 0
        prev = [x]
        n2 = 1  # for inter-cell
        start2 = 0
        for i, cell in enumerate(self.cells):
            n = 2
            start = 1
            weights = F.softmax(self.alphas[i] / self._temp, dim=-1)
            weights2 = F.softmax(self.betas[i][0:1], dim=-1)
            weights3 = F.softmax(self.gamma[start2:start2 + n2] / self._temp, dim=-1)  # one-hot?
            if count == 0 and i == 0:
                print(f'alpha: {self.alphas}')
                print(f'weights: {weights}')
            if count == 0:
                print(f'gamma: {weights3}')
            start2 += n2
            n2 += 1
            for j in range(self._steps - 1):
                end = start + n
                tw2 = F.softmax(self.betas[i][start:end], dim=-1)
                start = end
                n += 1
                weights2 = torch.cat([weights2, tw2], dim=0)
            x = sum(prev[k] * weights3[k] for k in range(len(weights3)))
            x = cell(x, weights, weights2)
            prev.append(x)
            skip = self.skip_connect[i](x) + skip

        state = torch.max(F.relu(skip), dim=-1, keepdim=True)[0]
        out = F.relu(self.end_linear_1(state))
        logits = self.end_linear_2(out)

        return logits

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(1 + i))  # 计算DAG的edge_num
        k2 = sum(1 for i in range(self._layers) for n in range(1 + i))
        num_ops = len(PRIMITIVES)

        self.alphas = Variable(1e-3 * torch.randn(self._layers, k, num_ops).to(DEVICE), requires_grad=True)
        self.betas = Variable(1e-3 * torch.randn(self._layers, k).to(DEVICE), requires_grad=True)
        self.gamma = Variable(1e-3 * torch.randn(k2).to(DEVICE), requires_grad=True)  # inter-cell
        self._arch_parameters = [self.alphas, self.betas, self.gamma]

    def arch_parameters(self):
        return self._arch_parameters

    def loss(self, x, y):
        """
        compute loss with current w
        :param x: transformed x
        :param y: transformed y
        :return:
        """
        logits = self(x, -1)
        logits = logits.transpose(1, 3)
        y = torch.unsqueeze(y, dim=1)
        predict = self._scaler.inverse_transform(logits)
        loss = masked_mae(predict, y, 0.0)

        return loss

    def new(self):
        """
        new model with same alpha as the original model
        :return:
        """
        model_new = Network(self._adj_mx, self._scaler, self._args).to(DEVICE)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def genotype(self):
        """
        derive for a cell
        :return:
        """

        def _parse(weights, weights2):
            gene = []
            n = 1
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]

                max_edge = sorted(
                    range(i),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:1]
                edges = max_edge + [i]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene = []
        for i, cell in enumerate(self.cells):
            n = 2
            start = 1
            weights2 = F.softmax(self.betas[i][0:2], dim=-1)
            for _ in range(self._steps - 1):
                end = start + n
                tw2 = F.softmax(self.betas[i][start:end], dim=-1)
                start = end
                n += 1
                weights2 = torch.cat([weights2, tw2], dim=0)

            gene.append(_parse(F.softmax(self.alphas[i], dim=-1).data.cpu().numpy(),
                               weights2.data.cpu().numpy()))
        concat = range(1 + self._steps - 4, self._steps + 1)
        genotype = Genotype(gene, concat)

        return genotype
