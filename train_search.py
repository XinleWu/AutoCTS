import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from model_search import Network
from architect import Architect
from utils import load_adj, load_dataset, masked_mae, masked_rmse, metric, Temp_Scheduler, get_adj_matrix, generate_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='AutoCTS')
parser.add_argument('--adj_mx', type=str, default='data/METR-LA/adj_mx.pkl',
                    help='location of the data')
parser.add_argument('--data', type=str, default='data/METR-LA',
                    help='location of the adjacency matrix')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=358)
parser.add_argument('--hid_dim', type=int, default=32,
                    help='for residual_channels and dilation_channels')
parser.add_argument('--randomadj', type=bool, default=True,
                    help='whether random initialize adaptive adj')
parser.add_argument('--seq_len', type=int, default=12)
parser.add_argument('--layers', type=int, default=4, help='number of cells')
parser.add_argument('--steps', type=int, default=4, help='number of nodes of a cell')
parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
parser.add_argument('--lr_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--unrolled', action='store_true', default=False,
                    help='use one-step unrolled validation loss')  # First-order Approximation or not
parser.add_argument('--grad_clip', type=float, default=5,
                    help='gradient clipping')
parser.add_argument('--arch_lr', type=float, default=3e-4,
                    help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3,
                    help='weight decay for arch encoding')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--temp', type=float, default=5.0,
                    help='initial softmax temperature')
parser.add_argument('--temp_min', type=float, default=0.001,
                    help='minimal softmax temperature')
args = parser.parse_args()


def train(train_iterator, val_iterator, model, architect, optimizer, lr, scaler, epoch):
    model = model.train()

    train_loss = []
    train_rmse = []
    for i, (x, y) in enumerate(train_iterator):
        x = torch.Tensor(x).to(DEVICE)
        x = x.transpose(1, 3)
        y = torch.Tensor(y).to(DEVICE)
        y = y.transpose(1, 3)[:, 0, :, :]

        # get a random minibatch from the search queue with replacement
        x_search, y_search = next(iter(val_iterator))
        x_search = torch.Tensor(x_search).to(DEVICE)
        x_search = x_search.transpose(1, 3)
        y_search = torch.Tensor(y_search).to(DEVICE)
        y_search = y_search.transpose(1, 3)[:, 0, :, :]  # [64, 207, 12]

        # update alpha
        if epoch >= 15:
            architect.step(x, y, x_search, y_search, lr, optimizer, unrolled=args.unrolled)

        # update w
        optimizer.zero_grad()
        logits = model(x, i)
        logits = logits.transpose(1, 3)

        y = torch.unsqueeze(y, dim=1)
        predict = scaler.inverse_transform(logits)
        loss = masked_mae(predict, y, 0.0)
        rmse = masked_rmse(predict, y, 0.0)

        train_loss.append(loss.item())
        train_rmse.append(rmse.item())
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        if i % 20 == 0:
            print(f'train_loss: {train_loss[-1]}, train_rmse: {train_rmse[-1]}')

    return train_loss[-1], train_rmse[-1]


def infer(val_iterator, model, scaler):
    with torch.no_grad():
        model = model.eval()

        valid_loss = []
        valid_rmse = []
        for i, (x, y) in enumerate(val_iterator):
            x = torch.Tensor(x).to(DEVICE)
            x = x.transpose(1, 3)
            y = torch.Tensor(y).to(DEVICE)
            y = y.transpose(1, 3)[:, 0, :, :]  # [64, 207, 12]

            logits = model(x, -1)
            logits = logits.transpose(1, 3)  # [64, 1, 207, 12]

            y = torch.unsqueeze(y, dim=1)
            predict = scaler.inverse_transform(logits)

            loss = masked_mae(predict, y, 0.0)
            valid_loss.append(loss.item())
            rmse = masked_rmse(predict, y, 0.0)
            valid_rmse.append(rmse.item())

        print(f'val_loss: {np.mean(valid_loss)}, val_armse: {np.mean(valid_rmse)}')

    return np.mean(valid_loss), np.mean(valid_rmse)


def main():
    print(args)
    # lr scheduler: cosine annealing (fix lr now)
    # temp scheduler: exponential annealing (self-defined in utils)
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if torch.cuda:
    #     torch.cuda.manual_seed(args.seed)

    # # load dataset
    # _, _, adj_mx = load_adj(args.adj_mx)
    # dataloader = load_dataset(args.data, args.batch_size, args.batch_size)
    # scaler = dataloader['scaler']

    adj_mx = get_adj_matrix('data/pems/PEMS03/PEMS03.csv', args.num_nodes, id_filename='data/pems/PEMS03/PEMS03.txt')
    dataloader = generate_data('data/pems/PEMS03/PEMS03.npz', args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    model = Network(adj_mx, scaler, args)
    model.to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters', params)
    # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(
    #     model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, int(args.epochs), eta_min=args.lr_min)

    temp_scheduler = Temp_Scheduler(args.epochs, model._temp, args.temp, temp_min=args.temp_min)

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        t1 = time.time()
        # scheduler.step()
        # lr = scheduler.get_lr()[0]
        lr = args.lr
        model._temp = temp_scheduler.step()

        genotype = model.genotype()
        print(f'genotype: {genotype}')

        # alphas = model.arch_parameters()
        print(f'temperature: {model._temp}')
        # print(f'a: {F.softmax(alphas[0] / model._temp, dim=-1)}')

        train_iterator = dataloader['train_loader_1'].get_iterator()
        val_iterator_1 = dataloader['train_loader_2'].get_iterator()
        val_iterator_2 = dataloader['val_loader'].get_iterator()

        # training
        train_loss, train_armse = train(train_iterator, val_iterator_1, model,
                                        architect, optimizer, lr, scaler, epoch)

        # validation
        valid_loss, valid_armse = infer(val_iterator_2, model, scaler)
        print(f'search time: {time.time() - t1}')
        # save()


if __name__ == '__main__':
    main()
