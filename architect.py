"""Update architecture encoding"""
import numpy as np
import torch
from torch.autograd import Variable


def _concat(xs):
    """
    将张量列表中每一个张量都转为一维，然后concat起来
    :param xs: 张量列表
    :return:
    """
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    """
    Update architecture encoding
    """

    def __init__(self, model, args):
        self.args = args
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(  # arch optimizer
            self.model.arch_parameters(),  # alpha
            lr=args.arch_lr,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay)

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        """
        Update architecture parameter.
        :param input_train: a batch of training data
        :param target_train:
        :param input_valid: a batch of validation data
        :param target_valid:
        :param eta: learning rate
        :param network_optimizer: optimizer for w
        :param unrolled: whether to apply first-order approximation
        :return:
        """
        self.optimizer.zero_grad()
        if unrolled:  # second order
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:  # first order
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()  # update alpha

    def _backward_step(self, input_valid, target_valid):
        """
        First-order Approximation
        compute dalpha{L_val(w, alpha)}
        对应的操作是在验证集上计算w参数对应的损失，然后用来更新alpha参数
        :param input_valid: a batch of validation data
        :param target_valid:
        :return:
        """
        loss = self.model.loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        """
        Second-order Approximation
        compute dalpha{L_val(w', alpha)} - eta * (dalpha{L_train(w+, alpha)} - dalpha{L_train(w-, alpha)}) / 2*epsilon
        where w' = w - eta * dtheta{L_train(w, alpha)}
        :param input_train: a batch of training data
        :param target_train:
        :param input_valid: a batch of validation data
        :param target_valid:
        :param eta: learning rate for w
        :param network_optimizer: optimizer for w
        :return:
        """
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)  # w' model
        unrolled_loss = unrolled_model.loss(input_valid, target_valid)  # L_val(w', alpha)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]  # dalpha{L_val(w', alpha)}
        vector = [v.grad.data for v in unrolled_model.parameters()]  # dw'{L_val(w', alpha)} related to epsilon
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)  # new alpha

        for v, g in zip(self.model.arch_parameters(), dalpha):
            # update alpha
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        """
        compute w' and return the model with w'
        w' = w - eta * dtheta{L_train(w, alpha)}
        """
        loss = self.model.loss(input, target)  # loss on training data
        theta = _concat(self.model.parameters()).data  # w
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data \
                 + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def _construct_model_from_theta(self, theta):
        """
        replace w with w'
        :param theta: w'
        :return:
        """
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        if self.args.cuda:
            return model_new.cuda()
        else:
            return model_new

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        """
        compute (dalpha{L_train(w+, alpha)} - dalpha{L_train(w-, alpha)}) / 2*epsilon
        w+ = w + epsilon * dw'{L_val(w', alpha)} = w + R * v
        w- = w - epsilon * dw'{L_val(w', alpha)} = w - R * v
        :return:
        """
        R = r / _concat(vector).norm()  # epsilon
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)  # w to w+
        loss = self.model.loss(input, target)  # L_train(w+, alpha)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())  # dalpha{L_train(w+, alpha)}

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)  # w+ to w-
        loss = self.model.loss(input, target)  # L_train(w-, alpha)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())  # dalpha{L_train(w-, alpha)}

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)  # w- to w

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
