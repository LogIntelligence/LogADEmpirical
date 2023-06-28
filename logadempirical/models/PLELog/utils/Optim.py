# -*- coding: utf-8 -*-
import torch.optim
from torch.nn.utils.clip_grad import clip_grad_norm
import random
import numpy as np




# Setup optimizer (should always come after model.cuda())
# iterable of dicts for per-param options where each dict
# is {'params' : [p1, p2, p3...]}.update(generic optimizer args)
# Example:
# optim.SGD([
# {'params': model.base.parameters()},
# {'params': model.classifier.parameters(), 'lr': 1e-3}
# ], lr=1e-2, momentum=0.9)



class Optimizer(object):
    # Class dict to map lowercase identifiers to actual classes
    methods = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'asgd': torch.optim.ASGD,
        'rprop': torch.optim.Rprop,
        'rmsprop': torch.optim.RMSprop,
    }

    @staticmethod
    def get_params(model):
        """Returns all name, parameter pairs with requires_grad=True."""
        return list(
            filter(lambda p: p[1].requires_grad, model.named_parameters()))

    def __init__(self,
                 name,
                 model,
                 lr=0,
                 weight_decay=0,
                 grad_clip=None,
                 optim_args=None,
                 **kwargs):
        """
        :param decay_method: Method of learning rate decay.

        """

        self.name = name
        self.model = model
        self.init_lr = lr
        self.weight_decay = weight_decay
        self.gclip = grad_clip

        self._count = 0

        # pass external optimizer configs
        if optim_args is None:
            optim_args = {}

        self.optim_args = optim_args

        # If an explicit lr given, pass it to torch optimizer
        if self.init_lr > 0:
            self.optim_args['lr'] = self.init_lr

        # Get all parameters that require grads
        self.named_params = self.get_params(self.model)

        # Filter out names for gradient clipping
        self.params = [param for (name, param) in self.named_params]

        if self.weight_decay > 0:
            weight_group = {
                'params': [p for n, p in self.named_params if 'bias' not in n],
                'weight_decay': self.weight_decay,
            }
            bias_group = {
                'params': [p for n, p in self.named_params if 'bias' in n],
            }
            self.param_groups = [weight_group, bias_group]

        else:
            self.param_groups = [{'params': self.params}]

        # Safety check
        n_params = len(self.params)
        for group in self.param_groups:
            n_params -= len(group['params'])
        assert n_params == 0, "Not all params are passed to the optimizer."

        # Create the actual optimizer
        self.optim = self.methods[self.name](self.param_groups,
                                             **self.optim_args)

        # Assign shortcuts
        self.zero_grad = self.optim.zero_grad

        # Skip useless if evaluation logic if gradient_clip not requested
        if self.gclip == 0 or self.gclip is None:
            self.step = self.optim.step

    def zero_grad(self):
        self.optim.zero_grad()

    def step(self, closure=None):
        """Gradient clipping aware step()."""
        if self.gclip is not None and self.gclip > 0:
            clip_grad_norm(self.params, self.gclip)
        self.optim.step(closure)

    def rescale_lrate(self, scale, min_lrate=-1.0):
        if isinstance(scale, list):
            for scale_, group in zip(scale, self.optim.param_groups):
                group['lr'] = max(group['lr'] * scale_, min_lrate)
        else:
            for group in self.optim.param_groups:
                group['lr'] = max(group['lr'] * scale, min_lrate)

    def get_lrate(self):
        for group in self.optim.param_groups:
            yield group['lr']

    def set_lrate(self, lr):
        if isinstance(lr, list):
            for lr_, group in zip(lr, self.optim.param_groups):
                group['lr'] = lr_
        else:
            for group in self.optim.param_groups:
                group['lr'] = lr

    def __repr__(self):
        s = "Optimizer => {} (lr: {}, weight_decay: {}, g_clip: {})".format(
            self.name, self.init_lr, self.weight_decay, self.gclip)
        return s
