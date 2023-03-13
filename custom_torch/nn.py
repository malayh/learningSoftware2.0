from __future__ import  annotations
import torch

from functools import cached_property


class Module:
    def __call__(self, x):
        raise NotImplementedError

    @cached_property
    def parameters(self):
        raise NotImplementedError



class Linear(Module):
    def __init__(self,fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    @cached_property
    def parameters(self):
        return [self.weight] + [self.bias] if self.bias is not None else [self.weight]

class Tanh(Module):
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    @cached_property
    def parameters(self):
        return []

class BatchNorm(Module):
    def __init__(self, fan_out, eps=1e-5, momentum=0.01):
        self.eps = eps
        self.momentum = momentum

        self.training = True

        # trainable parameters
        self.scale_by = torch.ones(fan_out)
        self.shift_by = torch.zeros(fan_out)

        # buffers
        self.running_mean = torch.zeros(fan_out)
        self.running_var = torch.ones(fan_out)

    def __call__(self, x):
        if self.training:
            # The batch norm assumes that x is 2d. But if its 3d we have calculate the mean and variance over the first two dimensions
            dim = None # so that it throws an error if x.ndim is not 2 or 3
            if x.ndim == 3:
                dim = (0, 1)
            elif x.ndim == 2:
                dim = 0

            mean = x.mean(dim, keepdim=True)
            var = x.var(dim, keepdim=True, unbiased=True)

            with torch.no_grad():
                self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / (var + self.eps).sqrt()
        self.out = self.scale_by * x_hat + self.shift_by

        return self.out

    @cached_property
    def parameters(self):
        return [self.scale_by, self.shift_by]

class Embedding(Module):
    def __init__(self, embedding_size, embedding_dimension):
        self.weight = torch.randn((embedding_size, embedding_dimension))

    def __call__(self, x):
        self.out = self.weight[x]
        return self.out

    @cached_property
    def parameters(self):
        return [self.weight]

class FlattenConsecutive(Module):
    def __init__(self, n_consecutive):
        self.n_consecutive = n_consecutive

    def __call__(self, x):
        batch, seq_len, rest = x.shape

        x = x.view(batch, seq_len // self.n_consecutive, self.n_consecutive * rest)   
        if x.shape[1] == 1:
            x = x.squeeze(1)

        self.out = x
        return self.out

    @cached_property
    def parameters(self):
        return []

class Sequential(Module):
    def __init__(self, *layers: list):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    @cached_property
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters]