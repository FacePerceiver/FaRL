import torch.nn as nn
import torch.distributed as dist
from . import functions as funcs


class ConsumeVariable(nn.Module):
    def __init__(self, set_ones_grad=True):
        """
        If set_ones_grad=True then the gradient w.r.t tensor_to_consume
        is set to 1 during backprop. Otherwise, it is set to 0.
        """
        super(ConsumeVariable, self).__init__()
        self.set_ones_grad = set_ones_grad

    def forward(self, tensor_to_consume, *tensors_to_return):
        tensors_to_return = funcs.ConsumeVariableFunc.apply(
            tensor_to_consume, self.set_ones_grad, *tensors_to_return)
        return tensors_to_return


class Send(nn.Module):
    def __init__(self, dst, group=dist.group.WORLD, tag=0):
        super(Send, self).__init__()
        self.dst = dst
        self.group = group
        self.tag = tag

    def forward(self, tensor):
        return funcs.SendFunc.apply(tensor, self.dst, self.group, self.tag)


class Recv(nn.Module):
    def __init__(self,
                 src=None,
                 group=dist.group.WORLD,
                 tag=0,
                 next_backprop=None,
                 inplace=True):
        super(Recv, self).__init__()
        self.next_backprop = next_backprop
        self.src = src
        self.group = group
        self.tag = tag
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, tensor):
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        tensor, sender = funcs.RecvFunc.apply(tensor, self.src, self.group,
                                              self.tag, self.inplace)
        return tensor, sender.item()


class Broadcast(nn.Module):
    def __init__(self,
                 src,
                 group=dist.group.WORLD,
                 next_backprop=None,
                 inplace=True):
        super(Broadcast, self).__init__()
        self.src = src
        self.group = group
        self.next_backprop = next_backprop
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, tensor):
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        return funcs.BroadcastFunc.apply(tensor, self.src, self.group,
                                         self.inplace)


class Gather(nn.Module):
    def __init__(self,
                 dst=None,
                 group=dist.group.WORLD,
                 next_backprop=None,
                 inplace=True):
        super(Gather, self).__init__()
        self.dst = dst
        self.group = group
        self.next_backprop = next_backprop
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, tensor, gather_list=None):
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        if dist.get_rank(self.group) == self.dst:
            return list(
                funcs.GatherFunc.apply(tensor, self.dst, self.group,
                                       self.inplace, *gather_list))
        else:
            return funcs.GatherFunc.apply(tensor, self.dst, self.group,
                                          self.inplace, None)


class Scatter(nn.Module):
    def __init__(self,
                 src=None,
                 group=dist.group.WORLD,
                 next_backprop=None,
                 inplace=True):
        super(Scatter, self).__init__()
        self.src = src
        self.group = group
        self.next_backprop = next_backprop
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, tensor, scatter_list=None):
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        if dist.get_rank(self.group) == self.src:
            return funcs.ScatterFunc.apply(tensor, self.src, self.group,
                                           self.inplace, *scatter_list)
        else:
            return funcs.ScatterFunc.apply(tensor, self.src, self.group,
                                           self.inplace, None)


class AllGather(nn.Module):
    def __init__(self,
                 group=dist.group.WORLD,
                 next_backprop=None,
                 inplace=True):
        super(AllGather, self).__init__()
        self.group = group
        self.next_backprop = next_backprop
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, gather_list, tensor):
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        return list(
            funcs.AllGatherFunc.apply(tensor, self.group, self.inplace,
                                      *gather_list))
