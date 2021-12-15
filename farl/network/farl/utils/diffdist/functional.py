from . import modules as mods
import torch.distributed as dist


def consume_variable(tensor_to_consume, tensors_to_return, set_ones_grad=True):
    return mods.ConsumeVariable(set_ones_grad)(tensor_to_consume,
                                               *tensors_to_return)


def send(tensor, dst, group=dist.group.WORLD, tag=0):
    return mods.Send(dst, group, tag)(tensor)


def recv(tensor,
         src=None,
         group=dist.group.WORLD,
         tag=0,
         next_backprop=None,
         inplace=True):
    return mods.Recv(src, group, tag, next_backprop, inplace)(tensor)


def broadcast(tensor,
              src,
              group=dist.group.WORLD,
              next_backprop=None,
              inplace=True):
    return mods.Broadcast(src, group, next_backprop, inplace)(tensor)


def gather(tensor,
           gather_list=None,
           dst=None,
           group=dist.group.WORLD,
           next_backprop=None,
           inplace=True):
    return mods.Gather(dst, group, next_backprop, inplace)(tensor, gather_list)


def scatter(tensor,
            scatter_list=None,
            src=None,
            group=dist.group.WORLD,
            next_backprop=None,
            inplace=True):
    return mods.Scatter(src, group, next_backprop, inplace)(tensor,
                                                            scatter_list)


def all_gather(gather_list,
               tensor,
               group=dist.group.WORLD,
               next_backprop=None,
               inplace=True):
    return mods.AllGather(group, next_backprop, inplace)(gather_list, tensor)
