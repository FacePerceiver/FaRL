from torch.autograd import Function
from . import extra_collectives as dist_extra
import torch.distributed as dist
import torch


class ConsumeVariableFunc(Function):
    @staticmethod
    def forward(ctx, tensor_to_consume, set_ones_grad, *tensors_to_return):
        ctx.save_for_backward(tensor_to_consume)
        ctx.set_ones_grad = set_ones_grad
        return tensors_to_return

    @staticmethod
    def backward(ctx, *grad_outputs):
        tensor_to_consume, = ctx.saved_tensors
        if ctx.set_ones_grad:
            fake_grad = torch.ones_like(tensor_to_consume)
        else:
            fake_grad = torch.zeros_like(tensor_to_consume)

        return (fake_grad, None) + grad_outputs


class SendFunc(Function):
    @staticmethod
    def forward(ctx, tensor, dst, group=dist.group.WORLD, tag=0):
        ctx.save_for_backward(tensor)
        ctx.dst = dst
        ctx.group = group
        ctx.tag = tag
        dist.send(tensor, dst, group, tag)
        return tensor.new_tensor([])

    @staticmethod
    def backward(ctx, grad_output):
        tensor, = ctx.saved_tensors
        # TODO: Add ctx.needs_input_grad check
        grad_tensor = torch.zeros_like(tensor)
        dist.recv(grad_tensor, ctx.dst, ctx.group, ctx.tag)

        return grad_tensor, None, None, None


class RecvFunc(Function):
    @staticmethod
    def forward(ctx,
                tensor,
                src=None,
                group=dist.group.WORLD,
                tag=0,
                inplace=True):
        if not inplace:
            tensor = torch.zeros_like(tensor).requires_grad_(False)
        ctx.src = src
        ctx.group = group
        ctx.tag = tag
        sender = dist.recv(tensor, src, group, tag)
        if src:
            assert sender == src
        else:
            ctx.src = sender
        sender = torch.tensor(sender)
        ctx.mark_non_differentiable(sender)
        return tensor, sender

    @staticmethod
    def backward(ctx, grad_tensor, grad_sender):
        dist.send(grad_tensor, ctx.src, ctx.group, ctx.tag)
        return grad_tensor, None, None, None, None


class BroadcastFunc(Function):
    @staticmethod
    def forward(ctx, tensor, src, group=dist.group.WORLD, inplace=True):
        ctx.src = src
        ctx.group = group
        if dist.get_rank(group) == src:
            if not inplace:
                with torch.no_grad():
                    tensor = tensor.clone().requires_grad_(False)
        else:
            if not inplace:
                tensor = torch.zeros_like(tensor).requires_grad_(False)
        dist.broadcast(tensor, src, group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        dist.reduce(grad_output,
                    ctx.src,
                    op=dist.ReduceOp.SUM,
                    group=ctx.group)
        return grad_output, None, None, None


class AllReduceFunc(Function):
    @staticmethod
    def forward(ctx, i):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class ReduceFunc(Function):
    @staticmethod
    def forward(ctx, i):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class AllGatherFunc(Function):
    @staticmethod
    def forward(ctx, tensor, group, inplace, *gather_list):
        ctx.save_for_backward(tensor)
        ctx.group = group
        gather_list = list(gather_list)
        if not inplace:
            gather_list = [torch.zeros_like(g) for g in gather_list]
        dist.all_gather(gather_list, tensor, group)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        # print("rank {} grads before scatter & reduce: {}".format(dist.get_rank(ctx.group), list(grads)))
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        dist_extra.reduce_scatter(grad_out, list(grads), group=ctx.group)
        return (grad_out, None, None) + grads


class GatherFunc(Function):
    @staticmethod
    def forward(ctx, input, dst, group, inplace, *gather_list):
        ctx.dst = dst
        ctx.group = group
        ctx.save_for_backward(input)
        if dist.get_rank(group) == dst:
            gather_list = list(gather_list)
            if not inplace:
                gather_list = [torch.zeros_like(g) for g in gather_list]
            dist.gather(input, gather_list=gather_list, dst=dst, group=group)
            return tuple(gather_list)
        else:
            dist.gather(input, [], dst=dst, group=group)
            return input.new_tensor([])

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        if dist.get_rank(ctx.group) == ctx.dst:
            grad_outputs = list(grads)
            dist.scatter(grad_input,
                         grad_outputs,
                         src=ctx.dst,
                         group=ctx.group)
            return (grad_input, None, None, None) + grads
        else:
            dist.scatter(grad_input, [], src=ctx.dst, group=ctx.group)
            return grad_input, None, None, None, None


class ScatterFunc(Function):
    @staticmethod
    def forward(ctx,
                tensor,
                src,
                group=dist.group.WORLD,
                inplace=True,
                *scatter_list):
        ctx.src = src
        ctx.group = group
        if not inplace:
            tensor = torch.zeros_like(tensor)
        if dist.get_rank(group) == src:
            ctx.save_for_backward(*scatter_list)
            scatter_list = list(scatter_list)
            dist.scatter(tensor, scatter_list, src=src, group=group)
        else:
            dist.scatter(tensor, [], src=src, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_tensor):
        if dist.get_rank(ctx.group) == ctx.src:
            grad_outputs = [torch.zeros_like(g) for g in ctx.saved_tensors]
            dist.gather(grad_tensor, grad_outputs, ctx.src, group=ctx.group)
            return (grad_tensor, None, None, None) + tuple(grad_outputs)
        else:
            dist.gather(grad_tensor, [], ctx.src, group=ctx.group)
            return grad_tensor, None, None, None, None
