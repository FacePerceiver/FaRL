import torch.distributed as dist
from torch.distributed import ReduceOp


class AsyncOpList(object):
    def __init__(self, ops):
        self.ops = ops

    def wait(self):
        for op in self.ops:
            op.wait()

    def is_completed(self):
        for op in self.ops:
            if not op.is_completed():
                return False
        return True


def reduce_scatter(tensor,
                   tensor_list,
                   op=ReduceOp.SUM,
                   group=dist.group.WORLD,
                   async_op=False):
    rank = dist.get_rank(group)
    if tensor is None:
        tensor = tensor_list[rank]
    if tensor.dim() == 0:
        tensor = tensor.view(-1)
    tensor[:] = tensor_list[rank]
    # print("rank {} grads after scatter before reduce: {}".format(rank, tensor))
    ops = []
    for i in range(dist.get_world_size(group)):
        if i == rank:
            tmp = dist.reduce(tensor.contiguous(), rank, op, group, async_op=True)
            # print("rank {} grads after scatter & reduce: {}".format(rank, tensor))
        else:
            tmp = dist.reduce(tensor_list[i].contiguous(), i, op, group, async_op=True)
        ops.append(tmp)

    oplist = AsyncOpList(ops)
    if async_op:
        return oplist
    else:
        oplist.wait()
