"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import subprocess
from typing import List

import torch
import torch.distributed as dist

from typing import Optional, Type, TypeVar

_T = TypeVar("_T")


def assert_is_instance(obj: object, cls: Type[_T]) -> _T:
    if obj and not isinstance(obj, cls):
        raise TypeError(f"obj is not an instance of cls: obj={obj}, cls={cls}")
    return obj


def none_throws(x: Optional[_T], msg: Optional[str] = None) -> _T:
    if x is None:
        if msg:
            raise ValueError(msg)
        else:
            raise ValueError("x cannot be None")
    return x

def os_environ_get_or_throw(x: str) -> str:
    if x not in os.environ:
        raise RuntimeError(f"Could not find {x} in ENV variables")
    return none_throws(os.environ.get(x))

def setup(config) -> None:
    if config["submit"]:
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            try:
                config['local_rank'] = int(os.environ['LOCAL_RANK'])
                config['rank'] = int(os.environ['RANK'])
                config['World_size'] = int(os.environ['WORLD_SIZE'])
                config['master_addr'] = os.environ['MASTER_ADDR']
                config['master_port'] = os.environ['MASTER_PORT']

                # ensures GPU0 does not have extra context/higher peak memory
                torch.cuda.set_device(config["local_rank"])

                dist.init_process_group(
                    backend=config["distributed_backend"]
                )

                if initialized() and is_master():
                    print(
                        "Initialized process group: {}/{}, backend: {}, world_size: {}.".format(
                            config['master_addr'], config['master_port'], 
                            config["distributed_backend"], config['World_size']
                        )
                    )
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass

def cleanup() -> None:
    dist.destroy_process_group()


def initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if initialized() else 1


def is_master() -> bool:
    return get_rank() == 0


def synchronize() -> None:
    if get_world_size() == 1:
        return
    dist.barrier()


def broadcast(
    tensor: torch.Tensor, src, group=dist.group.WORLD, async_op: bool = False
) -> None:
    if get_world_size() == 1:
        return
    dist.broadcast(tensor, src, group, async_op)


def all_reduce(
    data, group=dist.group.WORLD, average: bool = False, device=None
) -> torch.Tensor:
    if get_world_size() == 1:
        return data
    tensor = data
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data)
    if device is not None:
        tensor = tensor.cuda(device)
    dist.all_reduce(tensor, group=group)
    if average:
        tensor /= get_world_size()
    if not isinstance(data, torch.Tensor):
        result = tensor.cpu().numpy() if tensor.numel() > 1 else tensor.item()
    else:
        result = tensor
    return result


def all_gather(
    data, group=dist.group.WORLD, device=None
) -> List[torch.Tensor]:
    if get_world_size() == 1:
        return [data]
    tensor = data
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data)
    if device is not None:
        tensor = tensor.cuda(device)
    tensor_list = [
        tensor.new_zeros(tensor.shape) for _ in range(get_world_size())
    ]
    dist.all_gather(tensor_list, tensor, group=group)
    if not isinstance(data, torch.Tensor):
        result = [tensor.cpu().numpy() for tensor in tensor_list]
    else:
        result = tensor_list
    return result