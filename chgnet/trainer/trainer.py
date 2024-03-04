from __future__ import annotations

import inspect
import os
import random
import shutil
import time
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
)
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel

from chgnet.model.model import CHGNet
from chgnet.data.dataset import collate_graphs
from chgnet.utils import AverageMeter, cuda_devices_sorted_by_free_mem, mae, write_json, distutils

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from chgnet import TrainTask

class Trainer:
    """A trainer to train CHGNet using energy, force, stress and magmom."""

    def __init__(
        self,
        model: nn.Module | None = None,
        targets: TrainTask = "ef",
        energy_loss_ratio: float = 1,
        force_loss_ratio: float = 1,
        stress_loss_ratio: float = 0.1,
        mag_loss_ratio: float = 0.1,
        contrastive_loss_ratio: float = 0.1,
        optimizer: str = "Adam",
        scheduler: str = "CosLR",
        criterion: str = "MSE",
        epochs: int = 50,
        starting_epoch: int = 0,
        learning_rate: float = 1e-3,
        print_freq: int = 100,
        local_rank: int = 0,
        torch_seed: int | None = None,
        data_seed: int | None = None,
        use_device: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize all hyper-parameters for trainer.

        Args:
            model (nn.Module): a CHGNet model
            targets ("ef" | "efs" | "efsm" | "efsmc"): The training targets. Default = "ef"
            energy_loss_ratio (float): energy loss ratio in loss function
                Default = 1
            force_loss_ratio (float): force loss ratio in loss function
                Default = 1
            stress_loss_ratio (float): stress loss ratio in loss function
                Default = 0.1
            mag_loss_ratio (float): magmom loss ratio in loss function
                Default = 0.1
            optimizer (str): optimizer to update model. Can be "Adam", "SGD", "AdamW",
                "RAdam". Default = 'Adam'
            scheduler (str): learning rate scheduler. Can be "CosLR", "ExponentialLR",
                "CosRestartLR". Default = 'CosLR'
            criterion (str): loss function criterion. Can be "MSE", "Huber", "MAE"
                Default = 'MSE'
            epochs (int): number of epochs for training
                Default = 50
            starting_epoch (int): The epoch number to start training at.
            learning_rate (float): initial learning rate
                Default = 1e-3
            print_freq (int): frequency to print training output
                Default = 100
            local_rank (int): local rank to decide the idx of GPU 
                Default = 0,
            torch_seed (int): random seed for torch
                Default = None
            data_seed (int): random seed for random
                Default = None
            use_device (str, optional): device name to train the CHGNet.
                Can be "cuda", "cpu"
                Default = None
            **kwargs (dict): additional hyper-params for optimizer, scheduler, etc.
        """
        # Store trainer args for reproducibility
        self.trainer_args = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "__class__", "model", "kwargs"]
        }
        self.trainer_args.update(kwargs)

        # Determine the device to use
        if use_device is not None:
            self.device = use_device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        if self.device == "cuda":
            # Determine cuda device with most available memory
            # device_with_most_available_memory = cuda_devices_sorted_by_free_mem()[-1]
            # self.device = f"cuda:{device_with_most_available_memory}"
            self.device = f"cuda:{local_rank}"

        self.model = model
        self.model.to(self.device)

        if distutils.initialized():
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device]
            )
        self.targets = targets
        # if torch_seed is not None:
        #     torch.manual_seed(torch_seed)
        # if data_seed:
        #     random.seed(data_seed)

        # Define optimizer
        if optimizer == "SGD":
            momentum = kwargs.pop("momentum", 0.9)
            weight_decay = kwargs.pop("weight_decay", 0)
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer == "Adam":
            weight_decay = kwargs.pop("weight_decay", 0)
            self.optimizer = torch.optim.Adam(
                model.parameters(), learning_rate, weight_decay=weight_decay
            )
        elif optimizer == "AdamW":
            weight_decay = kwargs.pop("weight_decay", 1e-2)
            self.optimizer = torch.optim.AdamW(
                model.parameters(), learning_rate, weight_decay=weight_decay
            )
        elif optimizer == "RAdam":
            weight_decay = kwargs.pop("weight_decay", 0)
            self.optimizer = torch.optim.RAdam(
                model.parameters(), learning_rate, weight_decay=weight_decay
            )

        # Define learning rate scheduler
        if scheduler in ["MultiStepLR", "multistep"]:
            scheduler_params = kwargs.pop(
                "scheduler_params",
                {
                    "milestones": [4 * epochs, 6 * epochs, 8 * epochs, 9 * epochs],
                    "gamma": 0.3,
                },
            )
            self.scheduler = MultiStepLR(self.optimizer, **scheduler_params)
            self.scheduler_type = "multistep"
        elif scheduler in ["ExponentialLR", "Exp", "Exponential"]:
            scheduler_params = kwargs.pop("scheduler_params", {"gamma": 0.98})
            self.scheduler = ExponentialLR(self.optimizer, **scheduler_params)
            self.scheduler_type = "exp"
        elif scheduler in ["CosineAnnealingLR", "CosLR", "Cos", "cos"]:
            scheduler_params = kwargs.pop("scheduler_params", {"decay_fraction": 1e-2})
            decay_fraction = scheduler_params.pop("decay_fraction")
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=10 * epochs,  # Maximum number of iterations.
                eta_min=decay_fraction * learning_rate,
            )
            self.scheduler_type = "cos"
        elif scheduler == "CosRestartLR":
            scheduler_params = kwargs.pop(
                "scheduler_params", {"decay_fraction": 1e-2, "T_0": 10, "T_mult": 2}
            )
            decay_fraction = scheduler_params.pop("decay_fraction")
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                eta_min=decay_fraction * learning_rate,
                **scheduler_params,
            )
            self.scheduler_type = "cosrestart"
        else:
            raise NotImplementedError

        # Define loss criterion
        self.criterion = CombinedLoss(
            target_str=self.targets,
            criterion=criterion,
            is_intensive=self._unwrapped_model.is_intensive,
            energy_loss_ratio=energy_loss_ratio,
            force_loss_ratio=force_loss_ratio,
            stress_loss_ratio=stress_loss_ratio,
            mag_loss_ratio=mag_loss_ratio,
            contrastive_loss_ratio=contrastive_loss_ratio,
            **kwargs,
        )
        self.epochs = epochs
        self.starting_epoch = starting_epoch

        self.print_freq = print_freq
        self.training_history: dict[
            str, dict[Literal["train", "val", "test"], list[float]]
        ] = {key: {"train": [], "val": [], "test": []} for key in self.targets}
        self.best_model = None

    @property
    def _unwrapped_model(self):
        module = self.model
        while isinstance(module, DistributedDataParallel):
            module = module.module
        return module

    def load_datasets(
        self, 
        dataset: Dataset,
        batch_size: int = 64,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        return_test: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        seed: int = 0
    ):
        """Randomly partition a dataset into train, val, test loaders.

        Args:
            dataset (Dataset): The dataset to partition.
            batch_size (int): The batch size for the data loaders
                Default = 64
            train_ratio (float): The ratio of the dataset to use for training
                Default = 0.8
            val_ratio (float): The ratio of the dataset to use for validation
                Default: 0.1
            return_test (bool): Whether to return a test data loader
                Default = True
            num_workers (int): The number of worker processes for loading the data
                see torch Dataloader documentation for more info
                Default = 0
            pin_memory (bool): Whether to pin the memory of the data loaders
                Default: True
            seed (int): Random seed for sampler
                Default = 0

        Returns:
            train_loader, val_loader and optionally test_loader
        """
        total_size = len(dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)

        train_indices = indices[0:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)
        self.test_dataset = Subset(dataset, test_indices)

        num_replicas = distutils.get_world_size()
        rank = distutils.get_rank()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, seed=seed, drop_last=True, num_replicas=num_replicas, rank=rank)
        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=num_replicas, rank=rank)
        self.test_sampler = DistributedSampler(
            self.test_dataset, shuffle=False, num_replicas=num_replicas, rank=rank)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            sampler=self.val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if return_test:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                collate_fn=collate_graphs,
                sampler=self.test_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            self.test_loader = None

    def train(
        self,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        save_dir: str | None = None,
        save_test_result: bool = False,
        train_composition_model: bool = False,
    ) -> None:
        """Train the model using torch data_loaders.

        Args:
            train_loader (DataLoader): train loader to update CHGNet weights
            val_loader (DataLoader): val loader to test accuracy after each epoch
            test_loader (DataLoader):  test loader to test accuracy at end of training.
                Can be None.
                Default = None
            save_dir (str): the dir name to save the trained weights
                Default = None
            save_test_result (bool): Whether to save the test set prediction in a JSON
                file. Default = False
            train_composition_model (bool): whether to train the composition model
                (AtomRef), this is suggested when the fine-tuning dataset has large
                elemental energy shift from the pretrained CHGNet, which typically comes
                from different DFT pseudo-potentials.
                Default = False
        """
        if self.model is None:
            raise ValueError("Model needs to be initialized")
        global best_checkpoint  # noqa: PLW0603
        if distutils.is_master():
            if save_dir is None:
                save_dir = f"{datetime.now():%Y-%m-%d-%H-%m-%S}"
            os.makedirs(save_dir, exist_ok=True)
            print(f"training targets: {self.targets}")
        print(f"Begin Training: using {self.device} device")
        # self.model.to(self.device)

        # Turn composition model training on / off
        # for param in self.model.composition_model.parameters():
        #     param.requires_grad = train_composition_model
        for param in self._unwrapped_model.composition_model.parameters():
            param.requires_grad = train_composition_model

        for epoch in range(self.starting_epoch, self.epochs):
            self.train_sampler.set_epoch(epoch)
            # train
            train_mae = self._train(self.train_loader, epoch)
            if "e" in train_mae and train_mae["e"] != train_mae["e"]:
                if distutils.is_master():
                    print("Exit due to NaN")
                break

            # val
            val_mae = self._validate(self.val_loader)
            for key in self.targets:
                self.training_history[key]["train"].append(train_mae[key])
                self.training_history[key]["val"].append(val_mae[key])

            if "e" in val_mae and val_mae["e"] != val_mae["e"]:
                if distutils.is_master():
                    print("Exit due to NaN")
                break
            
            if distutils.is_master():
                self.save_checkpoint(epoch, val_mae, save_dir=save_dir)

        if self.test_loader is not None:
            # test best model
            if distutils.is_master():
                print("---------Evaluate Model on Test Set---------------")
            for file in os.listdir(save_dir):
                if file.startswith("bestE_"):
                    test_file = file
                    best_checkpoint = torch.load(os.path.join(save_dir, test_file))

            self.model.load_state_dict(best_checkpoint["model"]["state_dict"])
            if save_test_result:
                test_mae = self._validate(
                    self.test_loader, is_test=True, test_result_save_path=save_dir
                )
            else:
                test_mae = self._validate(
                    self.test_loader, is_test=True, test_result_save_path=None
                )

            for key in self.targets:
                self.training_history[key]["test"] = test_mae[key]
            if distutils.is_master():
                self.save(filename=os.path.join(save_dir, test_file))

    def _train(self, train_loader: DataLoader, current_epoch: int) -> dict:
        """Train all data for one epoch.

        Args:
            train_loader (DataLoader): train loader to update CHGNet weights
            current_epoch (int): used for resume unfinished training

        Returns:
            dictionary of training errors
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = {}
        for target in self.targets:
            mae_errors[target] = AverageMeter()

        # switch to train mode
        self.model.train()

        start = time.perf_counter()  # start timer
        for idx, (graphs, targets) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.perf_counter() - start)

            # get input
            for g in graphs:
                requires_force = "f" in self.targets
                g.atom_frac_coord.requires_grad = requires_force
            graphs = [g.to(self.device) for g in graphs]
            targets = {k: self.move_to(v, self.device) for k, v in targets.items()}

            # compute output
            if "c" in self.targets:
                prediction = self.model(graphs, task=self.targets, return_crystal_feas=True)
                contrastive_prediction = self.model(graphs, task="e", return_crystal_feas=True)
                combined_loss = self.criterion(targets, prediction, contrastive_prediction)
            else:
                prediction = self.model(graphs, task=self.targets)
                combined_loss = self.criterion(targets, prediction)

            losses.update(combined_loss["loss"].data.cpu().item(), len(graphs))
            for key in self.targets:
                mae_errors[key].update(
                    combined_loss[f"{key}_MAE"].cpu().item(),
                    combined_loss[f"{key}_MAE_size"],
                )

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            combined_loss["loss"].backward()
            self.optimizer.step()

            # adjust learning rate every 1/10 of the epoch
            if idx + 1 in np.arange(1, 11) * len(train_loader) // 10:
                self.scheduler.step()

            # free memory
            del graphs, targets
            del prediction, combined_loss

            # measure elapsed time
            batch_time.update(time.perf_counter() - start)
            start = time.perf_counter()

            if idx == 0 or (idx + 1) % self.print_freq == 0:
                message = (
                    f"Epoch: [{current_epoch}][{idx + 1}/{len(train_loader)}] | "
                    f"Time ({batch_time.avg:.3f})({data_time.avg:.3f}) | "
                    f"Loss {losses.val:.4f}({losses.avg:.4f}) | MAE "
                )
                for key in self.targets:
                    message += (
                        f"{key} {mae_errors[key].val:.3f}({mae_errors[key].avg:.3f})  "
                    )
                if distutils.is_master():
                    print(message)
        return {key: round(err.avg, 6) for key, err in mae_errors.items()}

    def _validate(
        self,
        val_loader: DataLoader,
        is_test: bool = False,
        test_result_save_path: str | None = None,
    ) -> dict:
        """Validation or test step.

        Args:
            val_loader (DataLoader): val loader to test accuracy after each epoch
            is_test (bool): whether it's test step
            test_result_save_path (str): path to save test_result

        Returns:
            dictionary of training errors
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = {}
        for key in self.targets:
            mae_errors[key] = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        if is_test:
            test_pred = []

        end = time.perf_counter()
        for idx, (graphs, targets) in enumerate(val_loader):
            if "f" in self.targets or "s" in self.targets:
                for g in graphs:
                    requires_force = "f" in self.targets
                    g.atom_frac_coord.requires_grad = requires_force
                graphs = [g.to(self.device) for g in graphs]
                targets = {k: self.move_to(v, self.device) for k, v in targets.items()}
            else:
                with torch.no_grad():
                    graphs = [g.to(self.device) for g in graphs]
                    targets = {
                        k: self.move_to(v, self.device) for k, v in targets.items()
                    }

            # compute output
            if "c" in self.targets:
                prediction = self.model(graphs, task=self.targets, return_crystal_feas=True)
                contrastive_prediction = self.model(graphs, task="e", return_crystal_feas=True)
                combined_loss = self.criterion(targets, prediction, contrastive_prediction)
            else:
                prediction = self.model(graphs, task=self.targets)
                combined_loss = self.criterion(targets, prediction)

            losses.update(combined_loss["loss"].data.cpu().item(), len(graphs))
            for key in self.targets:
                mae_errors[key].update(
                    combined_loss[f"{key}_MAE"].cpu().item(),
                    combined_loss[f"{key}_MAE_size"],
                )
            if is_test and test_result_save_path:
                for idx, graph_i in enumerate(graphs):
                    tmp = {
                        "mp_id": graph_i.mp_id,
                        "graph_id": graph_i.graph_id,
                        "energy": {
                            "ground_truth": targets["e"][idx].cpu().detach().tolist(),
                            "prediction": prediction["e"][idx].cpu().detach().tolist(),
                        },
                    }
                    if "f" in self.targets:
                        tmp["force"] = {
                            "ground_truth": targets["f"][idx].cpu().detach().tolist(),
                            "prediction": prediction["f"][idx].cpu().detach().tolist(),
                        }
                    if "s" in self.targets:
                        tmp["stress"] = {
                            "ground_truth": targets["s"][idx].cpu().detach().tolist(),
                            "prediction": prediction["s"][idx].cpu().detach().tolist(),
                        }
                    if "m" in self.targets:
                        if targets["m"][idx] is not None:
                            m_ground_truth = targets["m"][idx].cpu().detach().tolist()
                        else:
                            m_ground_truth = None
                        tmp["mag"] = {
                            "ground_truth": m_ground_truth,
                            "prediction": prediction["m"][idx].cpu().detach().tolist(),
                        }
                    test_pred.append(tmp)

            # free memory
            del graphs, targets
            del prediction, combined_loss

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            if (idx + 1) % self.print_freq == 0:
                name = "Test" if is_test else "Val"
                message = (
                    f"{name}: [{idx + 1}/{len(val_loader)}] | "
                    f"Time ({batch_time.avg:.3f}) | "
                    f"Loss {losses.val:.4f}({losses.avg:.4f}) | MAE "
                )
                for key in self.targets:
                    message += (
                        f"{key} {mae_errors[key].val:.3f}({mae_errors[key].avg:.3f})  "
                    )
                if distutils.is_master():
                    print(message)

        if is_test:
            message = "**  "
            if test_result_save_path:
                if distutils.initialized():
                    write_json(
                        test_pred, os.path.join(test_result_save_path, "test_result_{}.json".format(distutils.get_rank()))
                    )
                else:
                    write_json(
                        test_pred, os.path.join(test_result_save_path, "test_result.json".format(distutils.get_rank()))
                    )
        else:
            message = "*   "
        for key in self.targets:
            message += f"{key}_MAE ({mae_errors[key].avg:.3f}) \t"
        if distutils.is_master():
            print(message)
        return {k: round(mae_error.avg, 6) for k, mae_error in mae_errors.items()}

    def get_best_model(self) -> CHGNet:
        """Get best model recorded in the trainer."""
        if self.best_model is None:
            raise RuntimeError("the model needs to be trained first")
        MAE = min(self.training_history["e"]["val"])
        if distutils.is_master():
            print(f"Best model has val {MAE =:.4}")
        return self.best_model

    @property
    def _init_keys(self) -> list[str]:
        return [
            key
            for key in list(inspect.signature(Trainer.__init__).parameters)
            if key not in (["self", "model", "kwargs"])
        ]

    def save(self, filename: str = "training_result.pth.tar") -> None:
        """Save the model, graph_converter, etc."""
        state = {
            # "model": self.model.as_dict(),
            "model": self._unwrapped_model.as_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "training_history": self.training_history,
            "trainer_args": self.trainer_args,
        }
        torch.save(state, filename)

    def save_checkpoint(
        self, epoch: int, mae_error: dict, save_dir: str | None = None
    ) -> None:
        """Function to save CHGNet trained weights after each epoch.

        Args:
            epoch (int): the epoch number
            mae_error (dict): dictionary that stores the MAEs
            save_dir (str): the directory to save trained weights
        """
        for fname in os.listdir(save_dir):
            if fname.startswith("epoch"):
                os.remove(os.path.join(save_dir, fname))

        err_str = "_".join(
            f"{key}{f'{mae_error[key] * 1000:.0f}' if key in mae_error else 'NA'}"
            for key in "efsm"
        )
        filename = os.path.join(save_dir, f"epoch{epoch}_{err_str}.pth.tar")
        self.save(filename=filename)

        # save the model if it has minimal val energy error or val force error
        if mae_error["e"] == min(self.training_history["e"]["val"]):
            self.best_model = self.model
            for fname in os.listdir(save_dir):
                if fname.startswith("bestE"):
                    os.remove(os.path.join(save_dir, fname))
            shutil.copyfile(
                filename,
                os.path.join(save_dir, f"bestE_epoch{epoch}_{err_str}.pth.tar"),
            )
        if mae_error["f"] == min(self.training_history["f"]["val"]):
            for fname in os.listdir(save_dir):
                if fname.startswith("bestF"):
                    os.remove(os.path.join(save_dir, fname))
            shutil.copyfile(
                filename,
                os.path.join(save_dir, f"bestF_epoch{epoch}_{err_str}.pth.tar"),
            )

    @classmethod
    def load(cls, path: str) -> Trainer:
        """Load trainer state_dict."""
        state = torch.load(path, map_location=torch.device("cpu"))
        model = CHGNet.from_dict(state["model"])
        print(f"Loaded model params = {sum(p.numel() for p in model.parameters()):,}")
        # drop model from trainer_args if present
        state["trainer_args"].pop("model", None)
        trainer = Trainer(model=model, **state["trainer_args"])
        trainer.model.to(trainer.device)
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.scheduler.load_state_dict(state["scheduler"])
        trainer.training_history = state["training_history"]
        trainer.starting_epoch = len(trainer.training_history["e"]["train"])
        return trainer

    @staticmethod
    def move_to(obj, device) -> Tensor | list[Tensor]:
        """Move object to device."""
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, list):
            out = []
            for tensor in obj:
                if tensor is not None:
                    out.append(tensor.to(device))
                else:
                    out.append(None)
            return out
        raise TypeError("Invalid type for move_to")


class CombinedLoss(nn.Module):
    """A combined loss function of energy, force, stress and magmom."""

    def __init__(
        self,
        target_str: str = "ef",
        criterion: str = "MSE",
        is_intensive: bool = True,
        energy_loss_ratio: float = 1,
        force_loss_ratio: float = 1,
        stress_loss_ratio: float = 0.1,
        mag_loss_ratio: float = 0.1,
        contrastive_loss_ratio: float = 0.1,
        delta: float = 0.1,
        tau: float = 0.05,
    ) -> None:
        """Initialize the combined loss.

        Args:
            target_str: the training target label. Can be "e", "ef", "efs", "efsm" etc.
                Default = "ef"
            criterion: loss criterion to use
                Default = "MSE"
            is_intensive (bool): whether the energy label is intensive
                Default = True
            energy_loss_ratio (float): energy loss ratio in loss function
                Default = 1
            force_loss_ratio (float): force loss ratio in loss function
                Default = 1
            stress_loss_ratio (float): stress loss ratio in loss function
                Default = 0.1
            mag_loss_ratio (float): magmom loss ratio in loss function
                Default = 0.1
            contrastive_loss_ratio (float): contrastive loss ratio in loss function
                Default = 0.1
            delta (float): delta for torch.nn.HuberLoss. Default = 0.1
            tau (float): temperature for contrastive loss. Default = 0.05
        """
        super().__init__()
        # Define loss criterion
        if criterion in ["MSE", "mse"]:
            self.criterion = nn.MSELoss()
        elif criterion in ["MAE", "mae", "l1"]:
            self.criterion = nn.L1Loss()
        elif criterion == "Huber":
            self.criterion = nn.HuberLoss(delta=delta)
        else:
            raise NotImplementedError
        self.contrastive_criterion = nn.CrossEntropyLoss()
        self.target_str = target_str
        self.is_intensive = is_intensive
        self.energy_loss_ratio = energy_loss_ratio
        self.tau = tau
        if "f" not in self.target_str:
            self.force_loss_ratio = 0
        else:
            self.force_loss_ratio = force_loss_ratio
        if "s" not in self.target_str:
            self.stress_loss_ratio = 0
        else:
            self.stress_loss_ratio = stress_loss_ratio
        if "m" not in self.target_str:
            self.mag_loss_ratio = 0
        else:
            self.mag_loss_ratio = mag_loss_ratio
        if "c" not in self.target_str:
            self.contrastive_loss_ratio = 0
        else:
            self.contrastive_loss_ratio = contrastive_loss_ratio

    def forward(
        self,
        targets: dict[str, Tensor],
        prediction: dict[str, Tensor],
        contrastive_prediction: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Compute the combined loss using CHGNet prediction and labels
        this function can automatically mask out magmom loss contribution of
        data points without magmom labels.

        Args:
            targets (dict): DFT labels
            prediction (dict): CHGNet prediction
            contrastive_prediction (dict, optional): CHGNet prediction for contrastive learning

        Returns:
            dictionary of all the loss, MAE and MAE_size
        """
        out = {"loss": 0.0}
        # Energy
        if "e" in targets:
            if self.is_intensive:
                out["loss"] += self.energy_loss_ratio * self.criterion(
                    targets["e"], prediction["e"]
                )
                out["e_MAE"] = mae(targets["e"], prediction["e"]) * prediction["e"].shape[0]
                out["e_MAE_size"] = prediction["e"].shape[0]
            else:
                e_per_atom_target = targets["e"] / prediction["atoms_per_graph"]
                e_per_atom_pred = prediction["e"] / prediction["atoms_per_graph"]
                out["loss"] += self.energy_loss_ratio * self.criterion(
                    e_per_atom_target, e_per_atom_pred
                )
                out["e_MAE"] = mae(e_per_atom_target, e_per_atom_pred) * prediction["e"].shape[0]
                out["e_MAE_size"] = prediction["e"].shape[0]

        # Force
        if "f" in targets:
            forces_pred = torch.cat(prediction["f"], dim=0)
            forces_target = torch.cat(targets["f"], dim=0)
            out["loss"] += self.force_loss_ratio * self.criterion(
                forces_target, forces_pred
            )
            out["f_MAE"] = mae(forces_target, forces_pred) * forces_target.shape[0]
            out["f_MAE_size"] = forces_target.shape[0]

        # Stress
        if "s" in targets:
            stress_pred = torch.cat(prediction["s"], dim=0)
            stress_target = torch.cat(targets["s"], dim=0)
            out["loss"] += self.stress_loss_ratio * self.criterion(
                stress_target, stress_pred
            )
            out["s_MAE"] = mae(stress_target, stress_pred) * stress_target.shape[0]
            out["s_MAE_size"] = stress_target.shape[0]

        # Mag
        if "m" in targets:
            mag_preds, mag_targets = [], []
            m_mae_size = 0
            for mag_pred, mag_target in zip(prediction["m"], targets["m"]):
                # exclude structures without magmom labels
                if mag_target is not None:
                    mag_preds.append(mag_pred)
                    mag_targets.append(mag_target)
                    m_mae_size += mag_target.shape[0]
            if mag_targets != []:
                mag_preds = torch.cat(mag_preds, dim=0)
                mag_targets = torch.cat(mag_targets, dim=0)
                out["loss"] += self.mag_loss_ratio * self.criterion(
                    mag_targets, mag_preds
                )
                out["m_MAE"] = mae(mag_targets, mag_preds) * m_mae_size
                out["m_MAE_size"] = m_mae_size
            else:
                mag_targets = mag_preds.detach()
                out["loss"] += self.mag_loss_ratio * self.criterion(
                    mag_targets, mag_preds
                )
                out["m_MAE"] = torch.tensor(0., device=prediction["e"].device) # torch.tensor([1], device=prediction["e"].device) * m_mae_size
                out["m_MAE_size"] = m_mae_size
        
        for key in out:
            if key not in ["loss"]:
                try:
                    out[key] = distutils.all_reduce(
                        out[key], average=False, device=prediction["e"].device
                    )
                except:
                    print("Reduce error.", key, out[key])
                    raise
        
        for key in targets:
            if key != 'c':
                out["{}_MAE".format(key)] /= out["{}_MAE_size".format(key)]
        
        # Contrastive learning
        if contrastive_prediction is not None:
            crystal_fea = prediction["crystal_fea"] / prediction["crystal_fea"].norm(dim=1, keepdim=True)
            contrastive_crystal_fea = contrastive_prediction["crystal_fea"] / contrastive_prediction["crystal_fea"].norm(dim=1, keepdim=True)

            full_crystal_fea = torch.stack([crystal_fea, contrastive_crystal_fea], dim=1) # B * 2 * D
            full_crystal_fea_aggregated = distutils.all_gather(full_crystal_fea)

            full_crystal_fea = torch.cat(full_crystal_fea_aggregated, dim=0) # N * 2 * D
            
            crystal_fea = full_crystal_fea[:, 0, :]
            contrastive_crystal_fea = full_crystal_fea[:, 1, :]

            sim_matrix = torch.mm(crystal_fea, contrastive_crystal_fea.t()) / self.tau
            labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
            contrastive_loss = self.contrastive_criterion(sim_matrix, labels)

            out["loss"] += self.contrastive_loss_ratio * contrastive_loss
            out["c_MAE"] = contrastive_loss * sim_matrix.shape[0]
            out["c_MAE_size"] = sim_matrix.shape[0]

        return out
