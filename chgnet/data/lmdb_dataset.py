import bisect
import logging
import pickle
import warnings
from pathlib import Path
from typing import List, Optional, TypeVar

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Optional, Type, TypeVar

T_co = TypeVar("T_co", covariant=True)
_T = TypeVar("_T")

def assert_is_instance(obj: object, cls: Type[_T]) -> _T:
    if obj and not isinstance(obj, cls):
        raise TypeError(f"obj is not an instance of cls: obj={obj}, cls={cls}")
    return obj

class LmdbDataset(Dataset):
    def __init__(self, config) -> None:
        super(LmdbDataset, self).__init__()
        self.config = config

        self.path = Path(self.config["src"])
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

            self._keys = []
            self.envs = []
            for db_path in db_paths:
                cur_env = self.connect_db(db_path)
                self.envs.append(cur_env)

                # If "length" encoded as ascii is present, use that
                length_entry = cur_env.begin().get("length".encode("ascii"))
                if length_entry is not None:
                    num_entries = pickle.loads(length_entry)
                else:
                    # Get the number of stores data from the number of entries
                    # in the LMDB
                    num_entries = cur_env.stat()["entries"]

                # Append the keys (0->num_entries) as a list
                self._keys.append(list(range(num_entries)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)

            # If "length" encoded as ascii is present, use that
            length_entry = self.env.begin().get("length".encode("ascii"))
            if length_entry is not None:
                num_entries = pickle.loads(length_entry)
            else:
                # Get the number of stores data from the number of entries
                # in the LMDB
                num_entries = assert_is_instance(
                    self.env.stat()["entries"], int
                )

            self._keys = list(range(num_entries))
            self.num_samples = num_entries

        # If specified, limit dataset to only a portion of the entire dataset
        # total_shards: defines total chunks to partition dataset
        # shard: defines dataset shard to make visible
        self.sharded = False
        if "shard" in self.config and "total_shards" in self.config:
            self.sharded = True
            self.indices = range(self.num_samples)
            # split all available indices into 'total_shards' bins
            self.shards = np.array_split(
                self.indices, self.config.get("total_shards", 1)
            )
            # limit each process to see a subset of data based off defined shard
            self.available_indices = self.shards[self.config.get("shard", 0)]
            self.num_samples = len(self.available_indices)

        self.key_mapping = self.config.get("key_mapping", None)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> T_co:
        # if sharding, remap idx to appropriate idx of the sharded set
        if self.sharded:
            idx = self.available_indices[idx]
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pickle.loads(datapoint_pickled)
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(
                f"{self._keys[idx]}".encode("ascii")
            )
            data_object = pickle.loads(datapoint_pickled)

        if self.key_mapping is not None:
            for _property in self.key_mapping:
                # catch for test data not containing labels
                if _property in data_object:
                    new_property = self.key_mapping[_property]
                    if new_property not in data_object:
                        data_object[new_property] = data_object[_property]
                        del data_object[_property]

        return data_object

    def connect_db(self, lmdb_path: Optional[Path] = None) -> lmdb.Environment:
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self) -> None:
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()

def collate_graphs(batch_data: list):
    """Collate of list of (graph, target) into batch data.

    Args:
        batch_data (list): list of (graph, target(dict))

    Returns:
        graphs (List): a list of graphs
        targets (Dict): dictionary of targets, where key and values are:
            e (Tensor): energies of the structures [batch_size]
            f (Tensor): forces of the structures [n_batch_atoms, 3]
            s (Tensor): stresses of the structures [3*batch_size, 3]
            m (Tensor): magmom of the structures [n_batch_atoms]
    """
    tgt_key_map = {
        'e': 'energy_per_atom',
        'f': 'force',
        's': 'stress',
        'm': 'magmom'
    }
    graphs = [graph for graph in batch_data]
    all_targets = {key: [] for key in ["e", "f", "s", "m"]}
    all_targets["e"] = torch.tensor(
        [g.energy_per_atom for g in batch_data], dtype=torch.float32
    )
    
    for g in batch_data:
        for tgt in ["f", "s", "m"]:
            key = tgt_key_map[tgt]
            if hasattr(g, key):
                value = getattr(g, key)

                if tgt == 's':
                    value = -0.1 * value # Convert from GPa to bar
                
                if tgt == 'm' and value is not None:
                    value = torch.abs(value)

                all_targets[tgt].append(value)

    return graphs, all_targets