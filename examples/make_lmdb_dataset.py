import argparse
import glob
import multiprocessing as mp
import os
import pickle

import ase.io
import lmdb
import numpy as np
import torch
from tqdm import tqdm

import json
import random

from pymatgen.core.structure import Structure

from chgnet import utils
from chgnet.data.dataset import StructureData, StructureJsonData
from chgnet.graph import CrystalGraphConverter

def write_images_to_lmdb(mp_arg):
    db_path, samples, sampled_ids, failed_ids, idx, pid, args = mp_arg
    a2g = CrystalGraphConverter(atom_graph_cutoff=6, bond_graph_cutoff=3)

    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    pbar = tqdm(
        total=5000 * len(samples),
        position=pid,
        desc="Preprocessing data into LMDBs",
    )
    for sample in samples:
        with open(sample, 'r') as f:
            data = json.load(f)

        for frame_id, frame in data.items():
            try:
                data_object = a2g(Structure.from_dict(frame['structure']))
                data_object.energy_per_atom = torch.tensor(frame['energy_per_atom'])
                data_object.force = torch.tensor(frame['force'])
                data_object.stress = torch.tensor(frame['stress'])
                
                if frame['magmom'] is not None:
                    data_object.magmom = torch.tensor(frame['magmom'])
                else:
                    data_object.magmom = None

                txn = db.begin(write=True)
                txn.put(
                    f"{idx}".encode("ascii"),
                    pickle.dumps(data_object, protocol=-1),
                )
                txn.commit()
                idx += 1
                sampled_ids.append(frame_id + "\n")
            except:
                failed_ids.append(frame_id + "\n")

            pbar.update(1)

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return sampled_ids, failed_ids, idx

def main(args: argparse.Namespace) -> None:
    xyz_logs = glob.glob(os.path.join(args.data_path, "*.json"))

    # Create output directory if it doesn't exist.
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    # Initialize lmdb paths
    db_paths = [
        os.path.join(args.out_path, "data.%04d.lmdb" % i)
        for i in range(args.num_workers)
    ]

    # Chunk the trajectories into args.num_workers splits
    chunked_txt_files = np.array_split(xyz_logs, args.num_workers)

    # Extract features
    sampled_ids, failed_ids, idx = [[]] * args.num_workers, [[]] * args.num_workers, [0] * args.num_workers

    pool = mp.Pool(args.num_workers)
    mp_args = [
        (
            db_paths[i],
            chunked_txt_files[i],
            sampled_ids[i],
            failed_ids[i],
            idx[i],
            i,
            args,
        )
        for i in range(args.num_workers)
    ]
    op = list(zip(*pool.imap(write_images_to_lmdb, mp_args)))
    sampled_ids, failed_ids, idx = list(op[0]), list(op[1]), list(op[2])

    # Log sampled image, trajectory trace
    for j, i in enumerate(range(args.num_workers)):
        ids_log = open(
            os.path.join(args.out_path, "data_log.%04d.txt" % i), "w"
        )
        ids_log.writelines(sampled_ids[j])

        failed_ids_log = open(
            os.path.join(args.out_path, "failed_data_log.%04d.txt" % i), "w"
        )
        failed_ids_log.writelines(failed_ids[j])


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to dir containing *.extxyz and *.txt files",
    )
    parser.add_argument(
        "--out-path",
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--get-edges",
        action="store_true",
        help="Store edge indices in LMDB, ~10x storage requirement. Default: compute edge indices on-the-fly.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )
    parser.add_argument(
        "--ref-energy", action="store_true", help="Subtract reference energies"
    )
    parser.add_argument(
        "--test-data",
        action="store_true",
        help="Is data being processed test data?",
    )
    return parser


if __name__ == "__main__":
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args()
    main(args)