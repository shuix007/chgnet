import os
import json
import argparse
import numpy as np

from pymatgen.core import Structure
from chgnet.trainer import Trainer
from chgnet.model import CHGNet
from chgnet.data.dataset import StructureData, get_train_val_test_loader

def load_structures_from_json(
        json_filename='../FERMat/Data/MPtrj_2022.9_full.json',
        sample_filename='../FERMat/Data/SampledPretrain/MPtraj-semdedup/rseed0-0.10/colabfit_ids.txt'
    ):
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    if sample_filename is not None:
        with open(sample_filename, 'r') as f:
            sample_ids = set(f.read().splitlines())
    
    structures = []
    energies_per_atom = []
    forces = []
    stresses = []
    magmoms = []

    mp_ids = sorted(list(data.keys()))
    for mp_id in mp_ids:
        frame_ids = sorted(list(data[mp_id].keys())) # deterministic order
        for frame_id in frame_ids:
            if sample_filename is not None and frame_id not in sample_ids:
                continue
            frame = data[mp_id][frame_id]
            
            structures.append(Structure.from_dict(frame['structure']))
            energies_per_atom.append(frame['energy_per_atom'])
            forces.append(frame['force'])
            stresses.append(frame['stress'])
            magmoms.append(frame['magmom'])
    print('Loaded', len(structures), 'structures')

    dataset = StructureData(
        structures=structures,
        energies=energies_per_atom,
        forces=forces,
        stresses=stresses,  # can be None
        magmoms=magmoms,  # can be None
    )
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset, batch_size=40, train_ratio=0.9, val_ratio=0.05
    )
    return train_loader, val_loader, test_loader

def main(args):
    train_loader, val_loader, test_loader = load_structures_from_json(
        json_filename=args.json_filename,
        sample_filename=args.sample_filename
    )

    model = CHGNet(
        atom_fea_dim=64,
        bond_fea_dim=64,
        angle_fea_dim=64,
        composition_model="MPtrj",
        num_radial=31,
        num_angular=31,
        n_conv=4,
        atom_conv_hidden_dim=64,
        update_bond=True,
        bond_conv_hidden_dim=64,
        update_angle=True,
        angle_layer_hidden_dim=0,
        conv_dropout=0.1,
        mlp_dropout=0.1,
        read_out="ave",
        gMLP_norm='layer',
        readout_norm='layer',
        mlp_hidden_dims=[64, 64, 64],
        mlp_first=True,
        is_intensive=True,
        non_linearity="silu",
        atom_graph_cutoff=6,
        bond_graph_cutoff=3,
        graph_converter_algorithm="fast",
        cutoff_coeff=8,
        learnable_rbf=True,
    )

    trainer = Trainer(
        model=model,
        targets='efsmc',
        energy_loss_ratio=1,
        force_loss_ratio=1,
        stress_loss_ratio=0.1,
        mag_loss_ratio=0.1,
        optimizer='Adam',
        weight_decay=0,
        scheduler='CosLR',
        scheduler_params={'decay_fraction': 0.5e-2},
        criterion='Huber',
        delta=0.1,
        epochs=30,
        starting_epoch=0,
        learning_rate=5e-3,
        use_device='cuda',
        print_freq=10
    )

    trainer.train(train_loader, val_loader, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_filename', type=str, default='../FERMat/Data/MPtrj_2022.9_full.json')
    # set the default value of sample_filename to None
    parser.add_argument('--sample_filename', type=str, default=None)
    # parser.add_argument('--sample_filename', type=str, default='../FERMat/Data/SampledPretrain/MPtraj-semdedup/rseed0-0.10/colabfit_ids.txt')
    args = parser.parse_args()
    main(args)