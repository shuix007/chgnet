import os
from importlib.metadata import version
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import wandb
from chgnet.model import StructOptimizer
from pymatgen.core import Structure
from tqdm import tqdm

from matbench_discovery import timestamp, today
from matbench_discovery.data import DATA_FILES, as_dict_handler, df_wbm
from matbench_discovery.plots import wandb_scatter
from matbench_discovery.slurm import slurm_submit

from strenum import StrEnum
from enum import unique
from typing import Union
from typing_extensions import Self

from pymatviz.utils import styled_html_tag
import argparse

class LabelEnum(StrEnum):
    """StrEnum with optional label and description attributes plus dict() method."""

    # def __new__(
    #     cls, val: str, label: str | None = None, desc: str | None = None
    # ) -> Self:
    def __new__(
        cls, val: str, label: Union[str, None] = None, desc: Union[str, None] = None
    ) -> Self:
        """Create a new class."""
        member = str.__new__(cls, val)
        member._value_ = val
        member.__dict__ |= dict(label=label, desc=desc)
        return member

    @property
    def label(self) -> str:
        """Make label read-only."""
        return self.__dict__["label"]

    @property
    def description(self) -> str:
        """Make description read-only."""
        return self.__dict__["desc"]

    @classmethod
    def key_val_dict(cls) -> dict[str, str]:
        """Map of keys to values."""
        return {key: str(val) for key, val in cls.__members__.items()}

    @classmethod
    def val_label_dict(cls) -> dict[str, Union[str, None]]:
        """Map of values to labels."""
        return {str(val): val.label for val in cls.__members__.values()}

    @classmethod
    def val_desc_dict(cls) -> dict[str, Union[str, None]]:
        """Map of values to descriptions."""
        return {str(val): val.description for val in cls.__members__.values()}

    @classmethod
    def label_desc_dict(cls) -> dict[Union[str, None], Union[str, None]]:
        """Map of labels to descriptions."""
        return {str(val.label): val.description for val in cls.__members__.values()}


@unique
class Key(LabelEnum):
    """Keys used to access dataframes columns."""

    arity = "arity", "Arity"
    bandgap_pbe = "bandgap_pbe", "PBE Band Gap"
    chem_sys = "chemical_system", "Chemical System"
    composition = "composition", "Composition"
    cse = "computed_structure_entry", "Computed Structure Entry"
    daf = "DAF", "Discovery Acceleration Factor"
    dft_energy = "uncorrected_energy", "DFT Energy"
    e_form = "e_form_per_atom_mp2020_corrected", "DFT E_form"
    e_form_pred = "e_form_per_atom_pred", "Predicted E_form"
    e_form_raw = "e_form_per_atom_uncorrected", "DFT E_form raw"
    e_form_wbm = "e_form_per_atom_wbm", "WBM E_form"
    each = "energy_above_hull", "E<sub>hull dist</sub>"
    each_pred = "e_above_hull_pred", "Predicted E<sub>hull dist</sub>"
    each_true = "e_above_hull_mp2020_corrected_ppd_mp", "E<sub>MP hull dist</sub>"
    each_wbm = "e_above_hull_wbm", "E<sub>WBM hull dist</sub>"
    final_struct = "relaxed_structure", "Relaxed Structure"
    forces = "forces", "Forces"
    form_energy = "formation_energy_per_atom", "Formation Energy (eV/atom)"
    formula = "formula", "Formula"
    init_struct = "initial_structure", "Initial Structure"
    magmoms = "magmoms", "Magnetic Moments"
    mat_id = "material_id", "Material ID"
    each_mean_models = "each_mean_models", "E<sub>hull dist</sub> mean of models"
    each_err_models = "each_err_models", "E<sub>hull dist</sub> mean error of models"
    model_std_each = "each_std_models", "Std. dev. over models"
    n_sites = "n_sites", "Number of Sites"
    site_nums = "site_nums", "Site Numbers", "Atomic numbers for each crystal site"
    spacegroup = "spacegroup", "Spacegroup"
    stress = "stress", "Stress"
    stress_trace = "stress_trace", "Stress Trace"
    struct = "structure", "Structure"
    task_id = "task_id", "Task ID"
    task_type = "task_type", "Task Type"
    train_task = "train_task", "Training Task"
    test_task = "test_task", "Test Task"
    targets = "targets", "Targets"
    # lowest WBM structures for a given prototype that isn't already in MP
    uniq_proto = "unique_prototype", "Unique Prototype"
    volume = "volume", "Volume (Å³)"
    wyckoff = "wyckoff_spglib", "Aflow-Wyckoff Label"  # relaxed structure Aflow label
    init_wyckoff = (
        "wyckoff_spglib_initial_structure",
        "Aflow-Wyckoff Label Initial Structure",
    )
    # number of structures in a model's training set
    train_set = "train_set", "Training Set"
    model_params = "model_params", "Model Params"  # model's parameter count
    model_type = "model_type", "Model Type"  # number of parameters in the model
    openness = "openness", "Openness"  # openness of data and code for a model


@unique
class Task(LabelEnum):
    """Thermodynamic stability prediction task types."""

    IS2RE = "IS2RE", "initial structure to relaxed energy"
    RS2RE = "RS2RE", "relaxed structure to relaxed energy"
    S2EFSM = "S2EFSM", "structure to energy force stress magmom"
    S2EFS = "S2EFS", "structure to energy force stress"
    # S2RE is for models that learned a discrete version of PES like CGCNN+P
    S2RE = "S2RE", "structure to relaxed energy"
    RP2RE = "RP2RE", "relaxed prototype to relaxed energy"
    IP2RE = "IP2RE", "initial prototype to relaxed energy"
    IS2E = "IS2E", "initial structure to energy"
    IS2RE_SR = "IS2RE-SR", "initial structure to relaxed energy after ML relaxation"

__author__ = "Janosh Riebesell"
__date__ = "2023-03-01"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', 
        type=str, 
        default='chgnet'
    )
    parser.add_argument(
        '--split', 
        type=int, 
        default=0
    )
    parser.add_argument(
        '--world_size', 
        type=int, 
        default=1
    )
    parser.add_argument(
        '--out_dir', 
        type=str, 
        default='./chgnet_relax_results'
    )
    args = parser.parse_args()

    task_type = Task.IS2RE
    slurm_array_task_count = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chgnet = StructOptimizer(use_device=device)  # load default pre-trained CHGNnet model
    if args.model != 'chgnet':
        trained_model_path = 'chgnet/pretrained/2024-03-13-10-19-28-MPtraj_SemDedup_Mean_030/bestE_epoch38_e36_f93_s462_m53.pth.tar'
        print("Loading pre-trained model from file ", trained_model_path)
        keys = chgnet.calculator.model.load_state_dict(torch.load(trained_model_path, map_location='cpu')['model']['state_dict'])
        chgnet.calculator.model.cuda()

    job_name = f"chgnet-{chgnet.version}-wbm-{task_type}"

    # %%
    data_path = {
        Task.RS2RE: DATA_FILES.wbm_computed_structure_entries,
        Task.IS2RE: DATA_FILES.wbm_initial_structures,
    }[task_type]
    print(f"\nJob started running {timestamp}")
    print(f"{data_path=}")
    e_pred_col = "chgnet_energy"
    ase_filter: Literal["FrechetCellFilter", "ExpCellFilter"] = "FrechetCellFilter"
    max_steps = 500
    fmax = 0.05

    df_in = pd.read_json(data_path).set_index(Key.mat_id)
    if args.world_size > 1:
        df_in = np.array_split(df_in, args.world_size)[args.split]

    # %%
    relax_results: dict[str, dict[str, Any]] = {}
    input_col = {Task.IS2RE: Key.init_struct, Task.RS2RE: Key.final_struct}[task_type]

    if task_type == Task.RS2RE:
        df_in[input_col] = [cse["structure"] for cse in df_in[Key.cse]]

    structures = df_in[input_col].map(Structure.from_dict).to_dict()

    # for material_id in tqdm(structures, desc="Relaxing"):
    for material_id in structures:
        if material_id in relax_results:
            continue
        try:
            relax_result = chgnet.relax(
                structures[material_id],
                verbose=False,
                steps=max_steps,
                fmax=fmax,
                relax_cell=max_steps > 0,
                ase_filter=ase_filter,
            )
            relax_results[material_id] = {
                e_pred_col: relax_result["trajectory"].energies[-1]
            }
            if max_steps > 0:
                relax_struct = relax_result["final_structure"]
                relax_results[material_id]["chgnet_structure"] = relax_struct
                # traj = relax_result["trajectory"]
                # relax_results[material_id]["chgnet_trajectory"] = traj.__dict__
        except Exception as exc:
            print(f"Failed to relax {material_id}: {exc!r}")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # %%
    df_out = pd.DataFrame(relax_results).T
    df_out.index.name = Key.mat_id
    df_out.to_csv(f"{args.out_dir}/{args.model}_relax_{args.split}_of_{args.world_size}.csv")