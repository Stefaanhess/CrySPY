import configparser
import random

import pandas as pd

from ..IO import io_stat, pkl_data
from ..IO import read_input as rin
import schnetpack as spk
import cryspyschnet
import torch
import os


def initialize(stat, init_struc_data, rslt_data):

    # ---------- log
    print('\n# ---------- Selection: 1')
    with open('cryspy.out', 'a') as fout:
        fout.write('\n# ---------- Selection 1\n')

    # ---------- check init_struc_data
    if None in init_struc_data.values():
        raise ValueError('init_struc_data includes None')

    # ---------- initialize
    n_selection = 1
    id_running = []
    id_select_hist = []

    # ---------- rslt_data, add and sort
    rslt_data['Select'] = pd.Series(dtype=int)
    rslt_data = rslt_data[['Select', 'Spg_num',
                           'Spg_sym', 'Spg_num_opt',
                           'Spg_sym_opt', 'E_eV_atom', 'Magmom', 'Opt']]
    pkl_data.save_rslt(rslt_data)

    # ---------- random select
    id_queueing = random.sample(list(range(len(init_struc_data))), rin.nselect_spkbo)

    # ---------- id_select_hist
    id_select_hist.append(id_queueing[:])

    # ---------- save for BO
    id_data = (n_selection, id_queueing, id_running, id_select_hist)
    pkl_data.save_spkbo_id(id_data)
    spkbo_data = (0)
    pkl_data.save_spkbo_data(spkbo_data)

    # ---------- status
    io_stat.set_common(stat, 'selection', n_selection)
    io_stat.set_id(stat, 'selected_id', id_queueing)
    io_stat.set_id(stat, 'id_queueing', id_queueing)
    io_stat.write_stat(stat)

    # ---------- out and log
    print('selected_id: {}'.format(' '.join(str(a) for a in id_queueing)))
    with open('cryspy.out', 'a') as fout:
        fout.write('selected_id: {}\n\n'.format(
            ' '.join(str(a) for a in id_queueing)))

    # build ase database for training of NN
    property_unit_dict = {
        "relaxation_energy": "eV",
        "energy": "eV",
        "forces": "eV/Ang",
        "stress": "eV/Ang/Ang/Ang",
        "structure_idx": "None",
    }
    if os.path.exists("./data/train.db"):
        os.remove("./data/train.db")
    database = spk.data.ASEAtomsData.create(
        datapath="./data/train.db",
        distance_unit="Ang",
        property_unit_dict=property_unit_dict,
    )
    database.update_metadata(structure_ids=[])

    # build NN-model
    n_models = 5
    n_interactions = 3
    n_features = 64
    cutoff = 5
    models = [
        spk.model.PropertyModel(
            datamodule=None,
            representation=spk.representation.SchNet(
                n_atom_basis=n_features,
                n_interactions=n_interactions,
                radial_basis=spk.nn.GaussianRBF(n_rbf=n_features, cutoff=cutoff),
                cutoff_fn=spk.nn.CosineCutoff(cutoff=cutoff)
            ),
            optimizer_cls=torch.optim.Adam,
            output=spk.atomistic.Atomwise(
                n_in=n_features,
                aggregation_mode="sum",
            ),
            properties="relaxation_energy",
            targets="relaxation_energy",
            loss_fn=torch.nn.MSELoss(),
            metrics={},
        ) for _ in range(n_models)
    ]
    ensemble = cryspyschnet.NNEnsemble(models=models, properties="relaxation_energy")
    torch.save(ensemble, "./data/initial_model.spk")
    torch.save(ensemble, "./data/best_model.spk")
