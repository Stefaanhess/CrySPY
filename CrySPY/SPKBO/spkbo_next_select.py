'''
Selection in Bayesian optimization with SchNetPack
'''

import configparser

import numpy as np

from ..BO.combo_cryspy import Policy_cryspy
from ..IO import io_stat, out_results, pkl_data
from ..IO import read_input as rin
import torch
import torch.nn as nn
import cryspyschnet
import schnetpack as spk
from scipy.stats import norm
from pymatgen.io.ase import AseAtomsAdaptor


def next_select(stat, rslt_data, bo_id_data):
    # ---------- out and log
    with open('cryspy.out', 'a') as fout:
        fout.write('# ------ SPK Bayesian optimization\n')
    print('# ------ SPK Bayesian optimization')

    # ---------- bo_id_data and bo_data
    n_selection, id_running, id_queueing, id_select_hist = bo_id_data

    # ---------- n_selection
    n_selection += 1

    # prepare nn model
    device = "cpu"
    best_value, bo_epoch, training_epoch = pkl_data.load_spkbo_data()
    # todo hardcoded n_models
    trained_models = [
        torch.load(f"./runs/job{training_epoch}_{i}/best_inference_model", map_location=device) for i in range(5)
    ]
    ensemble = cryspyschnet.spk_tools.NNEnsemble(nn.ModuleList(trained_models), properties=["relaxation_energy"])
    atoms_converter = cryspyschnet.spk_tools.AtomsConverter(
        transforms=[
            spk.transform.TorchNeighborList(cutoff=5.),
            spk.transform.CastTo32(),
        ]
    )
    # load all structures and compute expected improvement for unseen candidates
    initial_structures = pkl_data.load_init_struc()
    predicted_energies, uncertainties = [], []
    id_select_hist_flat = [item for subl in id_select_hist for item in subl]
    for i, structure in initial_structures.items():
        structure = structure
        if i in id_select_hist_flat:
            predicted_energies.append(1)
            uncertainties.append(1e-3)
        else:
            atms = AseAtomsAdaptor.get_atoms(structure=structure)
            spk_input = atoms_converter(atms)
            spk_input = {k: v.to(device) for k, v in spk_input.items()}
            predicted_energy, uncertainty = \
                [pred["relaxation_energy"] for pred in ensemble(spk_input)]
            predicted_energies.append(predicted_energy.item())
            uncertainties.append(uncertainty.item())

    best_value, bo_epoch, training_started = pkl_data.load_spkbo_data()

    ei = expected_improvement(
        predictions=np.array(predicted_energies),
        uncertainties=np.array(uncertainties),
        best_value=best_value,
        mask=id_select_hist_flat,
    )

    # get ids of candidate structures
    # todo: double check selection of ei
    nselect = min(rin.nselect_spkbo, len(initial_structures)-len(id_select_hist_flat))
    id_queueing = np.argsort(ei)[-nselect:].tolist()

    # ---------- id_select_hist
    id_select_hist.append(id_queueing[:])
    out_results.out_bo_id_hist(id_select_hist)

    # ---------- save
    id_data = (n_selection, id_queueing, id_running, id_select_hist)
    pkl_data.save_spkbo_id(id_data)

    # ---------- status
    io_stat.set_common(stat, 'selection', n_selection)
    io_stat.set_id(stat, 'selected_id', id_queueing)
    io_stat.set_id(stat, 'id_queueing', id_queueing)
    io_stat.write_stat(stat)

    # ---------- out and log
    print('\n\n# ---------- Selection: {}'.format(n_selection))
    print('selected_id: {}'.format(' '.join(str(a) for a in id_queueing)))
    with open('cryspy.out', 'a') as fout:
        fout.write('\n\n# ---------- Selection: {}\n'.format(n_selection))
        fout.write('selected_id: {}\n\n'.format(
            ' '.join(str(a) for a in id_queueing)))


def expected_improvement(
    predictions,
    uncertainties,
    best_value,
    mask=None,
    xi=0.01,
):
    with np.errstate(divide="warn"):
        imp = -(predictions - best_value) - xi
        Z = imp / uncertainties
        ei = imp * norm.cdf(Z) + uncertainties * norm.pdf(Z)

    if mask is not None:
        ei[mask] = -1
    return ei
