# Enhanced Vignes Equation (EVE): functions/driver.py
# Driver
# ---
# Jens Wagner, 11.01.2026
# ----------------------------------------------------------------------------------------------------------------------

# Import Packages and Modules
import torch
import glob
import numpy as np
import os
import pickle
import warnings

from functions.parser import parse
from functions.results import plot
from functions.models import iDSM, eDSM, SE, VE, IV
from utils.HANNA import HANNA, HANNA_Ensemble
from utils.Utils import predict, create_embedding_matrix
from utils.Utils import initiliaze_ChemBERTA
from sklearn.exceptions import InconsistentVersionWarning
from transformers import logging


# Driver
def drive(data):

    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)  # Mute sklearn warning
    logging.set_verbosity_error()   # Mute ChemBERTa warning

    data, X, warns = parse(data)

    # HADES Ensemble
    x = np.linspace(0, 1, 100)
    xs = torch.from_numpy(np.hstack((np.expand_dims(x, -1), np.expand_dims(1 - x, -1)))).float()
    x = torch.Tensor([[0., 1.]])
    X = torch.Tensor(X).unsqueeze(0)

    b_i = []
    b_j = []
    for p in glob.glob('params/HADES/HADES_*.pt'):
        m = iDSM()
        m.load_state_dict(torch.load(p, map_location='cpu'))
        m.eval()

        with torch.no_grad():
            pred_i, _ = m(X, x)
            pred_j, _ = m(X.flip(1), x)
            b_i.append(pred_i.squeeze())
            b_j.append(pred_j.squeeze())

    v_ideal = IV([data['vis_i'], data['vis_j']], xs)
    D_i = np.mean([x.detach().numpy() for x in b_i], 0) * SE(data['M_i'] * 1e-3, v_ideal * 1e-3, data['T_m']) * 1e9
    std_i = np.std([x.detach().numpy() for x in b_i], 0) * SE(data['M_i'] * 1e-3, v_ideal * 1e-3, data['T_m']) * 1e9
    D_j = np.mean([x.detach().numpy() for x in b_j], 0) * SE(data['M_j'] * 1e-3, v_ideal * 1e-3, data['T_m']) * 1e9
    std_j = np.std([x.detach().numpy() for x in b_j], 0) * SE(data['M_j'] * 1e-3, v_ideal * 1e-3, data['T_m']) * 1e9

    # HANNA Ensemble
    model_paths = [f'HANNA/HANNA_Production10_seed{seed}_Final.pt' for seed in range(42, 52)]
    scaler_path = f'HANNA/scaler_HANNA_Production.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    Embedding_BERT = 384
    hidden_size = 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HANNA_Ensemble(model_class=HANNA, model_paths=model_paths, Embedding_ChemBERT=Embedding_BERT,
                           nodes=hidden_size, device=device)
    model.eval()
    ChemBERTA, tokenizer = initiliaze_ChemBERTA(model_name="DeepChem/ChemBERTa-77M-MTR", device=None)

    x = np.linspace(0, 1, 100)

    lngi = []
    lngj = []
    dlngj_dxi = []
    dlngi_dxj = []
    tcf = []
    tcfs_ii = []
    tcfs_jj = []

    for x_i in x:

        x_i = np.array([x_i])
        embedding_matrix = create_embedding_matrix(data['SMILES_i'], data['SMILES_j'], data['T_m'], device,
                                                   ChemBERTA, tokenizer, x_i)
        x_pred, ln_gammas_pred = predict(embedding_matrix, scaler, model, device)

        ln_gamma_i = ln_gammas_pred[:, 0]
        ln_gamma_j = ln_gammas_pred[:, 1]
        lngi.append(ln_gamma_i.item())
        lngj.append(ln_gamma_j.item())

        grad_i = torch.autograd.grad(ln_gamma_i, x_pred, retain_graph=True)[0]
        grad_j = torch.autograd.grad(ln_gamma_j, x_pred)[0]
        dlngj_dxi.append(grad_j[:, 0].item())
        dlngi_dxj.append(-grad_i[:, 0].item())

        tcf_ii = 1 + x_i * grad_i[:, 0].item()
        tcf_jj = 1 + (1 - x_i) * (-grad_j[:, 0].item())
        tcf.append(np.mean([tcf_ii.item(), tcf_jj.item()]))
        tcfs_ii.append(tcf_ii.item())
        tcfs_jj.append(tcf_jj.item())

    # EVE Ensemble
    xs = torch.from_numpy(np.hstack((np.expand_dims(x, -1), np.expand_dims(1 - x, -1)))).float()
    X = torch.Tensor(X).repeat(100, 1, 1)
    dI = torch.stack([torch.tensor(i) for i in [lngi, dlngi_dxj, tcfs_ii, (xs[:, 1] * torch.Tensor(dlngi_dxj)).tolist(),
                                                [D_i[0]] * 100]], dim=1).type(torch.float)
    dJ = torch.stack([torch.tensor(i) for i in [lngj, dlngj_dxi, tcfs_jj, (xs[:, 0] * torch.Tensor(dlngj_dxi)).tolist(),
                                                [D_j[-1]] * 100]], dim=1).type(torch.float)

    pred = []
    for p in glob.glob('params/EVE/EVE_*.pt'):
        m = eDSM()
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        m.to(device)
        pred.append(m(X, xs, dI, dJ).cpu().detach().numpy())
    D_MS_ij = np.array(pred).mean(0)
    std_MS_ij = np.array(pred).std(0)

    # VE
    D_VE_ij = []
    for x_i in x:
        D_VE_ij.append(VE(np.array([D_i[0], D_j[-1]]), np.array([x_i, 1 - x_i])))

    data['x_i'] = x
    data['D_i'] = D_i
    data['std_i'] = std_i
    data['D_j'] = D_j
    data['std_j'] = std_j
    data['D_MS_ij'] = D_MS_ij
    data['std_MS_ij'] = std_MS_ij
    data['TCF'] = tcf
    data['D_VE_ij'] = D_VE_ij

    # Results
    path = '{}_{}_{}K'.format(data['SMILES_i'], data['SMILES_j'], str(data['T_m']))
    os.makedirs('RESULTS/{}'.format(path), exist_ok=True)
    plot(data, path)

    block = ''
    if warns:
        lines = "\n".join(warns)
        block = f"\nWarning!\n--------\n{lines}\n"

    txt = f'''EVE - Enhanced Vignes Equation (with HADES and HANNA)
-------------------------
    
Input
-----
Component i: {data['SMILES_i']}
Component j: {data['SMILES_j']}
Viscosity i / mPas: {data['vis_i']}
Viscosity j / mPas: {data['vis_j']}
Temperature / K: {data['T_m']}
{block}
Results saved to RESULTS/{path}.
'''

    print(txt)
