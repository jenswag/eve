# Enhanced Vignes Equation (EVE): functions/parser.py
# Parser
# ---
# Jens Wagner, 11.01.2026
# ----------------------------------------------------------------------------------------------------------------------

# Import Packages and Modules
import numpy as np

from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors, NumHeteroatoms, RingCount, NumAliphaticRings, NumAromaticRings


# Parser
def parse(data):

    warnings = []

    smiles_i = data['SMILES_i']
    smiles_j = data['SMILES_j']
    mol_i = MolFromSmiles(smiles_i)  # Component i
    mol_j = MolFromSmiles(smiles_j)  # Component j

    assert smiles_i != smiles_j, 'Invalid input: Component i and component j must be different components.'

    # Molar mass
    mm_i = ExactMolWt(MolFromSmiles(smiles_i))  # Molar mass i [g/mol]
    mm_j = ExactMolWt(MolFromSmiles(smiles_j))  # Molar mass j [g/mol]

    if mm_i > 1000:
        warnings.append('Molar mass of component i is outside the validated applicability domain of HADES and EVE.')
    if mm_j > 1000:
        warnings.append('Molar mass of component j is outside the validated applicability domain of HADES and EVE.')

    def heavierCl(mol):
        return any(atom.GetAtomicNum() > 17 for atom in mol.GetAtoms())

    def charged(mol):
        return any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())

    if heavierCl(mol_i):
        warnings.append('Component i contains atoms outside the validated applicability domain of HADES and EVE.')
    if heavierCl(mol_j):
        warnings.append('Component j contains atoms outside the validated applicability domain of HADES and EVE.')

    if charged(mol_i):
        warnings.append('Component i carries a formal charge and is outside the validated applicability domain of HADES and EVE.')
    if charged(mol_j):
        warnings.append('Component j carries a formal charge and is outside the validated applicability domain of HADES and EVE.')

    # Number of non-H atoms in Solute i and Solvent j
    atoms_i, atoms_j = len(mol_i.GetAtoms()), len(mol_j.GetAtoms())

    # Calculate H-Sites i
    if smiles_i == 'O' or smiles_i == '[2H]O[2H]':  # Special case: Water and Deuterium oxide
        sites_i = np.array([0.5, 0.5])  # H-Acceptors, H-Donors
    else:
        sites_i = np.array([NumHAcceptors(mol_i),  # H-Acceptors
                            NumHDonors(mol_i)]  # H-Donors
                           )

    # Calculate H-Sites j
    if smiles_j == 'O' or smiles_j == '[2H]O[2H]':  # Special case: Water and Deuterium oxide
        sites_j = np.array([0.5, 0.5])  # H-Acceptors, H-Donors
    else:
        sites_j = np.array([NumHAcceptors(mol_j),  # H-Acceptors
                            NumHDonors(mol_j)]  # H-Donors
                           )

    data['M_i'] = mm_i  # Molar mass i
    data['Atoms_i'] = atoms_i  # Atoms i (no H)
    data['Hetero_i'] = NumHeteroatoms(mol_i)  # Hetero atoms i (no H, no C)
    data['Halos_i'] = sum(1 for atom in mol_i.GetAtoms() if atom.GetSymbol() in ['F', 'Cl'])  # Halogens i (F, Cl)
    data['Rings_i'] = RingCount(mol_i)  # Rings i
    data['HAcc_i'] = sites_i[0]  # H-Acceptors i
    data['HDon_i'] = sites_i[1]  # H-Donors i

    data['M_j'] = mm_j  # Molar mass j
    data['Atoms_j'] = atoms_j  # Atoms j (no H)
    data['Hetero_j'] = NumHeteroatoms(mol_j)  # Hetero atoms j (no H, no C)
    data['Halos_j'] = sum(1 for atom in mol_j.GetAtoms() if atom.GetSymbol() in ['F', 'Cl'])  # Halogens j (F, Cl)
    data['Rings_j'] = RingCount(mol_j)  # Rings j
    data['HAcc_j'] = sites_j[0]  # H-Acceptors j
    data['HDon_j'] = sites_j[1]  # H-Donors j

    X = [[
        mm_i * 1e-3,    # M i
        data['HAcc_i'] / data['Atoms_i'],   # rHAcc i
        data['HDon_i'] / data['Atoms_i'],   # rHDon i
        data['Hetero_i'] / data['Atoms_i'],     # rHetero i
        data['Halos_i'] / data['Atoms_i'],  # rHalos i
        int(data['Rings_i'] >= 1),   # Rings i
        ],

        [
        mm_j * 1e-3,  # M j
        data['HAcc_j'] / data['Atoms_j'],  # rHAcc j
        data['HDon_j'] / data['Atoms_j'],  # rHDon j
        data['Hetero_j'] / data['Atoms_j'],  # rHetero j
        data['Halos_j'] / data['Atoms_j'],  # rHalos j
        int(data['Rings_j'] >= 1)  # Rings j
    ]]

    return data, X, warnings
