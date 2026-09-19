# Enhanced Vignes Equation (EVE): EVE.py
# EVE Ensemble with HADES and HANNA
# ---
# Jens Wagner, 11.01.2026
# ----------------------------------------------------------------------------------------------------------------------

# Import Packages and Modules
from functions.driver import drive

# === INPUT ===
i = 'CC(C)=O'     # SMILES i
j = 'C1CCCCC1'   # SMILES j
v_i = 0.307       # Viscosity i [mPas]
v_j = 0.8932      # Viscosity j [mPas]
T = 298.15  # Temperature [K]
# ===

# Run EVE
data = {'SMILES_i': i,
        'SMILES_j': j,
        'vis_i': v_i,
        'vis_j': v_j,
        'T_m': T,
        }

drive(data)
