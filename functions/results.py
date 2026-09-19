# Enhanced Vignes Equation (EVE): functions/results
# Results
# ---
# Jens Wagner, 09.01.2026
# ----------------------------------------------------------------------------------------------------------------------

# Import Packages and Modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Plot
def plot(data, path):

    plt.rcParams.update({'font.size': 8, 'font.family': 'Arial'})

    fig, ax = plt.subplots(figsize=(12 / 2.54, 12 / 2.54))

    ax.tick_params(direction='in', top=True, right=True)
    ax.set_title('{} ($i$) and {} ($j$) at {} K'.format(data['SMILES_i'], data['SMILES_j'], str(data['T_m'])),
                  loc='center')
    ax.set_ylabel(r'$D_{ij}$, $Đ_{ij}$ ${\cdot}$ $10^{9}$ / m$^2$ s$^{-1}$')
    ax.set_xlabel('$x_i$ / mol mol$^{-1}$')
    ax.set_xlim([0, 1])

    xs = np.hstack((np.expand_dims(data['x_i'], -1), np.expand_dims(1 - data['x_i'], -1)))

    #ax.plot(xs[:, 0], data['D_VE_ij'], c='steelblue', linestyle='--', label='$D_i$')
    ax.plot(xs[:, 0], data['D_i'], c='limegreen', label='$D_i$')
    # ax.fill_between(xs[:, 0], data['D_i'] - data['std_i'], data['D_i'] + data['std_i'], color='limegreen', alpha=0.3)
    # ax.fill_between(xs[:, 0], data['D_i'] - 2 * data['std_i'], data['D_i'] + 2 * data['std_i'], color='limegreen',
    #                alpha=0.15)

    ax.plot(xs[:, 0], data['D_j'], c='forestgreen', label='$D_j$')
    # ax.fill_between(xs[:, 0], data['D_j'] - data['std_j'], data['D_j'] + data['std_j'], color='forestgreen', alpha=0.3)
    # ax.fill_between(xs[:, 0], data['D_j'] - 2 * data['std_j'], data['D_j'] + 2 * data['std_j'], color='forestgreen',
    #                alpha=0.15)


    ax.plot(xs[:, 0], data['D_MS_ij'], c='dodgerblue', label='$Đ_{ij}$')
    # ax.fill_between(xs[:, 0], data['D_MS_ij'] - data['std_MS_ij'], data['D_MS_ij'] + data['std_MS_ij'],
    #                 color='dodgerblue', alpha=0.3)
    # ax.fill_between(xs[:, 0], data['D_MS_ij'] - 2 * data['std_MS_ij'], data['D_MS_ij'] + 2 * data['std_MS_ij'],
    #                 color='dodgerblue', alpha=0.15)

    ax.plot(xs[:, 0], data['D_MS_ij'] * data['TCF'], c='steelblue', label='$D_{ij}$')
    # ax.fill_between(xs[:, 0], (data['D_MS_ij'] - data['std_MS_ij']) * data['TCF'],
    #                 (data['D_MS_ij'] + data['std_MS_ij']) * data['TCF'],
    #                 color='steelblue', alpha=0.3)
    # ax.fill_between(xs[:, 0], (data['D_MS_ij'] - 2 * data['std_MS_ij']) * data['TCF'],
    #                 (data['D_MS_ij'] + 2 * data['std_MS_ij']) * data['TCF'],
    #                 color='steelblue', alpha=0.15)

    ax.legend(loc='upper left', frameon=False)

    plt.tight_layout()
    # plt.show()
    plt.savefig('RESULTS/{}/{}.pdf'.format(path, path), dpi=600, transparent=True)
    plt.savefig('RESULTS/{}/{}.png'.format(path, path), dpi=600, transparent=True)
    plt.close()

    data['D_FL_ij'] = data['D_MS_ij'] * data['TCF']
    data['std_FL_ij'] = data['std_MS_ij'] * data['TCF']

    df = pd.DataFrame({k: data[k] for k in ['x_i', 'D_i', 'D_j', 'D_MS_ij', 'D_FL_ij']})
    df.to_excel('RESULTS/{}/{}.xlsx'.format(path, path), index=False)
