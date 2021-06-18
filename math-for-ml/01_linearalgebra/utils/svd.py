import matplotlib.pyplot as plt
import numpy as np


def compact(matrix, hermitian=False):
    return to_compact(*np.linalg.svd(matrix, hermitian=hermitian))


def to_compact(U, sigma, V_T, threshold=1e-8):
    U_compact, sigma_compact, V_T_compact = [], [], []

    for idx, singular_value in enumerate(sigma):
        # if the singular value isnt too close to 0
        if singular_value > threshold:
            # include that singular value in sigma_compact
            sigma_compact.append(singular_value)

            # add a row of V_T as a row of V_T_compact
            V_T_compact.append(V_T[idx])

            # add a column of U as a row of U_compact
            U_compact.append(U.T[idx])

        else:
            break

    # convert the lists to arrays
    V_T_compact = np.array(V_T_compact)
    # turn sigma
    sigma_compact = np.diag(sigma_compact)
    U_compact = np.array(U_compact).T

    return U_compact, sigma_compact, V_T_compact


def show_matrix(matrix, ax=None, add_colorbar=False):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        add_colorbar = True
    ax.axis("off")
    im = ax.matshow(matrix, cmap="Greys")
    if add_colorbar:
        plt.colorbar(im)


def show_svd(U, S, V_T):
    _, axs = plt.subplots(ncols=3, figsize=(18, 6))

    for matrix, ax in zip([U, S, V_T], axs):
        show_matrix(matrix, ax=ax)

    dim = max(max(U.shape), max(V_T.shape))
    for ax in axs:
        ax.set_ylim([dim - 0.5, 0 - 0.5])
        ax.set_xlim([0 - 0.5, dim - 0.5])
