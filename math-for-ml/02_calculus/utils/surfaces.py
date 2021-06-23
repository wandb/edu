import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa

import scipy.stats
from scipy.signal import convolve


def axis_equal_3d(ax, center=0):
    # FROM StackO/19933125

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    if center == 0:
        centers = [0, 0, 0]
    else:
        centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def gauss_random_field(x, y, scale):
    """creates a correlated gaussian random field
    by first generating 'white' uncorrelated field
    and then using a low-pass filter based on convolution
    with a gaussian function to make a 'red-shifted', correlated
    gaussian random field
    """
    white_field = np.random.standard_normal(size=x.shape)

    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    gauss_rv = scipy.stats.multivariate_normal([0, 0], cov=np.ones(2))
    gauss_pdf = gauss_rv.pdf(pos)
    red_field = scale * convolve(white_field, gauss_pdf, mode='same')

    return red_field


def plot_loss_surface(loss, N, mesh_extent):
    mesh = np.linspace(-mesh_extent, mesh_extent, N)
    weights1, weights2 = np.meshgrid(mesh, mesh)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax._axis3don = False

    ax.plot_surface(weights1, weights2, loss(weights1, weights2),
                    rstride=2, cstride=2, linewidth=0.2, edgecolor='b',
                    alpha=1, cmap='Blues', shade=True)

    axis_equal_3d(ax, center=True)
