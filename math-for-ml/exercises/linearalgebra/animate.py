import matplotlib
import numpy as np

unit_square_mesh = {'delta': 0.1,
                    'x_min': 0,
                    'x_max': 1,
                    'y_min': 0,
                    'y_max': 1}

all_quadrants_mesh = {'delta': 0.1,
                      'x_min': -1,
                      'x_max': 1,
                      'y_min': -1,
                      'y_max': 1}

COLOR_EIGVEC = "#003262"
COLOR_BASIS = "#FDB515"
CMAP_MESH = matplotlib.colors.ListedColormap(np.array([1., 1., 1., 1.]))


def setup_plot(T, mesh_properties=unit_square_mesh,
               animate_basis=False):
    """
    Setup the plot and axes for animating a linear transformation T.

    If asked, animate the basis vectors as arrows.

    Parameters
    ----------
    T                : 2x2 matrix representing a linear transformation
    mesh_properties  : dictionary that defines properties of meshgrid of points
                       that will be plotted and transformed.
                       needs to have five keys:
                         'delta' - mesh spacing
                         '{x,y}_{min,max}' - minium/maximum value on x/y axis
    animate_basis    : if True, animate the basis vectors

    Returns
    -------
    returns are meant to be consumed by animate_transformation

    fig       : matplotlib figure containing axes
    ax        : matplotlib axes containing scatter
    animate   : callable for use with matplotlib.FuncAnimation
    n_frames  : integer number of frames
    """
    global basis_vecs  # hack to get around arrows not being animatable
    T = np.asarray(T)

    xs, ys = make_mesh(mesh_properties)
    colors = np.linspace(0, 1, num=xs.shape[0] * xs.shape[1])

    with matplotlib.style.context("dark_background"):
        fig = matplotlib.figure.Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

    scatter = plot_mesh(ax, xs, ys, colors)

    start, end, _ = compute_trajectories(T, scatter)

    basis_vecs = setup_basis(animate_basis, ax)

    mn, mx = calculate_axis_bounds(start, end)
    lim = max(abs(mn), abs(mx))
    mn, mx = -lim, lim

    draw_coordinate_axes(mn, mx, ax=ax)
    set_axes_lims(mn, mx, ax=ax)

    scatter_offsets, arrow_positions = precompute_animation(T, scatter, animate_basis)
    n_frames = len(scatter_offsets)

    def animate(ii):
        global basis_vecs
        scatter.set_offsets(scatter_offsets[ii])
        basis_vecs = update_basis(basis_vecs, arrow_positions, ii, ax)

    return fig, ax, animate, n_frames


def make_mesh(mesh_props):
    num_dimensions = 2

    mins = (mesh_props['x_min'], mesh_props['y_min'])
    maxs = (mesh_props['x_max'], mesh_props['y_max'])
    delta = mesh_props['delta']

    for idx in range(num_dimensions):
        assert mins[idx] < maxs[idx], "min can't be bigger than max!"

    ranges = [np.arange(mins[idx], maxs[idx] + delta, delta)
              for idx in range(num_dimensions)]

    xs, ys = np.meshgrid(*ranges)

    return xs, ys


def plot_mesh(ax, xs, ys, colors):
    h = ax.scatter(xs.flatten(), ys.flatten(),
                   alpha=0.7, edgecolor='none',
                   s=36, linewidth=2,
                   zorder=6,
                   c=colors, cmap=CMAP_MESH)

    return h


def compute_trajectories(T, scatter):

    starting_positions = scatter.get_offsets()
    ending_positions = np.dot(T, starting_positions.T).T
    delta_positions = ending_positions-starting_positions

    return starting_positions, ending_positions, delta_positions


def set_axes_lims(mn, mx, ax):

    ax.set_ylim([mn, mx])
    ax.set_xlim([mn, mx])

    return


def calculate_axis_bounds(starting_positions, ending_positions, buffer_factor=1.1):
    # axis bounds to include starting and ending positions of each point

    mn = buffer_factor * min(np.min(starting_positions), np.min(ending_positions))
    mx = buffer_factor * max(np.max(starting_positions), np.max(ending_positions))

    if mn == 0:
        mn -= 0.1
    if mx == 0:
        mx += 0.1

    return mn, mx


def draw_coordinate_axes(mn, mx, ax):

    ax.hlines(0, mn, mx, zorder=4, linewidth=4, color='grey')
    ax.vlines(0, mn, mx, zorder=4, linewidth=4, color='grey')

    return


def setup_basis(animate_basis, ax):
    if not animate_basis:
        return None
    else:
        basis1 = plot_vector([1, 0], ax, color=COLOR_BASIS, label="basis vector")
        basis2 = plot_vector([0, 1], ax, color=COLOR_BASIS, label="basis vector")
        return [basis1, basis2]


def update_basis(basis_vecs, arrow_positions, index, ax):
    if basis_vecs is None:
        return None
    else:
        [vec.remove() for vec in basis_vecs]
        positions = arrow_positions[index]
        basis1 = plot_vector(positions[0], ax, color=COLOR_BASIS, label="basis vector")
        basis2 = plot_vector(positions[1], ax, color=COLOR_BASIS, label="basis vector")
        return [basis1, basis2]


def plot_vector(v, ax, color, label=None):
    return ax.arrow(0, 0, v[0], v[1], zorder=5,
                    linewidth=6, color=color, head_width=0.1, label=label)


def make_rotation(theta):
    rotation_matrix = [[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]]
    return np.asarray(rotation_matrix)


def precompute_animation(T, scatter, animate_basis=False, delta_t=0.01):

    T = np.asarray(T)

    start, _, delta = compute_trajectories(T, scatter)

    Id = np.eye(2)

    ts = np.arange(0, 1 + delta_t, delta_t)

    not_zeros = not(np.all(T == np.zeros(2)))

    offsets = [scatter.get_offsets()]
    arrow_positions = [[[1, 0], [0, 1]]]  # == Id.T

    if ((T[0, 0] == T[1, 1]) & (T[0, 1] == -1 * T[1, 0])) & \
            not_zeros:

        z = complex(T[0, 0], T[1, 0])
        dz = z ** (1 / len(ts))

        dT = [[dz.real, -dz.imag], [dz.imag, dz.real]]
        for idx, t in enumerate(ts):
            dT_toN = np.linalg.matrix_power(dT, idx+1)
            offsets.append(np.dot(dT_toN, start.T).T)
            arrow_positions.append(np.dot(dT_toN, Id).T)

    else:
        for idx, t in enumerate(ts):
            offsets.append(t * (np.dot(T, start.T).T) +
                           (1 - t) * np.dot(Id, start.T).T)
            arrow_positions.append((t * T +
                                   (1 - t) * Id).T)

    return offsets, arrow_positions
