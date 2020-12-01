import autograd
import autograd.numpy as np
import ipywidgets
from ipywidgets import interact
import matplotlib.pyplot as plt

BLUE = "#99c8f3"
GOLD = "#ffcc33"
RED = "#ebacac"
PURPLE = "#918ce3"
GRAY = "#595959"

TEST_PT_COLOR = RED
EPSILON_COLOR = BLUE
GRADIENT_COLOR = GOLD
FUNCTION_COLOR = GRAY
ERROR_COLOR = PURPLE


class GradApprox(object):

    def __init__(self, f, grad_f, starting_center):
        self.f = f
        self.grad_f = grad_f

        self.update(starting_center)

    def __call__(self, x):
        out = self.grad_f(self.center) * (x - self.center) + self.f(self.center)
        return out

    def update(self, center):
        self.center = center

    def error(self, x):
        return self(x) - self.f(x)


def setup(f, mn=-1., mx=1., N=500):

    midpoint = (mx + mn) / 2

    grad_f = autograd.grad(f)
    try:
        autograd.grad(mn)
        autograd.grad(mx)
        autograd.grad(midpoint)
    except:  # noqa
        raise("Error computing gradient of function")

    fig, ax = plt.subplots()

    _, = add_func_line(f, mn, mx, N, ax)

    add_axes(mn, mx, ax)

    starting_x, starting_y = midpoint, f(midpoint)
    approximator = GradApprox(f, grad_f, starting_x)

    center_point = add_center_point(starting_x, starting_y, ax)
    gradient_line = add_gradient_line(approximator, mn, mx, ax)
    test_point = add_test_point(starting_x, starting_y, ax)
    error_line = add_error_line(starting_x, starting_y, ax)
    epsilon_line = add_epsilon_line(starting_x, starting_y, ax)

    def update_point_and_gradient(x):
        y = f(x)
        center_point.set_offsets([x, y])
        approximator.center = x

        endpoints = get_endpoints(approximator, mn, mx)
        gradient_line.set_data(*endpoints)

        update_error_line()
        update_epsilon_line()

    def update_test_point(x):
        test_offsets = [x, f(x)]
        test_point.set_offsets(test_offsets)

        update_error_line()
        update_epsilon_line()

    def update_error_line():
        test_offsets = np.array(test_point.get_offsets().data)[0]
        x = test_offsets[0]
        approx_offsets = [x, approximator(x)]
        test_offsets = [x, f(x)]

        error_xs = test_offsets[0], approx_offsets[0]
        error_ys = test_offsets[1], approx_offsets[1]

        error_line.set_data(error_xs, error_ys)

    def update_epsilon_line():
        test_offsets = np.array(test_point.get_offsets().data)[0]
        x = approximator.center

        epsilon_xs = x, test_offsets[0]
        epsilon_ys = f(x), f(x)

        epsilon_line.set_data(epsilon_xs, epsilon_ys)

    def update_points(center_point=starting_x, test_point=starting_x):
        update_point_and_gradient(center_point)
        update_test_point(test_point)

    eps_size = (mx - mn) / 2000
    center_point_adjust_size = 5 * eps_size

    def interactor():
        return interact(update_points,
                        center_point=ipywidgets.FloatSlider(
                            value=starting_x, min=mn, max=mx,
                            step=center_point_adjust_size,
                            description=r"$x$"),
                        test_point=ipywidgets.FloatSlider(
                            value=starting_x, min=mn, max=mx,
                            step=eps_size,
                            description=r"$x + \varepsilon$"))

    return interactor


def add_func_line(f, mn, mx, N, ax, color=FUNCTION_COLOR):
    xs = np.linspace(mn, mx, num=N)
    ys = f(xs)

    line, = ax.plot(xs, ys, color=color, lw=3, zorder=3)

    return line,


def get_endpoints(approximator, mn, mx):
    y_mn = approximator(mn)
    y_mx = approximator(mx)

    return [[mn, mx], [y_mn, y_mx]]


def add_gradient_line(approximator, mn, mx, ax, color=GRADIENT_COLOR):
    endpoints = get_endpoints(approximator, mn, mx)
    gradient_line, = ax.plot(*endpoints, lw=3, color=color)

    return gradient_line


def add_center_point(x, y, ax, color=FUNCTION_COLOR):
    cp = ax.scatter(x, y, color=color, zorder=4, s=256)
    return cp


def add_test_point(x, y, ax, color=TEST_PT_COLOR):
    tp = ax.scatter(x, y, color=color, zorder=4, s=144)
    return tp


def add_error_line(x, y, ax, color=ERROR_COLOR):
    err_line, = ax.plot([x, x], [y, y], lw=3, color=color, alpha=0.7, zorder=2)
    return err_line


def add_epsilon_line(x, y, ax, color=EPSILON_COLOR):
    eps_line, = ax.plot([x, x], [0, 0], lw=3, color=color, zorder=3)
    return eps_line


def add_axes(mn, mx, ax):

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.hlines(0, *xlims)
    ax.vlines(0, *ylims)

    ax.set_xlim(mn, mx)
    ax.set_ylim(*ylims)
