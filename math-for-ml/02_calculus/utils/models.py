import inspect
import math
import time

import autograd
import autograd.numpy as np
from ipywidgets import interact
import ipywidgets as widgets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa
import pandas as pd
import wandb


wandb.jupyter.logger.setLevel("CRITICAL")


class Model(object):
    """Base class for the other *Model classes.
    Implements plotting and interactive components
    and interface with Parameters object."""

    def __init__(self, input_values, model_inputs, parameters, funk,
                 use_wandb=False, entity=None, project=None, wandb_path="./wandb"):
        self.input_values = input_values
        self.model_inputs = np.atleast_2d(model_inputs)
        self.parameters = parameters
        self.funk = funk
        self.use_wandb = use_wandb

        if self.use_wandb:
            self.setup_wandb(entity, project, wandb_path)

        self.plotted = False
        self.has_data = False
        self.show_MSE = False

        self._interactor = None

    def plot(self):
        if not self.plotted:
            self.initialize_plot()
        else:
            self.artist.set_data(self.input_values, self.outputs)
            self.fig.canvas.draw()
        return

    @property
    def outputs(self):
        return np.squeeze(self.funk(self.model_inputs))

    def initialize_plot(self):
        self.fig = plt.figure()
        self.ax = plt.subplot(111)
        self.artist, = plt.plot(self.input_values,
                                self.outputs,
                                linewidth=4)
        self.plotted = True
        self.ax.set_ylim([-10, 10])
        self.ax.set_xlim([-10, 10])

    def make_interactive(self):
        """called in a cell after Model.plot()
        to make the plot interactive."""

        @interact(**self.parameters.widgets)
        def make(**kwargs):
            frame = inspect.currentframe()
            args, _, _, values = inspect.getargvalues(frame)
            kwargs = values['kwargs']

            for parameter in kwargs.keys():
                self.parameters.dict[parameter] = kwargs[parameter]

            self.parameters.update()
            self.plot()

            MSE = self.compute_MSE()
            if self.show_MSE:
                print("MSE:\t"+str(MSE))

            if self.use_wandb:
                log_dict = {"MSE": MSE}
                log_dict.update(self.parameters.dict)
                self.log_wandb(log_dict)

            return

        self._interactor = make

        return

    def run_gd(self, n=25, lr=0.1, delta_t=0.1):
        if self._interactor is None:
            self.make_interactive

        def loss(parameters):
            outputs = self.funk(np.array(self.data_inputs), parameters)
            squared_errors = np.square(np.array(self.correct_outputs) - outputs)
            MSE = np.mean(squared_errors)
            return MSE

        loss_grad = autograd.elementwise_grad(loss)
        self._loss = loss
        self._loss_grad = loss_grad

        for _ in range(n):
            gradient = loss_grad(self.parameters.values)
            if gradient.ndim >= 2:
                gradient = np.squeeze(gradient)
            update_dict = self.make_grad_update_dict(np.atleast_1d(gradient), lr)
            self._interactor(**update_dict)
            time.sleep(delta_t)

    def make_grad_update_dict(self, gradient, lr=0.1):
        update_dict = {}
        for ii, key in enumerate(self.parameters.dict):
            update_dict[key] = self.parameters.dict[key] - lr * gradient[ii]

        return update_dict

    def set_data(self, xs, ys):
        self.data_inputs = self.transform_inputs(xs)
        self.correct_outputs = ys

        if self.has_data:
            _offsets = np.asarray([xs, ys]).T
            self.data_scatter.set_offsets(_offsets)
        else:
            self.data_scatter = self.ax.scatter(xs, ys,
                                                color='k', alpha=0.5, s=72)
            self.has_data = True

    def compute_MSE(self):
        """Used in fitting models lab to display MSE performance
        for hand-fitting exercises"""
        outputs = np.squeeze(self.funk(self.data_inputs))
        squared_errors = np.square(self.correct_outputs - outputs)
        MSE = np.mean(squared_errors)
        return MSE

    def setup_wandb(self, entity, project, wandb_path):
        wandb.init(entity=entity, project=project, dir=wandb_path, anonymous="allow")

    def log_wandb(self, log_dict):
        wandb.log(log_dict)

    def __delete__(self):
        wandb.join()


class LinearModel(Model):
    """A linear model is a model whose transform is
    the dot product of its parameters with its inputs.
    Technically really an affine model, as LinearModel.transform_inputs
    adds a bias term."""

    def __init__(self, input_values, parameters, model_inputs=None, **kwargs):

        if model_inputs is None:
            model_inputs = self.transform_inputs(input_values)
        else:
            model_inputs = model_inputs

        def funk(inputs, parameters=None):
            if parameters is None:
                parameters = self.parameters.values
            return np.dot(parameters, inputs)

        Model.__init__(self, input_values, model_inputs, parameters, funk, **kwargs)

    def transform_inputs(self, input_values):
        model_inputs = [[1]*input_values.shape[0], input_values]
        return model_inputs


class LinearizedModel(LinearModel):
    """A linearized model is a linear model applied to
    inputs that have been transformed according to
    1 or more transforms."""

    def __init__(self, transforms, input_values, parameters, **kwargs):

        self.transforms = [lambda x: np.power(x, 0),
                           lambda x: x] + transforms

        model_inputs = self.transform_inputs(input_values)

        LinearModel.__init__(
            self, input_values, parameters, model_inputs=model_inputs, **kwargs)

    def transform_inputs(self, input_values):
        transformed_inputs = []

        for transform in self.transforms:
            transformed_inputs.append(transform(input_values))

        return transformed_inputs


class NonlinearModel(Model):
    """A nonlinear model is a model whose outputs
    are related to its inputs by transforms that
    depend on its parameters."""

    def __init__(self, input_values, parameters, transform, **kwargs):

        def funk(inputs, parameters=None):
            if parameters is None:
                parameters = self.parameters.values
            return transform(parameters, inputs)

        Model.__init__(self, input_values, input_values, parameters, funk, **kwargs)

    def transform_inputs(self, input_values):
        return input_values


class Parameters(object):
    """Tracks and updates parameter values and metadata, like range and identity,
    for parameters of a model. Interfaces with widget-making tools
    via the Model class to make interactive widgets for Model plots."""

    def __init__(self, defaults, ranges, names=None):
        assert len(defaults) == len(ranges), \
               "must have default and range for each parameter"

        self.values = np.atleast_2d(defaults)

        self.num = len(defaults)

        self._zip = zip(defaults, ranges)

        if names is None:
            self.names = ['parameter_'+str(idx) for idx in range(self.num)]
        else:
            self.names = names

        # if len(self.values.shape) > 1:
        #     values_for_dict = np.squeeze(self.values)
        # else:
        #     values_for_dict = self.values

        self.dict = dict(zip(self.names, self.values))
        # self.dict = dict(zip(self.names,np.squeeze(self.values)))

        self.defaults = defaults
        self.ranges = ranges

        self.make_widgets()

    def make_widgets(self):
        self._widgets = [self.make_widget(parameter, idx)
                         for idx, parameter
                         in enumerate(self._zip)]

        self.widgets = {self.names[idx]: _widget
                        for idx, _widget
                        in enumerate(self._widgets)}

    def make_widget(self, parameter, idx):
        default = parameter[0]
        range = parameter[1]
        name = self.names[idx]
        return widgets.FloatSlider(value=default,
                                   min=range[0],
                                   max=range[1],
                                   step=0.01,
                                   description=name
                                   )

    def update(self):
        sorted_keys = sorted(self.dict.keys())
        self.values = np.atleast_2d([self.dict[key] for key in sorted_keys])

###
# Helper Functions
###


def cleanup(model):
    for widget in model.parameters._widgets:
        widget.close()
    plt.close(model.fig)


def make_default_parameters(number, rnge=1, names=None):
    defaults = [0]*number
    ranges = [[-rnge, rnge]]*number
    return Parameters(defaults, ranges, names)


def make_sine_parameters(degree):
    defaults = [(-1) ** (n//2) / math.factorial(n)
                if n % 2 != 0
                else 0
                for n in range(degree+1)]
    ranges = [[-1, 1]] * (degree+1)
    return Parameters(defaults, ranges, names=make_polynomial_parameter_names(degree))


def make_polynomial_transforms(max_degree):
    curried_power_transforms = [
        lambda n: lambda x: np.power(x, n) for _ in range(2, max_degree+1)]
    transforms = [curried_power_transform(n)
                  for curried_power_transform, n
                  in zip(curried_power_transforms, range(2, max_degree+1))
                  ]
    return transforms


def make_polynomial_parameters(max_degree, rnge=1):
    parameter_names = make_polynomial_parameter_names(max_degree)
    return make_default_parameters(max_degree+1, rnge=rnge,
                                   names=parameter_names)


def make_polynomial_parameter_names(max_degree):
    return ['x^'+str(n) for n in range(max_degree+1)]


def make_linear_parameters():
    return make_polynomial_parameters(1)


def make_linearized_parameters(transforms):
    return make_default_parameters(len(transforms)+2)


def make_trig_transform(f):
    return lambda theta, x: f(theta*x)


# def make_nonlinear_parameters(default, range_tuple):
#     return Parameters(default, range_tuple, ['theta'])


def random_weights(d=1):
    return np.random.standard_normal(size=(d, 1))


def plot_model(x, y):
    plt.figure()
    plt.plot(np.squeeze(x), np.squeeze(y), linewidth=4)


def setup_x(N, x_mode='linspace', x_range=[-2, 2]):
    if x_mode == 'uniform':
        x = uniform_inputs(x_range[0], x_range[1], N)
    elif x_mode == 'gauss':
        xWidth = x_range[1] - x_range[0]
        mu = (x_range[1] + x_range[0])/2
        sd = xWidth/3
        x = gauss_inputs(mu, sd, N)
    elif x_mode == 'linspace':
        x = linspace_inputs(x_range[0], x_range[1], N)
    else:
        print("mode unrecognized, defaulting to linspace")
        x = linspace_inputs(-1, 1, N)

    return x


def random_linear_model(noise_level, x_mode='linspace', N=1000):
    if x_mode == 'uniform':
        x = uniform_inputs(-1, 1, N)
    elif x_mode == 'gauss':
        x = gauss_inputs(0, 1, N)
    elif x_mode == 'linspace':
        x = linspace_inputs(-1, 1, N)
    else:
        print("mode unrecognized, defaulting to linspace")
        x = linspace_inputs(-1, 1, N)

    all_ones = np.ones(N)
    regressors = np.vstack([x, all_ones])

    linear_weights = random_weights(2)

    epsilon = noise_level * np.random.standard_normal(size=(1, N))

    linear_y = np.dot(linear_weights.T, regressors) + epsilon

    linear_model_dataframe = pd.DataFrame.from_dict({'x': np.squeeze(x),
                                                     'y': np.squeeze(linear_y)})

    return linear_model_dataframe


def random_linearized_model(noise_level, max_degree,
                            x_mode='linspace', x_range=[-1, 1], N=1000):

    x = setup_x(N, x_mode=x_mode, x_range=x_range)

    all_ones = np.ones(N)

    poly_regressors = [np.power(x, n) for n in range(2, max_degree+1)]

    regressors = np.vstack([x, all_ones] + poly_regressors)

    weights = random_weights(max_degree + 1)

    epsilon = noise_level * np.random.standard_normal(size=(1, N))

    linear_y = np.dot(weights.T, regressors) + epsilon

    linearized_model_dataframe = pd.DataFrame.from_dict({'x': np.squeeze(x),
                                                         'y': np.squeeze(linear_y)})

    return linearized_model_dataframe


def random_nonlinear_model(noise_level, function,
                           x_mode='linspace', N=1000,
                           x_range=[-2, 2],
                           theta_range=[-1, 1]):

    x = setup_x(N, x_mode=x_mode, x_range=x_range)

    theta = setup_theta(theta_range)

    epsilon = noise_level * np.random.standard_normal(size=(1, N))

    nonlinear_y = function(theta, x) + epsilon

    nonlinear_model_dataframe = pd.DataFrame.from_dict({'x': np.squeeze(x),
                                                        'y': np.squeeze(nonlinear_y)})

    return nonlinear_model_dataframe


def uniform_inputs(mn, mx, N):
    return np.random.uniform(mn, mx, size=(1, N))


def gauss_inputs(mn, sd, N):
    return mn + sd * np.random.standard_normal(size=(1, N))


def linspace_inputs(mn, mx, N):
    return np.linspace(mn, mx, N)


def make_nonlinear_transform(transform, theta_first=True):
    if theta_first:
        return lambda theta, x: transform(theta, x)
    else:
        return lambda theta, x: transform(x, theta)


def make_power_transform():
    return make_nonlinear_transform(np.power, theta_first=False)


def make_LN_transform(f):
    """linear-nonlinear transforms"""
    return lambda theta, x: f(theta*x)


def make_nonlinear_parameters(default, range_tuple):
    return Parameters([default], [range_tuple], ['theta'])


def make_rectlin_transform():
    return lambda theta, x: np.where(x > theta, x - theta, 0)


def setup_trig(trig_function, theta_range=[-5, 5]):
    input_values = np.linspace(-10, 10, 200)

    parameters = make_nonlinear_parameters(1, theta_range)

    transform = make_LN_transform(trig_function)

    return input_values, parameters, transform


def setup_power(max_degree):
    input_values = np.linspace(0, 10, 200)

    parameters = make_nonlinear_parameters(1, [0, max_degree])

    transform = make_power_transform()

    return input_values, parameters, transform


def setup_LN(f, input_range, theta_range=[-1, 1]):
    input_values = np.linspace(*input_range, num=200)

    parameters = make_nonlinear_parameters(0, theta_range)

    transform = make_LN_transform(f)

    return input_values, parameters, transform


def setup_rectlin(theta_range=[-10, 10]):

    transform = make_rectlin_transform()

    input_values = np.linspace(-10, 10, 200)

    parameters = make_nonlinear_parameters(0, theta_range)

    return input_values, parameters, transform


def setup_theta(theta_range):
    theta_width = theta_range[1] - theta_range[0]
    theta = np.random.rand()*theta_width+theta_range[0]
    return theta
