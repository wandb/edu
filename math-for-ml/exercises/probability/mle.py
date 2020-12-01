import ipywidgets
import matplotlib.pyplot as plt
import numpy as np


def make_gauss_fitter(data, true_mu, true_sigma):
    """given a dataset with a given mean mu and variance Sigma,
    construct a function that plots a gaussian pdf with variable
    mean and variance over a normed histogram of the data.
    intended for use with ipywidgets.interact.
    """

    N = len(data)
    xs = np.arange(-5 * np.sqrt(true_sigma)+true_mu,
                   5 * np.sqrt(true_sigma)+true_mu,
                   0.01)

    fig, ax = plt.subplots()
    _ = ax.hist(data, density=True, bins=max(10, N//50))
    fit, = plt.plot(xs, gauss_pdf(xs, 0, 1), lw=4)

    def fitter(mu=0., sigma=1.):
        fit.set_data(xs, gauss_pdf(xs, mu, sigma))
        surprise = compute_surprise(mu, data, sigma)
        print(f"Surprise: {surprise}")

    return fitter


def make_interactor(fitter, mu_lims=[-10, 10], sigma_lims=[1e-3, 10]):
    mu_slider = ipywidgets.FloatSlider(
        -2, min=mu_lims[0], max=mu_lims[1], step=1e-2,
        description=r"$\mu$")
    sigma_slider = ipywidgets.FloatSlider(
        1., min=sigma_lims[0], max=sigma_lims[1], step=1e-3,
        description=r"$\sigma$")
    interactor = ipywidgets.interact(fitter, mu=mu_slider, sigma=sigma_slider)
    return interactor


def make_plot(data, num_gaussians, true_mu=0.):
    fig, surprise_ax, data_ax = setup_figure(num_gaussians)

    mu_mle = np.mean(data)  # MLE is the mean of the data
    sigma = np.std(data)

    mus = true_mu + np.linspace(-2 * sigma, 2 * sigma, 200)

    mle_surprise = compute_surprise(mu_mle, data, sigma)
    surprises = np.array([compute_surprise(mu, data, sigma)
                          for mu in mus])

    plot_surprises(mus, surprises, surprise_ax)
    plot_mle_surprise(mu_mle, mle_surprise, surprise_ax)
    plot_true_mean(true_mu, surprise_ax)

    plot_data_histogram(data, data_ax)
    plot_evenlyspaced_gaussians(num_gaussians, data, data_ax, mus, sigma)

    plt.tight_layout()
    return fig


def setup_figure(num_gaussians):
    fig, axs = plt.subplots(2, figsize=(8, 4), sharex=True)
    surprise_ax, data_ax = axs

    surprise_ax.set_xlabel(r"$\mu$", fontsize=12)
    surprise_ax.set_ylabel("Model Surprise", fontsize=12)
    surprise_ax.set_title(
        r"Model Surprise as a Function of $\mu$",
        fontsize=14)

    data_ax.set_title(
        f"{num_gaussians} Example Gaussians and Data Distribution",
        fontsize=14)
    data_ax.set_ylabel(r"$p$", fontsize=12)

    return fig, surprise_ax, data_ax


def plot_surprises(mus, surprises, ax):
    ax.plot(mus, surprises, "r")


def plot_mle_surprise(mu_mle, mle_surprise, ax):
    ax.plot(mu_mle, mle_surprise, "*", color="k",
            markersize=10)


def plot_true_mean(true_mu, ax):
    ymin, ymax = ax.get_ylim()
    ax.vlines(true_mu, ymin, ymax, linewidth=2, linestyle='dashed')
    ax.set_ylim(ymin, ymax)


def plot_data_histogram(data, ax):
    ax.hist(data, density=True, bins=20,
            histtype='stepfilled', alpha=0.2, color='gray')


def plot_evenlyspaced_gaussians(num_gaussians, data, data_ax, mus, sigma):
    xmin, xmax = data_ax.get_xlim()

    xs = np.linspace(np.min(data) - 10, np.max(data) + 10, 1000)
    step = len(mus) / num_gaussians
    idxs = np.arange(0, len(mus), step)
    for idx in idxs:
        mu = mus[int(np.floor(idx))]
        gaussian = gauss_pdf(xs, mu, sigma)
        data_ax.plot(xs, gaussian, linewidth=2)

    data_ax.set_xlim(xmin, xmax)


def gauss_pdf(xs, mu, sigma):
    normalizing_factor = (2 * np.pi * sigma ** 2) ** (1/2)
    distance_to_mean = (xs - mu) ** 2
    scaled_distance_to_mean = distance_to_mean / (2 * sigma ** 2)
    unnormalized_density = np.exp(-scaled_distance_to_mean)

    normalized_density = 1 / normalizing_factor * unnormalized_density

    return normalized_density


def compute_surprise(mus, data, sigma):
    likelihoods = gauss_pdf(data, mus, sigma)
    loglikelihoods = np.log(likelihoods)
    surprises = -loglikelihoods

    return np.mean(surprises)
