import numpy as np
import matplotlib
import matplotlib.animation


def setup_and_run_animation(pmf, iters):
    """
        Produces a javascript-based animated figure
        for the addition of iters iid random variables
        with a given probability mass function (pmf).
    """
    assert min(pmf) >= 0, "no negative numbers in pmf"
    assert np.isclose(sum(pmf), 1), "doesn't sum to 1"
    assert max(pmf) < 1, "must have non-zero variance"

    figure, bar_plot, pmf = setup_run(pmf, iters)

    animate = make_animate(pmf, bar_plot)

    animation = matplotlib.animation.FuncAnimation(
            figure, animate, frames=iters)

    anim_as_html = animation.to_jshtml()

    return anim_as_html


def setup_run(pmf, iters):
    x_max = iters * (len(pmf))
    x_locations = list(range(x_max+2))
    x_labels = [str(loc) if (loc % (len(pmf) - 1)) == 0 else ''
                for loc in x_locations]
    extended_PMF = np.hstack([pmf, [0] * (x_max + 2 - len(pmf))])
    edge = 2
    fig = matplotlib.figure.Figure(figsize=(12, 6))
    pmf_ax = fig.add_subplot(111)
    pmf_bars = pmf_ax.bar(x_locations, extended_PMF,
                          width=1, align='center', alpha=0.8,
                          linewidth=0,)

    setup_plot(pmf_ax, x_locations, edge, x_labels)

    fig.suptitle("Adding Up "+str(iters)+" Random Numbers",
                 weight='bold', y=1.)

    return fig, pmf_bars, pmf


def make_animate(orig_pmf, bar_plot):
    """
        Recursive convolution gives the pmf for adding random
        variables from the same distribution,
        so the resulting pmfs are the distributions of
        sums of independent and identically distributed random variables
    """

    def animate(frame_idx):
        pmf = orig_pmf
        for ii in range(frame_idx):
            pmf = np.convolve(pmf, orig_pmf)
        [bar_plot[idx].set_height(h)
         for idx, h in enumerate(pmf)]

    return animate


def setup_plot(ax, locs, edge, labels):
    ax.set_ylim([0, 1])
    ax.set_xlim([locs[0] - edge, locs[1] + edge])
    ax.xaxis.set_ticks(locs)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticks([0, 0.5, 1])
    ax.tick_params(axis='x', top=False)
    ax.tick_params(axis='y', right=False)
    ax.set_ylabel('Probability', fontsize='xx-large', fontweight='bold')
