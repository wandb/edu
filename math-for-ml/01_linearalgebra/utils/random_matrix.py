import autograd.numpy as np
import scipy.integrate

PRECISION = 1e-4


class RandomMatrix(object):

    def __init__(self):
        self.symmetric = False
        return

    def eigvals(self):
        if self.symmetric:
            return np.linalg.eigvalsh(self.M)
        else:
            return np.linalg.eigvals(self.M)

    def expected_cumulative_spectral_distribution(self, lam, precision=PRECISION, accumulate=False):
        if lam < self.min_lam:
            return 0.

        lams = self.generate_lams(lam, precision)

        singular_mass = self.expected_spectral_singular_mass()
        density_values = [self.expected_spectral_density(lam) for lam in lams]
        if not accumulate:
            accumulated_density = scipy.integrate.trapz(density_values, lams)
            return ((lam >= 0) * singular_mass) + accumulated_density
        else:
            accumulated_densities = scipy.integrate.cumtrapz(density_values, lams)
            accumulated_masses = singular_mass * (lams[1:] >= 0) + accumulated_densities
            return accumulated_masses

    def __repr__(self):
        return self.M.__repr__()

    def display_expected_cumulative_spectral_distribution(
            self, ax, precision=PRECISION, **plot_kwargs):
        lams = self.generate_lams(self.max_lam + precision, precision)

        expected_csds = self.expected_cumulative_spectral_distribution(
            self.max_lam + precision, precision, accumulate=True)

        ax.plot(lams[1:], expected_csds, **plot_kwargs)

        return ax

    def generate_lams(self, lam, precision=PRECISION):
        return np.arange(self.min_lam - 2 * precision, lam + precision, precision)


class SymmetricWigner(RandomMatrix):

    def __init__(self, N):
        super().__init__()
        self.symmetric = True
        self.generate = self.generate_symmetric_gaussian
        self.M = self.generate(N)
        self.min_lam = -2.
        self.max_lam = 2.

    @staticmethod
    def generate_symmetric_gaussian(N):
        """generate an N by N symmetric gaussian random matrix with variance 1/N
        """
        base_matrix = SymmetricWigner.generate_gaussian(N)
        return (1 / np.sqrt(2)) * (base_matrix + base_matrix.T)

    @staticmethod
    def generate_gaussian(N):
        """generate an N by N gaussian random matrix with variance 1/N
        """
        return 1 / np.sqrt(N) * np.random.standard_normal(size=(N, N))

    def expected_spectral_singular_mass(self):
        return 0.

    def expected_spectral_density(self, lam):
        """Expected density for a symmetric gaussian random matrix with variance 1/N"""
        if lam > self.max_lam or lam < self.min_lam:
            return 0
        else:
            return 1 / (2 * np.pi) * np.sqrt(2 ** 2 - lam ** 2)


class Wishart(RandomMatrix):

    def __init__(self, N, k, negative=False):
        super().__init__()
        self.symmetric = True
        if negative:
            self.generate = self.generate_negative_wishart
            self.sign = -1
        else:
            self.generate = self.generate_wishart
            self.sign = 1

        self.N, self.k = N, k
        self.M = self.generate(self.N, self.k)
        self.sigma = 1.

        self.central_lam = self.sign * N / k
        self.scaling_factor = 1 / (2 * np.pi * self.sigma ** 2)

        self.lam_plus = self.sigma ** 2 * self.sign * (1 + np.sqrt(self.central_lam)) ** 2
        self.lam_minus = self.sigma ** 2 * self.sign * (1 - np.sqrt(self.central_lam)) ** 2

        if negative:
            self.max_lam = 0.
            self.min_lam = self.lam_plus
        else:
            self.max_lam = self.lam_plus
            self.min_lam = 0.

        self.expected_spectral_density = self.marchenkopastur_density

    @staticmethod
    def generate_wishart(N, k=1):
        """generate an N by N wishart random matrix with rank min(N,k)
        """
        self_outer_product = lambda x: x.dot(x.T)
        random_factor = np.random.standard_normal(size=(N, k))
        wishart_random_matrix = 1 / k * self_outer_product(random_factor)

        return wishart_random_matrix

    @staticmethod
    def generate_negative_wishart(N, k=1):
        """generate an N by N negative wishart random matric with rank min(N,k)
        """
        wishart_random_matrix = Wishart.generate_wishart(N, k)
        negative_wishart_random_matrix = -1 * wishart_random_matrix

        return negative_wishart_random_matrix

    def marchenkopastur_density(self, lam):
        """the density for the non-singular portion of the marchenko-pastur distribution,
        as given by https://en.wikipedia.org/wiki/Marchenko-Pastur_distribution.
        """

        # density is 0 on real half-line opposite its sign
        if np.sign(lam) != self.sign:
            return 0

        # that handled, we can solve as though lam were positive, since density invariant
        lam = np.abs(lam)
        lam_minus = self.sign * self.lam_minus
        lam_plus = self.sign * self.lam_plus

        if (lam > lam_minus and lam < lam_plus):
            unscaled_density = np.sqrt(
                (lam_plus - lam) * (lam - lam_minus)) / (self.central_lam * lam)
            return self.scaling_factor * unscaled_density
        else:
            return 0

    def expected_spectral_singular_mass(self):
        return max(1 - self.k / self.N, 0)


def generate_random_unit_vector(dim=25):
    gauss_random_vector = np.atleast_2d(np.random.standard_normal(size=dim)).T
    return gauss_random_vector / np.linalg.norm(gauss_random_vector)
