# coding: utf-8

from __future__ import division, print_function, absolute_import
from builtins import int
from future import standard_library
standard_library.install_aliases()

import numpy as _np
import numexpr
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from awkde.tools import standardize_nd_sample, shift_and_scale_nd_sample
from awkde.backend import kernel_sum


class GaussianKDE(BaseEstimator):
    """
    GaussianKDE

    Kernel denstiy estimate using gaussian kernels and a local kernel bandwidth.
    Implements the 'sklearn.BaseEstimator` class and can be used in a cross-
    validation gridsearch (`sklearn.model_selection`).

    Parameters
    ----------
    glob_bw : float or str
        The global bandwidth of the kernel, must be a float > 0 or one of
        ["silverman"|"scott"]. If alpha is not None, this is the bandwidth for
        the first estimate KDE from which the local bandwidth is calculated.
        If ["silverman"|"scott"] a rule of thumb is used to estimate the
        bandwidth. (default: "silverman")
    alpha : float or None
        If None, only the global bandwidth is used. If 0 <= alpha <= 1, an
        adaptive local kernel bandwith is used as described in [1]_.
        (default: 0.5)
    diag_cov : bool
        If True, only scale by variance, diagonal cov matrix. (default: False)
    max_gb : float
        Maximum gigabyte of RAM occupied in evaluating the KDE.

    Notes
    -----
    The unweighted kernel density estimator is defined as

    .. math:

      \hat{f}(x) = \sum_i \frac{1}{h\lambda_i}\cdot
                     K\left(\frac{x - X_i}{h\lambda_i}\right)


    where the product :math:`\lambda_i` takes the role of a local $\sigma_i$.

    The kernel bandwith is choosen locally to account for variations in the
    density of the data.
    Areas with large density gets smaller kernels and vice versa.
    This smoothes the tails and gets high resolution in high statistics regions.
    The local bandwidth paramter is defined as

    .. math: \lambda_i = (\hat{f}(X_i) / g)^{-\alpha}

    where :math:`\log g = n^{-1}\sum_i \log\hat{f}(X_i)` is some normalization
    and :math:`\hat{f}(X_i)` the KDE estimate at the data point :math:`X_i`.
    The local bandwidth is multiplied to the global abndwidth for each kernel.

    Furthermore different scales in data is accounted for by scaling it via its
    covariance matrix to an equal spread.
    First a global kernel bandwidth is applied to the transformed data and then
    based on that density a local bandwidth parameter is applied.

    All credit for the method goes to [1]_ and to S. Schoenen and L. Raedel for
    huge parts of the implementation :+1:.
    For information on Silverman or Scott rule, see [2]_ or [3]_.

    References
    ----------
    .. [1] B. Wang and X. Wang, "Bandwidth Selection for Weighted Kernel Density
           Estimation", Sep. 2007, DOI: 10.1214/154957804100000000.
    .. [2] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [3] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    """
    def __init__(self, glob_bw="silverman", alpha=0.5,
                 diag_cov=False, max_gb=2.):
        self._std_X = None  # Default indicating that no fit was done yet

        if type(glob_bw) is str:
            if glob_bw not in ["silverman", "scott"]:
                raise ValueError("glob_bw can be one of ['silverman'|'scott'].")
            pass
        elif glob_bw <= 0:
            raise ValueError("Global bandwidth must be > 0.")

        if max_gb <= 0:
            raise ValueError("max_gb must be > 0")

        self.alpha = alpha
        self.max_gb = max_gb
        self.glob_bw = glob_bw
        self.diag_cov = diag_cov

        # These can be set WITH CAUTION, to speed up the `fit` process, which
        # skips the reevaluation of the given dataset for the adaptive kernel.
        # But make sure, the exact same parameters and data is used.
        self._kde_values = None

        return

    # Properties
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        # Global, internal switch to indicate we use adaptive kernels
        if alpha is None:
            self._adaptive = False
        else:
            if alpha < 0 or alpha > 1:
                raise ValueError("alpha must be in [0, 1]")
            self._adaptive = True
        self._alpha = alpha

    # Public Methods
    def __call__(self, X):
        # Copy docstring, DNRY
        self.__call__.__func__.__doc__ = self.predict.__doc__
        return self.predict(X)

    def fit(self, X, bounds=None, weights=None):
        """
        Prepare KDE to describe the data.

        Data is transformed via global covariance matrix to equalize scales in
        different features.
        Then a symmetric kernel with cov = diag(1) is used to describe the pdf
        at each point.

        Parameters
        -----------
        X : array-like, shape (n_samples, n_features)
            Data points defining each kernel position. Each row is a point, each
            column is a feature.
        bounds : array-like, shape (n_features, 2)
            Boundary condition for each dimension. The method of mirrored points
            is used to improve prediction close to bounds. If no bound shall be
            given use None: [[0, None], ...]. (default: None)
        weights : Per event weights to consider. If `None` all weights are set
            to one. (default: None)

        Returns
        -------
        mean : array-like, shape (n_features)
            The (weighted) mean of the given data.
        cov : array-like, shape (n_features, n_features)
            The (weighted) covariance matrix of the given data.
        """
        if bounds is not None:
            # TODO: Use mirroring of points near boundary regions and then
            #       constrain KDE to values inside Region but taking all kernels
            #       into account. (only neccessary on hard cuts)
            raise NotImplementedError("TODO: Boundary conditions.")
        if len(X.shape) != 2:
            raise ValueError("X must have shape (n_samples, n_features).")

        # Max numbers in RAM = max_gb / 8B/double * 2^30B/GB
        self.max_len = int(self.max_gb / 8. * 2**30)
        if len(X) > self.max_len:
            raise ValueError("Data to big for given maximum memory size.")

        # Transform sample to zero mean and unity covariance matrix
        self.n_kernels, self.n_features = X.shape
        self._std_X, self.mean, self.cov = standardize_nd_sample(
            X, cholesky=True, ret_stats=True, diag=self.diag_cov)

        # Get global bandwidth number
        self.glob_bw = self._get_glob_bw(self.glob_bw)

        # Build local bandwidth parameter if alpha is set
        if self._adaptive:
            if self._kde_values is None:
                self._kde_values = self._evaluate(self._std_X, adaptive=False)

            # Get local bandwidth from local "density" g
            g = (_np.exp(_np.sum(_np.log(self._kde_values)) / self.n_kernels))
            # Needed inverted so use power of (+alpha), shape (n_samples)
            self.inv_loc_bw = (self._kde_values / g)**(self.alpha)

        return self.mean, self.cov

    def predict(self, X):
        """
        Evaluate KDE at given points X.

        Parameters
        -----------
        X : array-like, shape (n_samples, n_features)
            Data points we want to evaluate the KDE at. Each row is a point,
            each column is a feature.

        Returns
        -------
        prob : array-like, shape (len(X))
            The probability from the KDE pdf for each point in X.
        """
        if self._std_X is None:
            raise ValueError("KDE has not been fitted to data yet.")

        X = _np.atleast_2d(X)
        _, n_feat = X.shape
        if n_feat != self.n_features:
            raise ValueError("Dimensions of given points and KDE don't match.")

        # Standardize given points to be in the same space as the KDE
        X = standardize_nd_sample(X, mean=self.mean, cov=self.cov,
                                  cholesky=True, ret_stats=False,
                                  diag=self.diag_cov)

        # No need to backtransform, because we only return y-values
        return self._evaluate(X, adaptive=self._adaptive)

    def sample(self, n_samples, random_state=None):
        """
        Get random samples from the KDE model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. (default: 1)
        random_state : RandomState, optional
            Turn seed into a `np.random.RandomState` instance. Method from
            `sklearn.utils`. Can be None, int or RndState. (default: None)

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            Generated samples from the fitted model.
        """
        if self._std_X is None:
            raise ValueError("KDE has not been fitted to data yet.")

        rndgen = check_random_state(random_state)

        # Select randomly all kernels to sample from
        idx = rndgen.randint(0, self.n_kernels, size=n_samples)

        # Because we scaled to standard normal dist, we can draw uncorrelated
        # and the cov is only the inverse bandwidth of each kernel.
        means = self._std_X[idx]
        invbw = _np.ones(n_samples) / self.glob_bw
        if self._adaptive:
            invbw *= self.inv_loc_bw[idx]
        invbw = invbw.reshape(n_samples, 1)

        # Retransform to original space
        sample = _np.atleast_2d(rndgen.normal(means, 1. / invbw))
        return shift_and_scale_nd_sample(sample, self.mean, self.cov)

    def score(self, X):
        """
        Compute the total ln-probability of points X under the KDE model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data points included in the score calculation. Each row is a point,
            each column is a feature.

        Returns
        -------
        lnprob : float
            Total ln-likelihood of the data X given the KDE model.
        """
        probs = self.predict(X)
        if _np.any(probs <= 0):
            return -_np.inf
        else:
            return _np.sum(_np.log(probs))

    # Private Methods
    def _evaluate(self, X, adaptive):
        """
        Evaluate KDE at given points, returning the log-probability.

        Parameters
        -----------
        X : array-like, shape (n_samples, n_features)
            Data points we want to evaluate the KDE at. Each row is a point,
            each column is a feature.
        adaptive : bool, optional
            Wether to evaluate with fixed or with adaptive kernel.
            (default: True)

        Returns
        -------
        prob : array-like, shape (len(X))
            The probability from the KDE PDF for each point in X.
        """
        n = self.n_kernels
        d = self.n_features

        # Get fixed or adaptive bandwidth
        invbw = _np.ones(n) / self.glob_bw
        if adaptive:
            invbw *= self.inv_loc_bw

        # Total norm, including gaussian kernel norm with data covariance
        norm = invbw**d / _np.sqrt(_np.linalg.det(2 * _np.pi * self.cov)) / n
        self.norm = norm

        return kernel_sum(self._std_X, X, invbw, norm)

    def _get_glob_bw(self, glob_bw):
        """Simple wrapper to handle string args given for global bw."""
        dim = self.n_features
        nsam = self.n_kernels
        if glob_bw == "silverman":
            return _np.power(nsam * (dim + 2.0) / 4.0, -1. / (dim + 4))
        elif glob_bw == "scott":
            return _np.power(nsam, -1. / (dim + 4))
        else:
            return self.glob_bw
