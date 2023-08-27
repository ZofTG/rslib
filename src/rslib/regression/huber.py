'''
class PolynomialHuberRegressor(HuberRegressor):
    """
    Polynomial extension to the Robust Huber regression.

    Parameters
    ----------
    epsilon: float, default=1.35
        The parameter epsilon controls the number of samples that should be
        classified as outliers. The smaller the epsilon, the more robust it is
        to outliers. Epsilon must be in the range [1, inf).

    max_iter: int, default=100
        Maximum number of iterations that
        scipy.optimize.minimize(method="L-BFGS-B") should run for.

    alpha: float, default=0.0001
        Strength of the squared L2 regularization. Note that the penalty is
        equal to alpha * ||w||^2. Must be in the range [0, inf).

    warm_start: bool, default=False
        This is useful if the stored attributes of a previously used model
        has to be reused. If set to False, then the coefficients will be
        rewritten for every call to fit. See the Glossary.

    fit_intercept: bool, default=True
        Whether or not to fit the intercept. This can be set to False if the
        data is already centered around the origin.

    tol: float, default=1e-05
        The iteration will stop when max{|proj g_i | i = 1, ..., n} <= tol
        where pg_i is the i-th component of the projected gradient.

    degree: int, default=1
        the degree of the polynome.
    """

    _degree: int
    _has_intercept: bool

    def __init__(
        self,
        epsilon: float = 1.35,
        max_iter: int = 100,
        alpha: float = 0.0001,
        warm_start: bool = False,
        fit_intercept: bool = True,
        tol: float = 1e-05,
        degree: int = 1,
    ):
        super().__init__(
            epsilon=epsilon,
            max_iter=max_iter,
            alpha=alpha,
            warm_start=warm_start,
            fit_intercept=fit_intercept,
            tol=tol,
        )
        self._degree = degree
        self._has_intercept = fit_intercept

    @property
    def degree(self):
        """return the polynomial degree"""
        return self._degree

    @property
    def has_intercept(self):
        """return true if the intercept is included"""
        return self._has_intercept

    def _adjust_x(
        self,
        xarr: np.ndarray[Any, np.dtype[np.float_]],
    ):
        if xarr.ndim == 1:
            xarr2 = np.atleast_2d(xarr).T
        else:
            xarr2 = xarr
        return PolynomialFeatures(
            degree=self._degree,
            interaction_only=False,
            include_bias=False,  # type: ignore
        ).fit_transform(xarr2)

    def fit(
        self,
        X: np.ndarray[Any, np.dtype[np.float_]],
        y: np.ndarray[Any, np.dtype[np.float_]],
        sample_weight: np.ndarray[Any, np.dtype[np.float_]] | None = None,
    ):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y: array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight: array-like, shape (n_samples,)
            Weight given to each sample.

        Returns
        -------
        self: object
            Fitted HuberRegressor estimator.
        """
        return super().fit(self._adjust_x(X), y, sample_weight)

    def predict(
        self,
        X: np.ndarray[Any, np.dtype[np.float_]],
    ):
        """
        Predict using the linear model.

        Parameters
        ----------
        X:  array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C:  array, shape (n_samples,)
            Returns predicted values.
        """
        return super().predict(self._adjust_x(X))


class PowerHuberRegressor(PolynomialHuberRegressor):
    """
    Power regression extension to the Robust Huber regression.

    Parameters
    ----------
    epsilon: float, default=1.35
        The parameter epsilon controls the number of samples that should be
        classified as outliers. The smaller the epsilon, the more robust it is
        to outliers. Epsilon must be in the range [1, inf).

    max_iter: int, default=100
        Maximum number of iterations that
        scipy.optimize.minimize(method="L-BFGS-B") should run for.

    alpha: float, default=0.0001
        Strength of the squared L2 regularization. Note that the penalty is
        equal to alpha * ||w||^2. Must be in the range [0, inf).

    warm_start: bool, default=False
        This is useful if the stored attributes of a previously used model
        has to be reused. If set to False, then the coefficients will be
        rewritten for every call to fit. See the Glossary.

    fit_intercept: bool, default=True
        Whether or not to fit the intercept. This can be set to False if the
        data is already centered around the origin.

    tol: float, default=1e-05
        The iteration will stop when max{|proj g_i | i = 1, ..., n} <= tol
        where pg_i is the i-th component of the projected gradient.
    """

    def __init__(
        self,
        epsilon: float = 1.35,
        max_iter: int = 100,
        alpha: float = 0.0001,
        warm_start: bool = False,
        fit_intercept: bool = True,
        tol: float = 1e-05,
    ):
        super().__init__(
            epsilon=epsilon,
            max_iter=max_iter,
            alpha=alpha,
            warm_start=warm_start,
            fit_intercept=fit_intercept,
            tol=tol,
            degree=1,
        )

    def fit(
        self,
        X: np.ndarray[Any, np.dtype[np.float_]],
        y: np.ndarray[Any, np.dtype[np.float_]],
        sample_weight: np.ndarray[Any, np.dtype[np.float_]] | None = None,
    ):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y: array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight: array-like, shape (n_samples,)
            Weight given to each sample.

        Returns
        -------
        self: object
            Fitted HuberRegressor estimator.
        """
        out = super().fit(np.log(X), np.log(y), sample_weight)
        if not self.has_intercept:
            out.intercept_ = 1.0
        else:
            out.intercept_ = np.e**out.intercept_
        return out

    def predict(
        self,
        X: np.ndarray[Any, np.dtype[np.float_]],
    ):
        """
        Predict using the linear model.

        Parameters
        ----------
        X:  array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C:  array, shape (n_samples,)
            Returns predicted values.
        """
        return self.intercept_ * np.sum(X**self.coef_, axis=1)
'''
