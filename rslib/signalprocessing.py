"""
signalprocessing

a set of functions dedicated to the processing and analysis of 1D signals

Functions
---------
winter_derivative1
    obtain the first derivative of a 1D signal according to Winter 2009 method.

winter_derivative2
    obtain the second derivative of a 1D signal according to Winter 2009 method.

feedman_diaconis_bins
    digitize a 1D signal in bins defined according to the freedman-diaconis rule

fir_filt
    apply a FIR (Finite Impulse Response) filter to a 1D signal

mean_filt
    apply a moving average filter to a 1D signal

median_filt
    apply a median filter to a 1D signal

butterworth_filt
    apply a butterworth filter to a 1D signal

cubicspline_interp
    apply cubic spline interpolation to a 1D signal

residual_analysis
    get the optimal cut-off frequency for a filter on 1D signals according
    to Winter 2009 'residual analysis' method

crossovers
    get the x-axis coordinates of the junction between the lines best fitting
    a 1D signal in a least-squares sense.

psd
    obtain the power spectral density estimate of a 1D signal using the
    periodogram method.

crossings
    obtain the location of the samples being across a target value.

xcorr
    get the cross/auto-correlation and lag of of multiple/one 1D signal.
"""


#! IMPORTS

from types import FunctionType, MethodType
from typing import Any, Literal
import itertools as it
import numpy as np
import scipy.interpolate as si
import scipy.signal as ss


__all__ = [
    "winter_derivative1",
    "winter_derivative2",
    "freedman_diaconis_bins",
    "fir_filt",
    "mean_filt",
    "median_filt",
    "butterworth_filt",
    "cubicspline_interp",
    "residual_analysis",
    "crossovers",
    "psd",
    "crossings",
    "xcorr",
]

#! FUNCTIONS


def winter_derivative1(
    y_signal: np.ndarray[Any, np.dtype[np.float_]],
    x_signal: None | np.ndarray[Any, np.dtype[np.float_]] = None,
    time_diff: float | int = 1,
):
    """
    return the first derivative of y.

    Parameters
    ----------

    y_signal: np.ndarray[Any, np.dtype[np.float_]]
        the signal to be derivated

    x_signal: None | np.ndarray[Any, np.dtype[np.float_]]
        the optional signal from which y has to  be derivated (default = None)

    time_diff: float | int
        the difference between samples in y.
        NOTE: if x is provided, this parameter is ignored

    Returns
    -------

    z: ndarray
        an array being the first derivative of y

    References
    ----------

    Winter DA. Biomechanics and Motor Control of Human Movement. Fourth Ed.
        Hoboken, New Jersey: John Wiley & Sons Inc; 2009.
    """

    # get x
    if x_signal is None:
        x_sig = np.arange(len(y_signal)) * time_diff
    else:
        x_sig = x_signal

    # get the derivative
    return (y_signal[2:] - y_signal[:-2]) / (x_sig[2:] - x_sig[:-2])


def winter_derivative2(
    y_signal: np.ndarray[Any, np.dtype[np.float_]],
    x_signal: None | np.ndarray[Any, np.dtype[np.float_]] = None,
    time_diff: float | int = 1,
):
    """
    return the second derivative of y.

    Parameters
    ----------

    y_signal: np.ndarray[Any, np.dtype[np.float_]]
        the signal to be derivated

    x_signal: None | np.ndarray[Any, np.dtype[np.float_]]
        the optional signal from which y has to  be derivated (default = None)

    time_diff: float | int
        the difference between samples in y.
        NOTE: if x is provided, this parameter is ignored

    Returns
    -------

    z: np.ndarray[Any, np.dtype[np.float_]]
        an array being the second derivative of y

    References
    ----------

    Winter DA. Biomechanics and Motor Control of Human Movement. Fourth Ed.
        Hoboken, New Jersey: John Wiley & Sons Inc; 2009.
    """

    # get x
    if x_signal is None:
        x_sig = np.arange(len(y_signal)) * time_diff
    else:
        x_sig = np.copy(x_signal)

    # get the derivative
    diffy = (y_signal[2:] - y_signal[1:-1]) / (x_sig[2:] - x_sig[1:-1])
    diffy -= (y_signal[1:-1] - y_signal[:-2]) / (x_sig[1:-1] - x_sig[:-2])
    diffx = (x_sig[2:] - x_sig[:-2]) * 0.5
    return diffy / diffx


def freedman_diaconis_bins(
    y_signal: np.ndarray[Any, np.dtype[np.float_]],
):
    """
    return a digitized version of y where each value is linked to a
    bin (i.e an int value) according to the rule.

                             IQR(x)
            width = 2 * ---------------
                        len(x) ** (1/3)

    Parameters
    ----------

    y_signal: np.ndarray[Any, np.dtype[np.float_]]
        the signal to be digitized.

    Returns
    -------

    d: np.ndarray[Any, np.dtype[np.float_]]
        an array with the same shape of y containing the index
        of the bin of which the corresponding sample of y is part.

    References
    ----------
    Freedman D, Diaconis P.
        (1981) On the histogram as a density estimator:L 2 theory.
        Z. Wahrscheinlichkeitstheorie verw Gebiete 57: 453-476.
        doi: 10.1007/BF01025868
    """

    # y IQR
    qnt1 = np.quantile(y_signal, 0.25)
    qnt3 = np.quantile(y_signal, 0.75)
    iqr = qnt3 - qnt1

    # get the width
    wdt = 2 * iqr / (len(y_signal) ** (1 / 3))

    # get the number of intervals
    samp = int(np.floor(1 / wdt)) + 1

    # digitize z
    digitized = np.zeros(y_signal.shape)
    for i in np.arange(samp) + 1:
        loc = np.argwhere((y_signal >= (i - 1) * wdt) & (y_signal < i * wdt))
        digitized[loc] = i - 1
    return digitized


def mean_filt(
    signal: np.ndarray[Any, np.dtype[np.float_]],
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
):
    """
    apply a moving average filter to the signal.

    Parameters
    ----------

    signal: np.ndarray[Any, np.dtype[np.float_]],
        the signal to be filtered.

    order: int = 1,
        the number of samples to be considered as averaging window.

    pd: str = "edge"
        the type of padding style adopted to apply before implementing
        the filter. Available options are:

        constant (default)
        Pads with a constant value.

        edge
        Pads with the edge values of array.

        linear_ramp
        Pads with the linear ramp between end_value and the array
        edge value.

        maximum
        Pads with the maximum value of all or part of the vector
        along each axis.

        mean
        Pads with the mean value of all or part of the vector
        along each axis.

        median
        Pads with the median value of all or part of the vector
        along each axis.

        minimum
        Pads with the minimum value of all or part of the vector
        along each axis.

        reflect
        Pads with the reflection of the vector mirrored on the first
        and last values of the vector along each axis.

        symmetric
        Pads with the reflection of the vector mirrored along the edge
        of the array.

        wrap
        Pads with the wrap of the vector along the axis. The first values
        are used to pad the end and the end values are used to pad
        the beginning.

    offset: float
        a value within the [0, 1] range defining how the averaging window is
        obtained.
        Offset=0,
            indicate that for each sample, the filtered value will be the mean
            of the subsequent n-1 values plus the current sample.
        Offset=1,
            on the other hand, calculates the filtered value at each sample as
            the mean of the n-1 preceding values plus the current sample.
        Offset=0.5,
            centers the averaging window around the actual sample being
            evaluated.

    Returns
    -------
    z: 1D array
        The filtered signal.
    """

    # get the window range
    win = np.unique((np.arange(order) - offset * (order - 1)).astype(int))

    # get the indices of the samples
    idx = [win + order - 1 + j for j in np.arange(len(signal))]

    # padding
    filtered = np.pad(signal, order - 1, mode=pad_style)  # type: ignore

    # get the mean of each window
    return np.array([np.mean(filtered[j]) for j in idx]).flatten().astype(float)


def median_filt(
    signal: np.ndarray[Any, np.dtype[np.float_]],
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
):
    """
    apply a moving average filter to the signal.

    Parameters
    ----------

    signal: np.ndarray[Any, np.dtype[np.float_]],
        the signal to be filtered.

    order: int = 1,
        the number of samples to be considered as averaging window.

    pd: str = "edge"
        the type of padding style adopted to apply before implementing
        the filter. Available options are:

        constant (default)
        Pads with a constant value.

        edge
        Pads with the edge values of array.

        linear_ramp
        Pads with the linear ramp between end_value and the array
        edge value.

        maximum
        Pads with the maximum value of all or part of the vector
        along each axis.

        mean
        Pads with the mean value of all or part of the vector
        along each axis.

        median
        Pads with the median value of all or part of the vector
        along each axis.

        minimum
        Pads with the minimum value of all or part of the vector
        along each axis.

        reflect
        Pads with the reflection of the vector mirrored on the first
        and last values of the vector along each axis.

        symmetric
        Pads with the reflection of the vector mirrored along the edge
        of the array.

        wrap
        Pads with the wrap of the vector along the axis. The first values
        are used to pad the end and the end values are used to pad
        the beginning.

    offset: float
        a value within the [0, 1] range defining how the averaging window is
        obtained.
        Offset=0,
            indicate that for each sample, the filtered value will be the mean
            of the subsequent n-1 values plus the current sample.
        Offset=1,
            on the other hand, calculates the filtered value at each sample as
            the mean of the n-1 preceding values plus the current sample.
        Offset=0.5,
            centers the averaging window around the actual sample being
            evaluated.

    Returns
    -------
    z: 1D array
        The filtered signal.
    """

    # get the window range
    win = np.unique((np.arange(order) - offset * (order - 1)).astype(int))

    # get the indices of the samples
    idx = [win + order - 1 + j for j in np.arange(len(signal))]

    # padding
    filtered = np.pad(signal, order - 1, mode=pad_style)  # type: ignore

    # get the mean of each window
    out = [np.median(filtered[j]) for j in idx]
    out = np.array(out).flatten().astype(float)


def fir_filt(
    signal: np.ndarray[Any, np.dtype[np.float_]],
    fcut: float | int | list[float | int] | tuple[float | int] = 1,
    fsamp: float | int = 2,
    order: int = 5,
    ftype: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
    wtype: Literal[
        "boxcar",
        "triang",
        "blackman",
        "hamming",
        "hann",
        "bartlett",
        "flattop",
        "parzen",
        "bohman",
        "blackmanharris",
        "nuttall",
        "barthann",
        "cosine",
        "exponential",
        "tukey",
        "taylor",
    ] = "hamming",
    pstyle: Literal[
        "constant",
        "edge",
        "linear_ramp",
        "maximum",
        "mean",
        "median",
        "minimum",
        "reflect",
        "symmetric",
        "wrap",
    ] = "edge",
):
    """
    apply a FIR filter with the specified specs to the signal.

    Parameters
    ----------

    signal: np.ndarray[Any, np.dtype[np.float_]],
        the signal to be filtered.

    fcut: float | int | list[float | int], tuple[float | int] = 1,
        the cutoff frequency of the filter.

    fsamp: float | int = 2,
        The sampling frequency of the signal.

    order: int = 5,
        the order of the filter

    ftype: str = "lowpass",
        the type of filter. Any of "bandpass", "lowpass", "highpass",
        "bandstop".

    wn: str
        the type of window to be applied. Any of:
            "boxcar",
            "triang",
            "blackman",
            "hamming",
            "hann",
            "bartlett",
            "flattop",
            "parzen",
            "bohman",
            "blackmanharris",
            "nuttall",
            "barthann",
            "cosine",
            "exponential",
            "tukey",
            "taylor"

    pd: str
        the type of padding style adopted to apply before implementing
        the filter. Available options are:

        constant (default)
        Pads with a constant value.

        edge
        Pads with the edge values of array.

        linear_ramp
        Pads with the linear ramp between end_value and the array
        edge value.

        maximum
        Pads with the maximum value of all or part of the vector
        along each axis.

        mean
        Pads with the mean value of all or part of the vector
        along each axis.

        median
        Pads with the median value of all or part of the vector
        along each axis.

        minimum
        Pads with the minimum value of all or part of the vector
        along each axis.

        reflect
        Pads with the reflection of the vector mirrored on the first
        and last values of the vector along each axis.

        symmetric
        Pads with the reflection of the vector mirrored along the edge
        of the array.

        wrap
        Pads with the wrap of the vector along the axis. The first values
        are used to pad the end and the end values are used to pad
        the beginning.

    Returns
    -------

    filtered: 1D array
        the filtered signal.
    """
    coefs = ss.firwin(
        order,
        fcut,
        window=wtype,
        pass_zero=ftype,  # type: ignore
        fs=fsamp,
    )
    val = signal[0] if pstyle == "constant" else 0
    padded = np.pad(
        signal,
        pad_width=(2 * order - 1, 0),
        mode=pstyle,
        constant_values=val,
    )
    avg = np.mean(padded)
    out = ss.lfilter(coefs, 1.0, padded - avg)[(2 * order - 1) :]
    return np.array(out).flatten().astype(float) + avg


def butterworth_filt(
    signal: np.ndarray[Any, np.dtype[np.float_]],
    fcut: float | int | list[float | int] | tuple[float | int] = 1,
    fsamp: float | int = 2,
    order: int = 5,
    ftype: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
    phase_corrected: bool = True,
):
    """
    Provides a convenient function to call a Butterworth filter with the
    specified parameters.

    Parameters
    ----------

    signal: np.ndarray[Any, np.dtype[np.float_]],
        the signal to be filtered.

    fcut: float | int | list[float | int], tuple[float | int] = 1,
        the cutoff frequency of the filter.

    fsamp: float | int = 2,
        The sampling frequency of the signal.

    order: int = 5,
        the order of the filter

    ftype: str = "lowpass",
        the type of filter. Any of "bandpass", "lowpass", "highpass",
        "bandstop".

    phase_corrected: bool, optional
        should the filter be applied twice in opposite directions
        to correct for phase lag?

    Returns
    -------

    z: np.ndarray[Any, np.dtype[np.float_]],
        the resulting 1D filtered signal.
    """

    # get the filter coefficients
    sos = ss.butter(
        order,
        (np.array([fcut]).flatten() / (0.5 * fsamp)),
        ftype,
        analog=False,
        output="sos",
    )

    # get the filtered data
    if phase_corrected:
        return ss.sosfiltfilt(sos, signal)
    else:
        return ss.sosfilt(sos, signal)


def cubicspline_interp(
    y_old: np.ndarray[Any, np.dtype[np.float_]],
    nsamp: int | None = None,
    x_old: np.ndarray[Any, np.dtype[np.float_]] | None = None,
    x_new: np.ndarray[Any, np.dtype[np.float_]] | None = None,
):
    """
    Get the cubic spline interpolation of y.

    Parameters
    ----------

    y_old: np.ndarray[Any, np.dtype[np.float_]],
        the data to be interpolated.

    nsamp: int | None = None,
        the number of points for the interpolation.

    x_old: np.ndarray[Any, np.dtype[np.float_]] | None = None,
        the x coordinates corresponding to y. It is ignored if n is provided.

    x_new: np.ndarray[Any, np.dtype[np.float_]] | None = None,
        the newly (interpolated) x coordinates corresponding to y.
        It is ignored if n is provided.

    Returns
    -------
    z: np.ndarray[Any, np.dtype[np.float_]]
        the interpolated y axis
    """

    # control of the inputs
    if nsamp is not None:
        if x_old is None or x_new is None:
            raise ValueError("the pair x_old / x_new or nsamp must be defined")
        x_old = np.arange(len(y_old))  # type: ignore
        x_new = np.linspace(np.min(x_old), np.max(x_old), nsamp)  # type: ignore

    # get the cubic-spline interpolated y
    cspline = si.CubicSpline(x_old, y_old)
    return cspline(x_new).flatten().astype(float)


def residual_analysis(
    signal: np.ndarray[Any, np.dtype[np.float_]],
    ffun: FunctionType | MethodType,
    fnum: int = 1000,
    fmax: float | int | None = None,
    nseg: int = 2,
    minsamp: int = 2,
):
    """
    Perform Winter's residual analysis of the input signal.

    Parameters
    ----------

    signal: np.ndarray[Any, np.dtype[np.float_]],
        the signal to be investigated

    ffun: FunctionType | MethodType,
        the filter to be used for the analysis. The function must receive two
        inputs: the raw signal and the filter cut-off. The output must be the
        filtered signal.

    fnum: int = 1000,
        the number of frequencies to be tested within the (0, f_max) range to
        create the residuals curve of the Winter's residuals analysis approach.

    fmax: float | int | None = None,
        the maximum frequency to be tested in normalized units in the (0, 0.5)
        range. If None, it is defined as the frequency covering the 99% of
        the cumulative signal power.

    nseg: int = 2,
        the number of segments that can be used to fit the residuals curve in
        order to identify the best deflection point.
        NOTE: values above 3 will greatly increase the computation time.

    minsamp: int = 2,
        the minimum number of elements that have to be considered for each
        segment during the calculation of the best deflection point.

    Returns
    -------

    cutoff: float
        the suggested cutoff value

    frequencies: np.ndarray[Any, np.dtype[np.float_]],
        the tested frequencies

    residuals: np.ndarray[Any, np.dtype[np.float_]],
        the residuals corresponding to the given frequency

    Notes
    -----

    The signal is filtered over a range of frequencies and the sum of squared
    residuals (SSE) against the original signal is computer for each tested
    cut-off frequency. Next, a series of fitting lines are used to estimate the
    optimal disruption point defining the cut-off frequency optimally
    discriminating between noise and good quality signal.

    References
    ----------

    Winter DA 2009, Biomechanics and Motor Control of Human Movement.
        Fourth Ed. John Wiley & Sons Inc, Hoboken, New Jersey (US).

    Lerman PM 1980, Fitting Segmented Regression Models by Grid Search.
        Appl Stat. 29(1):77.
    """

    # data check
    if fmax is None:
        pwr, frq = psd(signal, 1)
        idx = int(np.where(np.cumsum(pwr) / np.sum(pwr) >= 0.99)[0][0])  # type: ignore
        fmax = float(frq[idx])
    assert 0 < fmax < 0.5, "fmax must lie in the (0, 0.5) range."
    assert minsamp >= 2, "'min_samples' must be >= 2."

    # get the optimal crossing over point
    frq = np.linspace(0, fmax, fnum + 1)[1:].astype(float)
    res = np.array([np.sum((signal - ffun(signal, i)) ** 2) for i in frq])
    iopt = crossovers(res, nseg, minsamp)[0][-1]
    fopt = float(frq[iopt])

    # return the parameters
    return fopt, frq, res.astype(float)


def _sse(
    xval: np.ndarray[Any, np.dtype[np.float_]],
    yval: np.ndarray[Any, np.dtype[np.float_]],
    segm: list[tuple[int]],
):
    """
    method used to calculate the residuals

    Parameters
    ----------

    xval: np.ndarray[Any, np.dtype[np.float_]],
        the x axis signal

    yval: np.ndarray[Any, np.dtype[np.float_]],
        the y axis signal

    segm: list[tuple[int]],
        the extremes among which the segments have to be fitted

    Returns
    -------

    sse: float
        the sum of squared errors corresponding to the error obtained
        fitting the y-x relationship according to the segments provided
        by s.
    """
    sse = 0.0
    for i in np.arange(len(segm) - 1):
        coords = np.arange(segm[i], segm[i + 1] + 1)
        coefs = np.polyfit(xval[i], yval[i], 1)
        vals = np.polyval(coefs, xval[coords])
        sse += np.sum((yval[coords] - vals) ** 2)
    return float(sse)


def crossovers(
    signal: np.ndarray[Any, np.dtype[np.float_]],
    segments: int = 2,
    min_samples: int = 5,
):
    """
    Detect the position of the crossing over points between K regression
    lines used to best fit the data.

    Parameters
    ----------

    signal:np.ndarray[Any, np.dtype[np.float_]],
        the signal to be fitted.

    segments:int=2,
        the number of segments that can be used to fit the residuals curve in
        order to identify the best deflection point.
        NOTE: values above 3 will greatly increase the computation time.

    min_samples:int=5,
        the minimum number of elements that have to be considered for each
        segment during the calculation of the best deflection point.

    Returns
    -------

    crossings: list[int]
        An ordered array of indices containing the samples corresponding to the
        detected crossing over points.

    coefs: list[tuple[float]]
        A list of tuples containing the slope and intercept of the line
        describing each fitting segment.

    Notes
    -----

    the steps involved in the calculations can be summarized as follows:

        1)  Get all the segments combinations made possible by the given
            number of crossover points.
        2)  For each combination, calculate the regression lines corresponding
            to each segment.
        3)  For each segment calculate the residuals between the calculated
            regression line and the effective data.
        5)  Once the sum of the residuals have been calculated for each
            combination, sort them by residuals amplitude.

    References
    ----------

    Lerman PM 1980, Fitting Segmented Regression Models by Grid Search.
    Appl Stat. 29(1):77.
    """

    # control the inputs
    assert min_samples >= 2, "'min_samples' must be >= 2."

    # get the X axis
    xaxis = np.arange(len(signal)).astype(float)

    # get all the possible combinations of segments
    combs = []
    for i in np.arange(1, segments):
        start = min_samples * i
        stop = len(signal) - min_samples * (segments - i)
        combs += [np.arange(start, stop)]
    combs = list(it.product(*combs))

    # remove those combinations having segments shorter than "samples"
    combs = [i for i in combs if np.all(np.diff(i) >= min_samples)]

    # generate the crossovers matrix
    combs = (
        np.zeros((len(combs), 1)),
        np.atleast_2d(combs),
        np.ones((len(combs), 1)) * len(signal) - 1,
    )
    combs = np.hstack(combs).astype(int)

    # calculate the residuals for each combination
    sse = np.array([_sse(xaxis, signal, i) for i in combs])

    # sort the residuals
    sortedsse = np.argsort(sse)

    # get the optimal crossovers order
    crs = xaxis[combs[sortedsse[0]]]

    # get the fitting slopes
    slopes = [np.arange(i0, i1) for i0, i1 in zip(crs[:-1], crs[1:])]
    slopes = [np.polyfit(i, signal[i], 1).astype(float) for i in slopes]

    # return the crossovers
    return crs[1:-1].astype(int).tolist(), slopes


def psd(
    signal: np.ndarray[Any, np.dtype[np.float_]],
    fsamp: float | int = 1.0,
):
    """
    compute the power spectrum of signal using fft

    Parameters
    ----------

    signal: np.ndarray[Any, np.dtype[np.float_]],
        A 1D numpy array

    fssamp: float | int = 1.0,
        the sampling frequency (in Hz) of the signal. If not provided the
        power spectrum frequencies are provided as normalized values within the
        (0, 0.5) range.

    Returns
    -------
    frq: np.ndarray[Any, np.dtype[np.float_]],
        the frequency corresponding to each element of pwr.

    pwr: np.ndarray[Any, np.dtype[np.float_]],
        the power of each frequency
    """

    # get the psd
    fft = np.fft.rfft(signal - np.mean(signal)) / len(signal)
    amp = abs(fft)
    pwr = np.concatenate([[amp[0]], 2 * amp[1:-1], [amp[-1]]]).flatten() ** 2
    frq = np.linspace(0, fsamp / 2, len(pwr))

    # return the data
    return frq.astype(float), pwr.astype(float)


def crossings(
    signal: np.ndarray[Any, np.dtype[np.float_]],
    value: int | float = 0.0,
):
    """
    Dectect the crossing points in x compared to value.

    Parameters
    ----------

        signal: np.ndarray[Any, np.dtype[np.float_]],
            the 1D signal from which the crossings have to be found.

        value: int | float = 0.0,

    Returns
    -------

        crs: 1D array
            the samples corresponding to the crossings.

        sgn: 1D array
            the sign of the crossings. Positive sign means crossings
            where the signal moves from values lower than "value" to
            values higher than "value". Negative sign indicate the
            opposite trend.
    """

    # get the sign of the signal without the offset
    sgn = np.sign(signal - value)

    # get the location of the crossings
    crs = np.where(abs(sgn[1:] - sgn[:-1]) == 2)[0].astype(int)

    # return the crossings
    return crs, -sgn[crs]


def xcorr(
    sig1: np.ndarray[Any, np.dtype[np.float_]],
    sig2: np.ndarray[Any, np.dtype[np.float_]] | None = None,
    biased: bool = False,
    full: bool = False,
):
    """
    set the (multiple) auto/cross correlation of the data in y.

    Parameters
    ----------
    sig1: np.ndarray[Any, np.dtype[np.float_]],
        the signal from which the auto or cross-correlation is provided.

    sig2: np.ndarray[Any, np.dtype[np.float_]] | None = None,
        the signal from which the auto or cross-correlation is provided.
        if None. The autocorrelation of x is provided. Otherwise the x-y
        cross-correlation is returned.

    biased:bool=False,
        if True, the biased auto/cross-correlation is provided.
        Otherwise, the 'unbiased' estimator is returned.

    full:bool=False,
        Should the negative lags be reported?

    Returns
    -------
    xcr: np.ndarray[Any, np.dtype[np.float_]]
        the auto/cross-correlation value.

    lag: np.ndarray[Any, np.dtype[np.float_]]
        the lags in sample units.
    """

    # take the autocorrelation if only y is provided
    if sig2 is None:
        sigx = np.atleast_2d(sig1)
        sigz = np.vstack([sigx, sigx])

    # take the cross-correlation (ensure the shortest signal is zero-padded)
    else:
        sigx = np.zeros((1, max(len(sig1), len(sig2))))
        sigy = np.copy(sigx)
        sigx[:, : len(sig1)] = sig1
        sigy[:, : len(sig2)] = sig2
        sigz = np.vstack([sigx, sigy])

    # get the matrix shape
    rows, cols = sigz.shape

    # remove the mean from each dimension
    sigv = sigz - np.atleast_2d(np.mean(sigz, 1)).T

    # take the cross-correlation
    xcr = []
    for i in np.arange(rows - 1):
        for j in np.arange(i + 1, rows):
            res = np.atleast_2d(ss.fftconvolve(sigv[i], sigv[j][::-1], "full"))
            xcr += [res]

    # average over all the multiples
    xcr = np.mean(np.concatenate(xcr, axis=0), axis=0)

    # adjust the output
    lags = np.arange(-(cols - 1), cols)
    if not full:
        xcr = xcr[(cols - 1) :]
        lags = lags[(cols - 1) :]

    # normalize
    xcr /= (cols + 1 - abs(lags)) if not biased else (cols + 1)

    # return the cross-correlation data
    return xcr.astype(float), lags.astype(int)
