import numpy as np
def design_notch_peak_filter(w0, Q, ftype):
    """
    Design notch or peak digital filter.
    Parameters
    ----------
    w0 : float
        Normalized frequency to remove from a signal. It is a
        scalar that must satisfy  ``0 < w0 < 1``, with ``w0 = 1``
        corresponding to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    ftype : str
        The type of IIR filter to design:
            - notch filter : ``notch``
            - peak filter  : ``peak``
    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.
    """

    # Guarantee that the inputs are floats
    w0 = float(w0)
    Q = float(Q)

    # Checks if w0 is within the range
    if w0 > 1.0 or w0 < 0.0:
        raise ValueError("w0 should be such that 0 < w0 < 1")

    # Get bandwidth
    bw = w0/Q

    # Normalize inputs
    bw = bw*np.pi
    w0 = w0*np.pi

    # Compute -3dB atenuation
    gb = 1/np.sqrt(2)

    if ftype == "notch":
        # Compute beta: formula 11.3.4 (p.575) from reference [1]
        beta = (np.sqrt(1.0-gb**2.0)/gb)*np.tan(bw/2.0)
    elif ftype == "peak":
        # Compute beta: formula 11.3.19 (p.579) from reference [1]
        beta = (gb/np.sqrt(1.0-gb**2.0))*np.tan(bw/2.0)
    else:
        raise ValueError("Unknown ftype.")

    # Compute gain: formula 11.3.6 (p.575) from reference [1]
    gain = 1.0/(1.0+beta)

    # Compute numerator b and denominator a
    # formulas 11.3.7 (p.575) and 11.3.21 (p.579)
    # from reference [1]
    if ftype == "notch":
        b = gain*np.array([1.0, -2.0*np.cos(w0), 1.0])
    else:
        b = (1.0-gain)*np.array([1.0, 0.0, -1.0])
    a = np.array([1.0, -2.0*gain*np.cos(w0), (2.0*gain-1.0)])

    return b, a
