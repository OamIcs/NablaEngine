"""Summary statistics of per-state errors shared by the accuracy scorers (numpy only).

  summary(e)            {n, median, p90, p99, max, share_gt_1e-3} of the finite values (numpy.quantile, linear),
                        nonfinite = count of the others; {n: 0, nonfinite} when nothing is finite
  summary(e, cos)       also cos_min and cos_p1 of the cosines at the same positions
"""

import numpy as np

THRESHOLD = 1e-3


def summary(e, cos=None, threshold=THRESHOLD):
    e = np.asarray(e, np.float64).ravel()
    fin = np.isfinite(e)
    out = dict(n=int(fin.sum()), nonfinite=int((~fin).sum()))
    if not fin.any():
        return out
    x = e[fin]
    out.update(median=float(np.median(x)), p90=float(np.quantile(x, 0.9)), p99=float(np.quantile(x, 0.99)),
               max=float(x.max()), **{share_key(threshold): float(np.mean(x > threshold))})
    if cos is not None:
        c = np.asarray(cos, np.float64).ravel()[fin]
        c = c[np.isfinite(c)]
        if c.size:
            out.update(cos_min=float(c.min()), cos_p1=float(np.quantile(c, 0.01)))
    return out


def share_key(threshold=THRESHOLD):
    """The key of the share above a threshold: share_gt_1e-3 for 1e-3."""
    m, e = f"{threshold:.0e}".split("e")
    return f"share_gt_{int(m)}e{int(e)}" if int(m) != 1 else f"share_gt_1e{int(e)}"


def median_ratio(a, b):
    """median(a) / median(b) over the finite values of each (None if either is empty or median(b) is 0)."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not a.size or not b.size or np.median(b) == 0:
        return None
    return float(np.median(a) / np.median(b))
