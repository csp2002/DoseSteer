"""GDP-HMM evaluation helpers (Metric 1 of the challenge).

Metric 1:

    union = ((ref > 5) | (pred > 5)) & (Body > 0)
    ref5  =  (ref > 5)               & (Body > 0)
    error = sum(|ref - pred|[union]) / sum(ref5)        # per plan
    metric = unweighted mean over plans

Doses are in Gy; `ref` is the reference rescaled so that PTV-high D97 ==
prescription.
"""

import numpy as np


def gdp_metric1(pred_gy, ref_gy, body, iso_gy=5.0):
    body = body > 0
    union = ((ref_gy > iso_gy) | (pred_gy > iso_gy)) & body
    ref5 = (ref_gy > iso_gy) & body
    if ref5.sum() == 0:
        return float('nan')
    return float(np.abs(ref_gy - pred_gy)[union].sum() / ref5.sum())


def gdp_mae(pred_gy, ref_gy, body, iso_gy=5.0):
    """Plain masked MAE on the union 5 Gy region."""
    body = body > 0
    union = ((ref_gy > iso_gy) | (pred_gy > iso_gy)) & body
    if not union.any():
        return float('nan')
    return float(np.abs(pred_gy - ref_gy)[union].mean())


def body_mae(pred_gy, ref_gy, body):
    return float(np.abs(pred_gy - ref_gy)[body > 0].mean())


def ptv_stats(dose_gy, ptv_mask):
    roi = dose_gy[ptv_mask > 0]
    if roi.size == 0:
        return {}
    d95, d05 = np.percentile(roi, 5), np.percentile(roi, 95)
    return {'D95': float(d95), 'D05': float(d05), 'Dmax': float(roi.max()),
            'D05_div_D95': float(d05 / (d95 + 1e-5)), 'max_div_D95': float(roi.max() / (d95 + 1e-5))}


def summarize(records):
    """Cohort means (unweighted over plans), overall and per site."""
    out = {'n': len(records)}
    for key in ('metric1', 'mae5', 'mae_body'):
        vals = [r[key] for r in records if key in r and r[key] == r[key]]
        if vals:
            out[key] = float(np.mean(vals))
        for s, name in ((1, 'HaN'), (2, 'Lung')):
            v = [r[key] for r in records if r['site'] == s and key in r and r[key] == r[key]]
            if v:
                out['%s_%s' % (key, name)] = float(np.mean(v))
    return out
