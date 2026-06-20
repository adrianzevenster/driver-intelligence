"""SHAP-based explanation for meta-learner predictions.

Returns the top-5 feature contributions for the current inference so the UI
can show *why* a risk level was assigned rather than just *what* it is.
Gracefully no-ops if shap is not installed or the meta-learner is not yet active.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("f1di.inference.explainer")

_META_PATH = Path("data/calibration/meta_learner.pkl")


def explain_findings(findings: list, iso_confidence: float) -> list[dict]:
    """Return top feature contributions for the current inference.

    Uses TreeExplainer for HGBC and LinearExplainer for LogisticRegression.
    Returns list of {feature, value, contribution} dicts sorted by |contribution| desc.
    Falls back to [] if shap is not installed or meta-learner has < 20 real labels.
    """
    try:
        import shap
    except ImportError:
        return []

    if not _META_PATH.exists():
        return []

    try:
        from f1di.inference.meta_learner import MetaLearner, FEATURE_NAMES, findings_to_array
        meta = MetaLearner.load(_META_PATH)
        if meta.n_real < 20:
            return []

        x = findings_to_array(findings, iso_confidence).reshape(1, -1)
        x_s = meta._scaler.transform(x)

        vals = None
        try:
            explainer = shap.TreeExplainer(meta._model)
            sv = explainer.shap_values(x_s)
            if isinstance(sv, list):
                vals = sv[1][0] if len(sv) > 1 else sv[0][0]
            elif sv.ndim == 3:
                vals = sv[0, :, 1]
            else:
                vals = sv[0]
        except Exception:
            try:
                import numpy as np
                background = x_s
                explainer = shap.LinearExplainer(meta._model, background)
                sv = explainer.shap_values(x_s)
                vals = sv[0] if isinstance(sv, list) else sv[0]
            except Exception:
                pass

        if vals is None:
            return []

        import numpy as np
        raw_x = x[0]
        contribs = []
        for name, sv_val, fv in zip(FEATURE_NAMES, vals, raw_x):
            contribs.append({
                "feature": name,
                "value": round(float(fv), 4),
                "contribution": round(float(sv_val), 4),
            })
        contribs.sort(key=lambda c: abs(c["contribution"]), reverse=True)
        return contribs[:5]

    except Exception as exc:
        logger.debug("explain_findings failed: %s", exc)
        return []
