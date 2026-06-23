from prometheus_client import Counter, Gauge, Histogram
import threading as _threading
from collections import deque as _deque

INSIGHT_LATENCY = Histogram(
    "f1di_insight_latency_ms",
    "Insight latency in milliseconds",
    buckets=(10, 25, 50, 100, 250, 500, 1000),
)
INSIGHTS_TOTAL = Counter(
    "f1di_insights_total",
    "Generated insights",
    ["risk", "policy", "audience"],
)
CONFIDENCE_GAUGE = Gauge("f1di_last_confidence", "Latest calibrated insight confidence")
RAG_RESULTS = Histogram(
    "f1di_rag_results",
    "Number of retrieved evidence items",
    buckets=(0, 1, 2, 3, 5, 8, 13),
)
HTTP_REQUESTS_TOTAL = Counter(
    "f1di_http_requests_total",
    "HTTP requests",
    ["method", "path", "status"],
)
HTTP_REQUEST_LATENCY = Histogram(
    "f1di_http_request_latency_ms",
    "HTTP request latency in milliseconds",
    ["method", "path"],
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500),
)
READY_CHECK_TOTAL = Counter("f1di_ready_check_total", "Readiness checks", ["status"])

FEATURE_DRIFT_ZSCORE = Gauge(
    "f1di_feature_drift_zscore",
    "Rolling Z-score of each telemetry input feature vs. recent baseline",
    ["feature"],
)
DRIFT_ALERT_ACTIVE = Gauge(
    "f1di_drift_alert_active",
    "1 if any feature Z-score exceeds the alert threshold, 0 otherwise",
)
CALIBRATION_ECE_GAUGE = Gauge(
    "f1di_calibration_ece",
    "Current calibration ECE from the live isotonic calibrator",
)
CALIBRATION_REGRESSION_BLOCKED = Gauge(
    "f1di_calibration_regression_blocked",
    "1 if the most recent retrain was blocked due to ECE regression, 0 otherwise",
)
SHADOW_V2_SCORE_DELTA = Histogram(
    "f1di_shadow_v2_score_delta",
    "v2 challenger minus v1 production confidence score per insight",
    buckets=(-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3),
)
CONCEPT_DRIFT_SUSPECTED = Gauge(
    "f1di_concept_drift_suspected",
    "1 if drift has persisted through multiple retrains, indicating a baseline shift",
)

_LATENCY_LOCK = _threading.Lock()
_LATENCY_WINDOW: _deque = _deque(maxlen=200)


def record_insight_latency(ms: float) -> None:
    """Record one insight latency sample into the rolling window."""
    with _LATENCY_LOCK:
        _LATENCY_WINDOW.append(ms)


def latency_percentiles() -> dict:
    """Return p50/p95/p99 from the rolling window of recent insight latencies."""
    with _LATENCY_LOCK:
        vals = sorted(_LATENCY_WINDOW)
    if not vals:
        return {"p50": None, "p95": None, "p99": None, "n": 0}
    n = len(vals)

    def _p(pct: float) -> float:
        return round(vals[min(int(pct / 100 * n), n - 1)], 1)

    return {"p50": _p(50), "p95": _p(95), "p99": _p(99), "n": n}
