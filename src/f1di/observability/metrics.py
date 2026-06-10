from prometheus_client import Counter, Gauge, Histogram

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
