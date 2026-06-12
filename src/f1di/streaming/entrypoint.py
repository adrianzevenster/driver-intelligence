"""Kafka worker entry point.

Run with:
    python -m f1di.streaming.entrypoint

Reads telemetry windows from the configured Kafka topic, runs inference,
and produces DriverInsight results to the insight topic.

Requires the [streaming] extra: pip install 'f1-driver-intelligence[streaming]'
"""
from __future__ import annotations

import logging
import signal
import sys
import time

logger = logging.getLogger("f1di.streaming.worker")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": "%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


def run() -> None:
    _setup_logging()

    try:
        from confluent_kafka import Consumer, Producer, KafkaError
    except ImportError:
        logger.error(
            "confluent-kafka not installed. "
            "Install with: pip install 'f1-driver-intelligence[streaming]'"
        )
        sys.exit(1)

    from f1di.config.settings import settings
    from f1di.inference.fusion import InferenceOrchestrator
    from f1di.streaming.contracts import decode_window
    from f1di.streaming.kafka_worker import process_payload

    logger.info(
        "kafka_worker_starting  bootstrap=%s  in=%s  out=%s",
        settings.kafka_bootstrap_servers,
        settings.telemetry_topic,
        settings.insight_topic,
    )

    orchestrator = InferenceOrchestrator()

    consumer = Consumer({
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "group.id": "f1di-insight-worker",
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
        "session.timeout.ms": 30000,
    })
    producer = Producer({
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "linger.ms": 5,
    })

    consumer.subscribe([settings.telemetry_topic])

    # Graceful shutdown on SIGTERM / SIGINT
    _running = True

    def _stop(signum, frame):
        nonlocal _running
        logger.info("kafka_worker_shutdown_requested signal=%s", signum)
        _running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    processed = 0
    errors = 0
    last_log = time.monotonic()

    logger.info("kafka_worker_ready  waiting for messages")

    while _running:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            continue

        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                logger.warning("kafka_consumer_error: %s", msg.error())
                errors += 1
            continue

        try:
            result = process_payload(msg.value(), orchestrator)
            window = decode_window(msg.value())
            producer.produce(
                settings.insight_topic,
                key=window.driver_id.encode("utf-8"),
                value=result,
            )
            producer.poll(0)
            processed += 1
        except Exception as exc:
            logger.warning("kafka_worker_process_error: %s", exc)
            errors += 1

        # Periodic throughput log
        now = time.monotonic()
        if now - last_log >= 60:
            logger.info("kafka_worker_stats  processed=%d errors=%d", processed, errors)
            last_log = now

    consumer.close()
    producer.flush(timeout=10)
    logger.info("kafka_worker_stopped  processed=%d errors=%d", processed, errors)


if __name__ == "__main__":
    run()
