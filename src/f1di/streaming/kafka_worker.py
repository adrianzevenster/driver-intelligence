from __future__ import annotations

from f1di.config.settings import settings
from f1di.inference.fusion import InferenceOrchestrator
from f1di.streaming.contracts import decode_window, encode_insight


def process_payload(payload: bytes, orchestrator: InferenceOrchestrator) -> bytes:
    window = decode_window(payload)
    insight = orchestrator.analyze(window)
    return encode_insight(insight)


class KafkaInsightWorker:
    """Kafka worker shell.

    Install `.[streaming]` to enable this worker. It is isolated from the core inference path so local
    development and regression tests do not require Kafka.
    """

    def __init__(self) -> None:
        from confluent_kafka import Consumer, Producer

        self.consumer = Consumer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": "f1di-insight-worker",
            "auto.offset.reset": "latest",
        })
        self.producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
        self.orchestrator = InferenceOrchestrator()

    def run_forever(self) -> None:
        self.consumer.subscribe([settings.telemetry_topic])
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            result = process_payload(msg.value(), self.orchestrator)
            window = decode_window(msg.value())
            self.producer.produce(settings.insight_topic, key=window.driver_id, value=result)
            self.producer.poll(0)
