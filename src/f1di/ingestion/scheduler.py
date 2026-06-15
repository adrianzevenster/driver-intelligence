from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date
from pathlib import Path

logger = logging.getLogger("f1di.ingestion.scheduler")

_DEFAULT_INTERVAL_HOURS = 6
_CURRENT_YEAR = date.today().year
_OUTCOME_LABELED_PATH = Path("data/calibration/outcome_labeled.json")
_OUTCOME_ROUNDS_PER_CYCLE = 4   # process at most this many new rounds per cycle
_FASTF1_REQUEST_DELAY_S = 3.0   # seconds between FastF1 API calls to avoid rate limits


class IngestionScheduler:
    def __init__(
        self,
        orchestrator,
        interval_hours: float = _DEFAULT_INTERVAL_HOURS,
        years: list[int] | None = None,
        n_per_year: int = 8,
    ) -> None:
        self.orchestrator = orchestrator
        self.interval_seconds = interval_hours * 3600
        self.years = years or [_CURRENT_YEAR, _CURRENT_YEAR - 1]
        self.n_per_year = n_per_year
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="ingestion-scheduler")
        logger.info("Ingestion scheduler started (interval=%.1fh)", self.interval_seconds / 3600)

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        await asyncio.sleep(30)
        cycle = 0
        while not self._stop.is_set():
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._run_pull)
            except Exception as exc:
                logger.error("Ingestion pull failed: %s", exc, exc_info=True)

            # Outcome labeling every other cycle (~12h at default interval).
            new_outcome_labels = 0
            if cycle % 2 == 0:
                try:
                    new_outcome_labels = await asyncio.get_event_loop().run_in_executor(
                        None, self._run_outcome_labeling
                    )
                except Exception as exc:
                    logger.warning("Outcome labeling failed: %s", exc)

            # Retrain calibrator + adjust agent thresholds when new outcome labels arrive.
            if cycle % 4 == 0 or new_outcome_labels > 0:
                try:
                    await asyncio.get_event_loop().run_in_executor(None, self._run_retrain)
                except Exception as exc:
                    logger.warning("Calibrator retrain failed: %s", exc)
                if new_outcome_labels > 0:
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._run_threshold_adjustment
                        )
                    except Exception as exc:
                        logger.warning("Threshold adjustment failed: %s", exc)
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._run_fit_tire_classifier
                        )
                    except Exception as exc:
                        logger.warning("Tire classifier fit failed: %s", exc)
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._run_fit_battery_classifier
                        )
                    except Exception as exc:
                        logger.warning("Battery classifier fit failed: %s", exc)
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._run_fit_weather_classifier
                        )
                    except Exception as exc:
                        logger.warning("Weather classifier fit failed: %s", exc)
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._run_fit_telemetry_classifier
                        )
                    except Exception as exc:
                        logger.warning("Telemetry classifier fit failed: %s", exc)
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._run_fit_meta_learner
                        )
                    except Exception as exc:
                        logger.warning("Meta-learner fit failed: %s", exc)

            cycle += 1
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                pass

    def _run_outcome_labeling(self) -> int:
        """Label stored insights against FastF1 race outcomes for past rounds.

        Tracks which (year, round_num) pairs have already been processed so
        FastF1 is not re-downloaded on every cycle.  Returns the total number
        of new FeedbackRecord rows written (correct + incorrect).
        """
        labeled: set[tuple[int, int]] = set()
        if _OUTCOME_LABELED_PATH.exists():
            try:
                labeled = {tuple(pair) for pair in json.loads(_OUTCOME_LABELED_PATH.read_text())}
            except Exception:
                pass

        new_labels_total = 0
        rounds_processed = 0
        for year in self.years:
            if rounds_processed >= _OUTCOME_ROUNDS_PER_CYCLE:
                break
            try:
                rounds = self._available_rounds(year)
            except Exception as exc:
                logger.warning("outcome_label: cannot fetch rounds for %d: %s", year, exc)
                continue

            # Process most-recent rounds first so new data is prioritised
            for round_num in reversed(rounds):
                if rounds_processed >= _OUTCOME_ROUNDS_PER_CYCLE:
                    break
                if (year, round_num) in labeled:
                    continue
                if rounds_processed > 0:
                    time.sleep(_FASTF1_REQUEST_DELAY_S)
                try:
                    from f1di.data.outcome_labeler import label_race
                    report = label_race(year, round_num)
                    n_new = report.n_labeled_correct + report.n_labeled_incorrect
                    logger.info(
                        "outcome_label year=%d round=%d correct=%d incorrect=%d no_match=%d",
                        year, round_num,
                        report.n_labeled_correct, report.n_labeled_incorrect, report.n_no_match,
                    )
                    # Mark as done only when FastF1 loaded successfully (incidents list
                    # is populated even for clean races; empty means the load failed).
                    if n_new > 0 or len(report.incidents_found) > 0:
                        labeled.add((year, round_num))
                        new_labels_total += n_new
                    rounds_processed += 1
                except Exception as exc:
                    logger.warning(
                        "outcome_label_failed year=%d round=%d: %s", year, round_num, exc
                    )
                    rounds_processed += 1

        if labeled:
            _OUTCOME_LABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
            _OUTCOME_LABELED_PATH.write_text(json.dumps(sorted(labeled)))

        return new_labels_total

    @staticmethod
    def _run_threshold_adjustment() -> None:
        try:
            from f1di.agents.threshold_fitter import adjust_from_labels
            result = adjust_from_labels()
            n = result.get("n_circuits", 0)
            if n > 0:
                logger.info("threshold_adjustment: %d circuit(s) updated from outcome labels", n)
        except Exception as exc:
            logger.warning("threshold_adjustment_failed: %s", exc)

    @staticmethod
    def _run_fit_battery_classifier() -> None:
        try:
            from f1di.agents.battery_classifier import train_from_labels
            r = train_from_labels()
            logger.info("battery_classifier_retrained: n_real=%d acc=%.3f", r.get("n_real", 0), r.get("accuracy", 0))
        except Exception as exc:
            logger.warning("battery_classifier_fit_failed: %s", exc)

    @staticmethod
    def _run_fit_weather_classifier() -> None:
        try:
            from f1di.agents.weather_classifier import train_from_labels
            r = train_from_labels()
            logger.info("weather_classifier_retrained: n_real=%d acc=%.3f", r.get("n_real", 0), r.get("accuracy", 0))
        except Exception as exc:
            logger.warning("weather_classifier_fit_failed: %s", exc)

    @staticmethod
    def _run_fit_meta_learner() -> None:
        try:
            from f1di.inference.meta_learner import train_from_labels
            r = train_from_labels()
            logger.info(
                "meta_learner_retrained: n_real=%d acc=%.3f active=%s",
                r.get("n_real", 0), r.get("accuracy", 0), r.get("active_in_inference"),
            )
        except Exception as exc:
            logger.warning("meta_learner_fit_failed: %s", exc)

    @staticmethod
    def _run_fit_telemetry_classifier() -> None:
        try:
            from f1di.agents.telemetry_classifier import train_from_labels
            r = train_from_labels()
            logger.info(
                "telemetry_classifier_retrained: n_real=%d acc=%.3f",
                r.get("n_real", 0), r.get("accuracy", 0),
            )
        except Exception as exc:
            logger.warning("telemetry_classifier_fit_failed: %s", exc)

    @staticmethod
    def _run_fit_tire_classifier() -> None:
        try:
            from f1di.agents.tire_classifier import train_from_labels
            report = train_from_labels()
            logger.info(
                "tire_classifier_retrained: n_real=%d acc=%.3f",
                report.get("n_real", 0), report.get("accuracy", 0),
            )
        except Exception as exc:
            logger.warning("tire_classifier_fit_failed: %s", exc)

    @staticmethod
    def _run_retrain() -> None:
        try:
            from f1di.confidence.online import retrain
            result = retrain()
            if not result.get("skipped"):
                logger.info("Scheduled retrain complete: ECE=%.4f", result.get("ece", 0))
        except ImportError:
            pass

    def _run_pull(self) -> None:
        from f1di.storage.database import db_session
        from f1di.storage.repository import already_ingested, mark_ingested

        for source, ingest_fn in self._ingesters():
            for year in self.years:
                try:
                    rounds = self._available_rounds(year)
                except Exception as exc:
                    logger.warning("Could not fetch round list for %d: %s", year, exc)
                    continue

                for round_num in rounds:
                    with db_session() as session:
                        if already_ingested(session, source=source, year=year, round_num=round_num):
                            continue
                    try:
                        docs = ingest_fn(self.orchestrator.retriever, years=[year], n_per_year=self.n_per_year)
                        with db_session() as session:
                            mark_ingested(session, source=source, year=year, round_num=round_num, documents_added=len(docs))
                        logger.info("Ingested %d docs [%s/%d/R%d]", len(docs), source, year, round_num)
                    except Exception as exc:
                        logger.warning("Ingestion failed [%s/%d/R%d]: %s", source, year, round_num, exc)
                    break  # ingest_fn pulls the full year in one call

    @staticmethod
    def _available_rounds(year: int) -> list[int]:
        try:
            import fastf1
            from datetime import date as _date
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            past = schedule[schedule["EventDate"].astype(str) <= str(_date.today())]
            return [int(r) for r in past["RoundNumber"].tolist()]
        except Exception:
            return list(range(1, 9))

    @staticmethod
    def _ingesters():
        try:
            from f1di.knowledge.fastf1_ingester import ingest as ff1_ingest
            yield "fastf1", ff1_ingest
        except ImportError:
            pass
        try:
            from f1di.knowledge.openf1_ingester import ingest as of1_ingest
            yield "openf1", of1_ingest
        except ImportError:
            pass
