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
            new_pull = False
            try:
                new_pull = await asyncio.get_event_loop().run_in_executor(None, self._run_pull)
            except Exception as exc:
                logger.error("Ingestion pull failed: %s", exc, exc_info=True)

            # Outcome labeling: every other cycle (scheduled) OR immediately after new data lands.
            new_outcome_labels = 0
            if cycle % 2 == 0 or new_pull:
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
                            None, self._run_fit_safety_car_classifier
                        )
                    except Exception as exc:
                        logger.warning("Safety car classifier fit failed: %s", exc)
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._run_fit_fuel_classifier
                        )
                    except Exception as exc:
                        logger.warning("Fuel classifier fit failed: %s", exc)
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._run_fit_meta_learner
                        )
                    except Exception as exc:
                        logger.warning("Meta-learner fit failed: %s", exc)
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._run_partial_fit_classifiers
                        )
                    except Exception as exc:
                        logger.warning("Partial fit classifiers failed: %s", exc)
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._check_precision_degradation
                        )
                    except Exception as exc:
                        logger.warning("Precision degradation check failed: %s", exc)
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._run_shadow_auto_promote
                        )
                    except Exception as exc:
                        logger.warning("Shadow auto-promote check failed: %s", exc)

            # Save retrieval eval snapshot after every knowledge pull cycle.
            if cycle % 2 == 0:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._run_retrieval_eval
                    )
                except Exception as exc:
                    logger.debug("Retrieval eval skipped: %s", exc)

            # Synthetic label audit every 4th cycle (~24h at default interval).
            if cycle % 4 == 0:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._run_synthetic_audit
                    )
                except Exception as exc:
                    logger.warning("Synthetic audit failed: %s", exc)

            # Data freshness check every 8th cycle (~48h at default interval).
            if cycle % 8 == 0:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._check_data_freshness
                    )
                except Exception as exc:
                    logger.warning("Data freshness check failed: %s", exc)
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._run_race_backtest
                    )
                except Exception as exc:
                    logger.warning("Race backtest failed: %s", exc)

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
                    try:
                        from f1di.data.outcome_labeler import label_quiet_stints
                        n_quiet = label_quiet_stints(year, round_num)
                        new_labels_total += n_quiet
                    except Exception as exc2:
                        logger.warning("null_outcome_label_failed year=%d round=%d: %s", year, round_num, exc2)
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
    def _run_fit_safety_car_classifier() -> None:
        try:
            from f1di.agents.safety_car_classifier import train_from_labels
            r = train_from_labels()
            logger.info(
                "safety_car_classifier_retrained: n_real=%d acc=%.3f",
                r.get("n_real", 0), r.get("accuracy", 0),
            )
        except Exception as exc:
            logger.warning("safety_car_classifier_fit_failed: %s", exc)

    @staticmethod
    def _run_fit_fuel_classifier() -> None:
        try:
            from f1di.agents.fuel_classifier import train_from_labels
            r = train_from_labels()
            logger.info(
                "fuel_classifier_retrained: n_real=%d acc=%.3f",
                r.get("n_real", 0), r.get("accuracy", 0),
            )
        except Exception as exc:
            logger.warning("fuel_classifier_fit_failed: %s", exc)

    def _run_retrieval_eval(self) -> None:
        try:
            from f1di.evaluation.retrieval_eval import evaluate_retriever, save_eval_report
            metrics = evaluate_retriever(self.orchestrator.retriever)
            save_eval_report(metrics)
            logger.info("retrieval_eval_saved: %s", metrics.summary())
        except Exception as exc:
            logger.debug("retrieval_eval_failed: %s", exc)

    @staticmethod
    def _run_partial_fit_classifiers() -> None:
        """Warm-start SGD classifiers with latest real labels — cheaper than full retrain."""
        for agent_mod in [
            "f1di.agents.tire_classifier",
            "f1di.agents.battery_classifier",
            "f1di.agents.weather_classifier",
            "f1di.agents.telemetry_classifier",
        ]:
            try:
                import importlib
                mod = importlib.import_module(agent_mod)
                r = mod.partial_fit_from_labels()
                if not r.get("skipped"):
                    logger.info(
                        "partial_fit %s: n_real=%d acc=%.3f",
                        agent_mod, r.get("n_real", 0), r.get("accuracy", 0),
                    )
            except Exception as exc:
                logger.warning("partial_fit_failed %s: %s", agent_mod, exc)

    @staticmethod
    def _check_data_freshness(stale_days: int = 14) -> None:
        """Alert via delivery channel if no new data has been ingested in stale_days days."""
        try:
            import datetime as _dt
            from f1di.storage.database import db_session
            from f1di.storage.models import IngestionRecord
            cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=stale_days)
            with db_session() as session:
                recent = (
                    session.query(IngestionRecord)
                    .filter(IngestionRecord.completed_at >= cutoff)
                    .first()
                )
            if recent is None:
                msg = (
                    f"F1DI — No new data ingested in the last {stale_days} days.\n"
                    "FastF1/OpenF1 ingestion may be failing silently. "
                    "Check logs for ingestion errors."
                )
                logger.warning("data_freshness: stale — no ingestion in %d days", stale_days)
                try:
                    from f1di.delivery.notifier import send_system_alert
                    send_system_alert("[F1DI] Data freshness alert", msg)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("_check_data_freshness failed: %s", exc)

    @staticmethod
    def _run_synthetic_audit() -> None:
        try:
            from f1di.evaluation.synthetic_audit import run_audit
            results = run_audit()
            misaligned = [a for a, r in results.items() if not r.get("skipped") and not r.get("aligned")]
            if misaligned:
                msg = (
                    "F1DI — Synthetic label misalignment detected\n"
                    + "\n".join(
                        f"  {a}: blending hurts {abs(results[a]['acc_delta'])*100:.1f}pp "
                        f"(synth {results[a]['acc_synth']:.3f} vs blend {results[a]['acc_blend']:.3f})"
                        for a in misaligned
                    )
                )
                logger.warning("synthetic_audit misaligned: %s", misaligned)
                try:
                    from f1di.delivery.notifier import send_system_alert
                    send_system_alert("[F1DI] Synthetic label misalignment", msg)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("_run_synthetic_audit failed: %s", exc)

    @staticmethod
    def _run_retrain() -> None:
        try:
            from f1di.confidence.online import retrain
            result = retrain()
            if not result.get("skipped"):
                logger.info("Scheduled retrain complete: ECE=%.4f", result.get("ece", 0))
        except ImportError:
            pass

    @staticmethod
    def _check_precision_degradation() -> None:
        """Alert via configured delivery channels if any agent precision degrades."""
        try:
            from f1di.confidence.online import check_precision_degradation
            alerts = check_precision_degradation()
            if not alerts:
                return
            lines = ["F1DI — Model precision degradation detected\n"]
            for a in alerts:
                lines.append(
                    f"  {a['agent']}: {a['precision_recent']*100:.0f}% (7d) "
                    f"vs {a['precision_baseline']*100:.0f}% (30d baseline) "
                    f"— drop {a['drop_pp']:.1f}pp  n={a['n_recent']}"
                )
            message = "\n".join(lines)
            logger.warning("precision_degradation: %s", message)
            try:
                from f1di.delivery.notifier import send_system_alert
                send_system_alert("[F1DI] Precision degradation alert", message)
            except Exception as exc:
                logger.debug("Could not send precision degradation alert: %s", exc)
        except Exception as exc:
            logger.warning("_check_precision_degradation failed: %s", exc)

    @staticmethod
    def _run_shadow_auto_promote() -> None:
        """Auto-promote shadow challenger when evaluation criteria are met.

        Criteria: promote=True from shadow_evaluate AND n_shadow >= 20.
        Writes to the same promotions.json used by the manual promote endpoint.
        """
        import json as _json
        from datetime import datetime, timezone
        from pathlib import Path as _Path
        try:
            from f1di.storage.database import db_session
            from f1di.storage.repository import shadow_evaluate as _evaluate
        except ImportError:
            return

        challenger_version = "weights-v2"
        promotions_path = _Path("data/calibration/promotions.json")

        try:
            with db_session() as session:
                evaluation = _evaluate(session, challenger_version)
        except Exception as exc:
            logger.warning("shadow_auto_promote: evaluate failed: %s", exc)
            return

        if not evaluation.get("promote"):
            logger.debug(
                "shadow_auto_promote: not eligible — %s",
                evaluation.get("recommendation", "unknown"),
            )
            return

        if evaluation.get("n_shadow", 0) < 20:
            logger.debug(
                "shadow_auto_promote: insufficient shadow inferences (%d < 20)",
                evaluation.get("n_shadow", 0),
            )
            return

        # Guard: don't promote more than once per day for the same challenger.
        promotions_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict] = []
        if promotions_path.exists():
            try:
                existing = _json.loads(promotions_path.read_text())
            except Exception:
                pass
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        already_today = any(
            r.get("challenger_version") == challenger_version
            and r.get("promoted_at", "")[:10] == today
            for r in existing
        )
        if already_today:
            logger.debug("shadow_auto_promote: already promoted today, skipping")
            return

        record = {
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "challenger_version": challenger_version,
            "forced": False,
            "auto": True,
            **evaluation,
        }
        existing.append(record)
        promotions_path.write_text(_json.dumps(existing, indent=2))
        logger.info(
            "shadow_auto_promoted challenger=%s n_shadow=%d p=%.4f",
            challenger_version, evaluation.get("n_shadow", 0), evaluation.get("p_value", 1.0),
        )
        try:
            from f1di.delivery.notifier import send_system_alert
            send_system_alert(
                "[F1DI] Shadow challenger auto-promoted",
                f"Challenger '{challenger_version}' auto-promoted.\n"
                f"n_shadow={evaluation.get('n_shadow')}  "
                f"p={evaluation.get('p_value', 1):.4f}  "
                f"rbc={evaluation.get('rank_biserial_correlation', 0):.3f}",
            )
        except Exception:
            pass

    @staticmethod
    def _run_race_backtest() -> None:
        """Compute per-round precision/recall from stored insight+feedback pairs."""
        try:
            from f1di.evaluation.race_backtest import run_backtest
            result = run_backtest()
            if result.get("alert"):
                msg = (
                    f"F1DI — Precision alert: overall precision {result['overall_precision']:.1%} "
                    f"is below threshold {result['alert_threshold']:.1%} "
                    f"(n={result['n_total']}  trend={result['trend']})"
                )
                logger.warning("race_backtest_alert: %s", msg)
                try:
                    from f1di.delivery.notifier import send_system_alert
                    send_system_alert("[F1DI] Backtest precision alert", msg)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("_run_race_backtest failed: %s", exc)

    def _run_pull(self) -> bool:
        """Pull new knowledge docs. Returns True when at least one new round was ingested."""
        from f1di.storage.database import db_session
        from f1di.storage.repository import already_ingested, mark_ingested

        any_new = False
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
                        any_new = True
                        # Immediately label the newly ingested round rather than waiting for the next cycle.
                        try:
                            from f1di.data.outcome_labeler import label_race
                            report = label_race(year, round_num)
                            logger.info(
                                "post-ingest outcome label: year=%d round=%d correct=%d incorrect=%d",
                                year, round_num, report.n_labeled_correct, report.n_labeled_incorrect,
                            )
                        except Exception as exc2:
                            logger.warning("post-ingest outcome label failed [%d/R%d]: %s", year, round_num, exc2)
                    except Exception as exc:
                        logger.warning("Ingestion failed [%s/%d/R%d]: %s", source, year, round_num, exc)
                    break  # ingest_fn pulls the full year in one call
        return any_new

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
