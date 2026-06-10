from __future__ import annotations

import asyncio
import logging
from datetime import date

logger = logging.getLogger("f1di.ingestion.scheduler")

_DEFAULT_INTERVAL_HOURS = 6
_CURRENT_YEAR = date.today().year


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
            cycle += 1
            if cycle % 4 == 0:
                try:
                    await asyncio.get_event_loop().run_in_executor(None, self._run_retrain)
                except Exception as exc:
                    logger.warning("Calibrator retrain failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                pass

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
