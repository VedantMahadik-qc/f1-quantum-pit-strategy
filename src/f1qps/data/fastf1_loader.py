"""Fetch + cache real race data via FastF1, and pull out SC/VSC periods.

Network-dependent. Not used by the test suite -- see scripts/fetch_sc_data.py
for the one-off manual fetch.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import fastf1
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".fastf1_cache")

# FastF1 TrackStatus codes (each lap's TrackStatus may contain several
# concatenated digits if multiple statuses applied during that lap).
SC_CODE = "4"
VSC_CODES = ("6", "7")


def enable_cache(cache_dir: str = CACHE_DIR) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def load_race(year: int, gp: str) -> fastf1.core.Session:
    session = fastf1.get_session(year, gp, "R")
    session.load()
    return session


def _is_sc_or_vsc(track_status: str) -> bool:
    if not isinstance(track_status, str):
        return False
    return SC_CODE in track_status or any(code in track_status for code in VSC_CODES)


@dataclass
class SCPeriod:
    lap_numbers: list[int]

    @property
    def start_lap(self) -> int:
        return min(self.lap_numbers)

    @property
    def end_lap(self) -> int:
        return max(self.lap_numbers)


def find_sc_vsc_periods(session: fastf1.core.Session) -> list[SCPeriod]:
    """Group laps flagged SC/VSC into contiguous periods by lap number.
    A race can have several separate SC/VSC periods; each becomes its own
    pit-decision QUBO instance.
    """
    laps = session.laps
    flagged = laps[laps["TrackStatus"].apply(_is_sc_or_vsc)]
    lap_numbers = sorted(flagged["LapNumber"].dropna().unique().astype(int).tolist())
    if not lap_numbers:
        return []

    periods: list[list[int]] = [[lap_numbers[0]]]
    for lap in lap_numbers[1:]:
        if lap - periods[-1][-1] <= 1:
            periods[-1].append(lap)
        else:
            periods.append([lap])
    return [SCPeriod(lap_numbers=p) for p in periods]


def laps_in_period(session: fastf1.core.Session, period: SCPeriod) -> pd.DataFrame:
    """All drivers' lap rows for the first lap of the SC/VSC period -- this is
    the decision point: who pits on this lap vs. who stays out.
    """
    laps = session.laps
    return laps[laps["LapNumber"] == period.start_lap].copy()


def green_flag_laps(session: fastf1.core.Session) -> pd.DataFrame:
    """Laps NOT under any yellow/SC/VSC/red condition -- used to fit normal
    tire degradation curves (SC laps run artificially slow behind the pace
    car and would bias degradation estimates).
    """
    laps = session.laps
    return laps[laps["TrackStatus"] == "1"].copy()
