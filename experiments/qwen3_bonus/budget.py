"""Rental-clock accounting. Ending a training process DOES NOT stop cloud billing."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RentalBudget:
    hourly_rate: float
    cap_yuan: float
    started_at: float
    quantum_seconds: int = 60

    def __post_init__(self):
        if any(not math.isfinite(v) for v in (self.hourly_rate, self.cap_yuan, self.started_at)):
            raise ValueError('Budget values must be finite')
        if self.hourly_rate <= 0 or self.cap_yuan <= 0 or self.quantum_seconds <= 0:
            raise ValueError('Rate, cap and billing quantum must be positive')

    def cost(self, now: float) -> float:
        if not math.isfinite(now) or now < self.started_at:
            raise ValueError('Rental start must not be in the future')
        billed = math.ceil((now-self.started_at)/self.quantum_seconds)*self.quantum_seconds
        return billed*self.hourly_rate/3600

    def remaining_seconds(self, now: float, reserve_yuan: float = 2) -> float:
        self.cost(now)
        return max(0, (self.cap_yuan-reserve_yuan)*3600/self.hourly_rate-(now-self.started_at))

    def deadline(self, reserve_yuan: float = 1, save_seconds: float = 120) -> float:
        if reserve_yuan < 0 or reserve_yuan >= self.cap_yuan or save_seconds < 0:
            raise ValueError('Invalid checkpoint reserve')
        return self.started_at+(self.cap_yuan-reserve_yuan)*3600/self.hourly_rate-save_seconds-self.quantum_seconds

    def can_start(self, now: float, reserve_yuan: float = 2) -> bool:
        return self.remaining_seconds(now, reserve_yuan) > 0


def parse_start(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Rental timestamp must include timezone, e.g. +08:00')
    return parsed.timestamp()
