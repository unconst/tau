"""Shared runtime budget across parent and subagents."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict


@dataclass
class BudgetSnapshot:
    """Immutable budget view for telemetry."""

    max_cost: float
    consumed_cost: float
    reserved_cost: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    reasoning_tokens: int = 0

    @property
    def remaining_cost(self) -> float:
        return max(0.0, self.max_cost - self.consumed_cost - self.reserved_cost)


class AgentBudget:
    """Thread-safe budget shared by nested agent loops."""

    def __init__(self, max_cost: float):
        self._max_cost = max(0.0, float(max_cost))
        self._consumed_cost = 0.0
        self._reserved_cost = 0.0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cached_tokens = 0
        self._reasoning_tokens = 0
        self._reservations: Dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def max_cost(self) -> float:
        return self._max_cost

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                max_cost=self._max_cost,
                consumed_cost=self._consumed_cost,
                reserved_cost=self._reserved_cost,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                cached_tokens=self._cached_tokens,
                reasoning_tokens=self._reasoning_tokens,
            )

    def can_reserve(self, amount: float) -> bool:
        amount = max(0.0, float(amount))
        with self._lock:
            return (self._consumed_cost + self._reserved_cost + amount) <= self._max_cost

    def reserve(self, key: str, amount: float) -> bool:
        amount = max(0.0, float(amount))
        if not key:
            return False
        with self._lock:
            if (self._consumed_cost + self._reserved_cost + amount) > self._max_cost:
                return False
            self._reservations[key] = self._reservations.get(key, 0.0) + amount
            self._reserved_cost += amount
            return True

    def release(self, key: str) -> None:
        if not key:
            return
        with self._lock:
            amount = self._reservations.pop(key, 0.0)
            self._reserved_cost = max(0.0, self._reserved_cost - amount)

    def consume(
        self,
        *,
        cost: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        reservation_key: str | None = None,
    ) -> bool:
        """Consume usage budget, returning False when budget exceeded."""
        cost = max(0.0, float(cost))
        with self._lock:
            covered_by_reservation = 0.0
            if reservation_key:
                covered_by_reservation = min(
                    cost,
                    self._reservations.get(reservation_key, 0.0),
                )
            unreserved_cost = max(0.0, cost - covered_by_reservation)
            if (self._consumed_cost + self._reserved_cost + unreserved_cost) > self._max_cost:
                return False
            self._consumed_cost += cost
            if covered_by_reservation > 0.0 and reservation_key:
                remaining = self._reservations.get(reservation_key, 0.0) - covered_by_reservation
                if remaining <= 1e-9:
                    self._reservations.pop(reservation_key, None)
                else:
                    self._reservations[reservation_key] = remaining
                self._reserved_cost = max(0.0, self._reserved_cost - covered_by_reservation)
            self._input_tokens += max(0, int(input_tokens))
            self._output_tokens += max(0, int(output_tokens))
            self._cached_tokens += max(0, int(cached_tokens))
            self._reasoning_tokens += max(0, int(reasoning_tokens))
            return True

    def is_exhausted(self) -> bool:
        with self._lock:
            return (self._consumed_cost + self._reserved_cost) >= self._max_cost
