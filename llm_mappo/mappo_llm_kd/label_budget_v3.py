"""Shared phase reservations; unknown outcomes remain fully reserved."""

from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from .labeling import AtomicLabelStore


class LabelBudgetV3:
    def __init__(self, path, *, job_id, job_cap_usd):
        self.path = Path(path)
        self.job_id = job_id
        self.job_cap = _amount(job_cap_usd)
        self.store = AtomicLabelStore(self.path)
        if not isinstance(job_id, str) or not job_id or not 0 < self.job_cap <= 50:
            raise ValueError("The medium job cap must lie in (0,50] USD.")

    def reserve(self, request_id, amount_usd):
        amount = _amount(amount_usd)
        if not isinstance(request_id, str) or not request_id or amount <= 0:
            raise ValueError("A request requires an identity and positive reservation.")
        with self._locked():
            reservations = self._read()
            if request_id in reservations:
                raise ValueError("Request reservation already exists.")
            phase, job = self._totals(reservations)
            if phase + amount > 100:
                raise ValueError("Shared phase budget would exceed 100 USD.")
            if job + amount > self.job_cap:
                raise ValueError("The frozen job budget would be exceeded.")
            self._append("start", request_id, amount)

    def settle(self, request_id, amount_usd):
        amount = _amount(amount_usd)
        with self._locked():
            reservations = self._read()
            previous = reservations.get(request_id)
            if (previous is None or previous["job_id"] != self.job_id
                    or previous["settled"] or amount > previous["reserved"]):
                raise ValueError("Settlement does not match an open reservation.")
            self._append("settlement", request_id, amount)

    def require_settled(self):
        with self._locked():
            if any(row["job_id"] == self.job_id and not row["settled"]
                   for row in self._read().values()):
                raise ValueError("An unsettled request requires outcome reconciliation.")

    def totals(self):
        with self._locked():
            phase, job = self._totals(self._read())
        return {"phase_reserved_or_spent_usd": float(phase),
                "job_reserved_or_spent_usd": float(job),
                "phase_cap_usd": 100.0,
                "historical_medium_reference_usd": 23.0797}

    def _totals(self, reservations):
        phase = sum((row["amount"] for row in reservations.values()), Decimal(0))
        job = sum((row["amount"] for row in reservations.values()
                   if row["job_id"] == self.job_id), Decimal(0))
        return phase, job

    def _read(self):
        reservations = {}
        for event in self.store.records():
            if (set(event) != {"schema", "event", "request_id", "job_id",
                               "job_cap_usd", "amount_usd"}
                    or event["schema"] != "label-budget-event-v3"):
                raise ValueError("Shared budget ledger event is invalid.")
            key, job = event["request_id"], event["job_id"]
            cap, amount = _amount(event["job_cap_usd"]), _amount(event["amount_usd"])
            if (not isinstance(key, str) or not isinstance(job, str)
                    or not 0 < cap <= 50):
                raise ValueError("Shared budget ledger identity is invalid.")
            if job == self.job_id and cap != self.job_cap:
                raise ValueError("The job budget cap changed across resume.")
            if event["event"] == "start" and key not in reservations and amount > 0:
                reservations[key] = {"job_id": job, "reserved": amount,
                                     "amount": amount, "settled": False}
            elif (event["event"] == "settlement" and key in reservations
                  and not reservations[key]["settled"]
                  and reservations[key]["job_id"] == job
                  and amount <= reservations[key]["reserved"]):
                reservations[key].update(amount=amount, settled=True)
            else:
                raise ValueError("Shared budget ledger sequence is invalid.")
        return reservations

    def _append(self, event, request_id, amount):
        self.store.append({
            "schema": "label-budget-event-v3", "event": event,
            "request_id": request_id, "job_id": self.job_id,
            "job_cap_usd": str(self.job_cap), "amount_usd": str(amount),
        })

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        # Never reclaim an abandoned lock or unresolved request automatically.
        lock.open("xb").close()
        try:
            yield
        finally:
            lock.unlink()


def _amount(value):
    if isinstance(value, bool):
        raise ValueError("USD amount must be finite and nonnegative.")
    try:
        amount = Decimal(str(value))
    except Exception as error:
        raise ValueError("USD amount is invalid.") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError("USD amount must be finite and nonnegative.")
    return amount
