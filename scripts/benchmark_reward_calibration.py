"""Frozen interface for the owner-run O1 runtime and memory gate."""

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    condition: str

    def __post_init__(self) -> None:
        if self.condition not in {"baseline", "h4", "h12"}:
            raise ValueError("Unsupported benchmark condition.")

    @property
    def horizon(self) -> int:
        return 4 if self.condition == "h4" else 12


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the owner-run O1 calibration benchmark."
    )
    parser.add_argument("--condition", choices=("baseline", "h4", "h12"), required=True)
    parser.parse_args()


if __name__ == "__main__":
    main()
