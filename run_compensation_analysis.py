from __future__ import annotations

import argparse

import compensation_analysis
import compensation_analysis_mu_incr
import compensation_analysis_mu_max_and_mu_incr
import compensation_analysis_mu_max_none


SCENARIO_RUNNERS = {
    "mu_max": compensation_analysis.main,
    "mu_max_no_constraint": compensation_analysis_mu_max_none.main,
    "mu_incr": compensation_analysis_mu_incr.main,
    "mu_max_and_mu_incr": compensation_analysis_mu_max_and_mu_incr.main,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified entry point for compensation-analysis workflows."
        )
    )
    parser.add_argument(
        "scenario",
        choices=sorted(SCENARIO_RUNNERS),
        help="Which compensation-analysis workflow to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SCENARIO_RUNNERS[args.scenario]()


if __name__ == "__main__":
    main()
