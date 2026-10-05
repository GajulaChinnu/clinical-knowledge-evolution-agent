"""Command-line interface to execute CKEA Phase 12 Evaluation Framework suites."""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.services.config_service import load_config
from app.services.evaluation_runner import EvaluationRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CKEA Phase 12 Evaluation Runner: execute evaluation suites and generate reports."
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "extraction", "comparison", "impact", "briefing", "governance", "e2e", "calibration"],
        help="Target evaluation suite to execute (default: all)",
    )
    parser.add_argument(
        "--live-llm",
        action="store_true",
        default=False,
        help="Opt-in flag to execute real live Groq LLM inferences instead of deterministic offline fixtures.",
    )
    args = parser.parse_args()

    config = load_config()
    print("=" * 75)
    print(" CKEA PHASE 12: CLINICAL KNOWLEDGE EVOLUTION AGENT EVALUATION RUNNER")
    print("=" * 75)
    print(f" Target Suite:   {args.suite}")
    print(f" Execution Mode: {'LIVE GROQ LLM' if args.live_llm else 'OFFLINE / DETERMINISTIC FIXTURES'}")
    print(f" Output Reports: data/evaluation/reports/")
    print("-" * 75)

    try:
        runner = EvaluationRunner(config=config, live_llm=args.live_llm)
        report = runner.run_all(suite_name=args.suite)

        print("\n" + "=" * 75)
        print(" EVALUATION EXECUTION SUMMARY")
        print("=" * 75)
        print(f" Total Cases Evaluated: {report.total_cases}")
        print(f" Passed:                {report.total_passed}")
        print(f" Expected Holds (G1-G4): {report.total_held_expected}")
        print(f" Failed:                {report.total_failed}")
        print(f" Errors:                {report.total_errors}")
        print(f" Overall Status:        {'SUCCESS / ALL PASSED' if report.all_passed else 'FAILURES DETECTED'}")
        print("-" * 75)
        print(" Reports written to data/evaluation/reports/:")
        for f in Path("data/evaluation/reports").glob("*.json"):
            print(f"  - {f.name}")
        print("=" * 75)

        return 0 if report.all_passed else 1

    except Exception as e:
        print(f"\nFATAL: Evaluation runner failed: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
