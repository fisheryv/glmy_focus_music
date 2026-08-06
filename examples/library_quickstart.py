"""Minimal state-sequence example for the public library."""

from focus_topology import AnalysisConfig, analyze_states


def main() -> None:
    states = [0, 1, 2, 0, 1, 2, 0]
    result = analyze_states(
        states,
        config=AnalysisConfig(thresholds=(0.95, 0.8, 0.5)),
        metadata={"track_id": "demo-cycle", "view": "pitch"},
    )

    print("H0:", result.betti_curve(0))
    print("H1:", result.betti_curve(1))
    print("Directed recurrence:", result.metrics["directed_recurrence"])


if __name__ == "__main__":
    main()
