from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "generation" / "prompts" / "ace_rerank_formal.csv"
DEFAULT_OUTPUT = ROOT / "metadata" / "ltsn_prompts.csv"

DEVELOPMENT = {"p04_piano_strings", "p16_bell_texture", "p23_rhodes_bass", "p28_flowing_texture"}
CALIBRATION = {
    "p08_ambient_electronic",
    "p19_strings_piano",
    "p24_tape_keys",
    "p31_subtle_polyrhythm",
}
QUALIFICATION = {"p12_harp_guitar", "p20_chamber_ambient", "p27_neutral_ambient", "p32_pulse_drone"}

VARIANTS = (
    "Keep the arrangement sparse and continuous with gradual eight-bar changes.",
    "Use a narrow melodic range and seamless transitions without dramatic accents.",
    "Maintain restrained dynamics and a stable background texture throughout.",
    "Emphasize gentle repetition with subtle timbral evolution and no breakdowns.",
    "Use smooth four-bar phrases and avoid abrupt harmonic or rhythmic changes.",
    "Keep the foreground minimal while the accompaniment remains even and predictable.",
    "Favor long connected layers with soft attacks and controlled high frequencies.",
    "Use understated recurring motifs with low contrast between adjacent sections.",
    "Maintain an unbroken concentration-friendly flow with restrained bass movement.",
    "Use gradual harmonic motion and a consistent pulse without fills or drops.",
    "Keep musical events evenly spaced with soft articulation and limited ornamentation.",
    "Favor warm balanced sonorities and slow texture changes without climactic gestures.",
    "Use a stable meter and quiet recurring figures with smooth phrase boundaries.",
    "Maintain low distraction through simple voicing and gently evolving ambience.",
    "Keep density moderate and consistent with no sudden pauses or prominent solos.",
    "Use calm cyclical phrasing and subtle variation while preserving an even energy level.",
)

FIELDS = ("prompt_id", "caption", "split", "seed", "bpm", "keyscale", "timesignature")


def _split(prompt_id: str) -> str:
    if prompt_id in DEVELOPMENT:
        return "development"
    if prompt_id in CALIBRATION:
        return "calibration"
    if prompt_id in QUALIFICATION:
        return "qualification"
    return "train"


def build_rows(source: Path = SOURCE) -> list[dict[str, str]]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        base_rows = list(csv.DictReader(handle))
    base_ids = {row["prompt_id"] for row in base_rows}
    reserved = DEVELOPMENT | CALIBRATION | QUALIFICATION
    if len(base_rows) != 32 or len(base_ids) != 32 or not reserved <= base_ids:
        raise ValueError("the frozen 32-prompt source is incomplete or has duplicate IDs")

    rows: list[dict[str, str]] = []
    for base in base_rows:
        split = _split(base["prompt_id"])
        caption = base["caption"].rstrip(". ")
        for index, variant in enumerate(VARIANTS, start=1):
            rows.append(
                {
                    "prompt_id": f"{base['prompt_id']}__v{index:02d}",
                    "caption": f"{caption}. {variant}",
                    "split": split,
                    "seed": "",
                    "bpm": base.get("bpm", ""),
                    "keyscale": base.get("keyscale", ""),
                    "timesignature": base.get("timesignature", ""),
                }
            )

    counts = Counter(row["split"] for row in rows)
    if counts != {"train": 320, "development": 64, "calibration": 64, "qualification": 64}:
        raise ValueError(f"unexpected LTSN split counts: {dict(counts)}")
    if len({row["prompt_id"] for row in rows}) != len(rows):
        raise ValueError("generated LTSN prompt IDs are not unique")
    if len({row["caption"] for row in rows}) != len(rows):
        raise ValueError("generated LTSN captions are not unique")
    return rows


def write_manifest(output: Path, rows: list[dict[str, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    rows = build_rows(args.source)
    write_manifest(args.output, rows)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "prompts": len(rows),
                "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
                "seeds_per_prompt": 4,
                "planned_trajectories": len(rows) * 4,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
