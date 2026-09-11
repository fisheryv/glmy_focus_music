from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_ltsn_v5_gate_passing_reference.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_ltsn_v5_reference", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_reference_passes_all_counterfactual_gates(tmp_path: Path) -> None:
    module = _load_script()
    result = module.build_reference(ROOT, tmp_path)

    assert result == {
        "ok": True,
        "output_dir": str(tmp_path.resolve()),
        "calibration_gate_passed": True,
        "development_gate_passed": True,
        "qualification_gate_passed": True,
        "reference_only": True,
    }
    for filename in (
        "calibration_reference.json",
        "guidance_development_reference.json",
        "qualification_reference.json",
    ):
        payload = json.loads((tmp_path / filename).read_text(encoding="utf-8"))
        assert payload["reference_only"] is True
        assert payload["synthetic"] is True
        assert payload["scientific_evidence"] is False
        assert payload["production_authorization"] is False

    qualification = json.loads(
        (tmp_path / "qualification_reference.json").read_text(encoding="utf-8")
    )
    assert qualification["qualification_passed"] is True
    assert all(qualification["gates"].values())
    assert qualification["metrics"]["interval_90_coverage"] == 0.9

    with (tmp_path / "v5_vs_passing_reference.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        comparison = list(csv.DictReader(handle))
    assert comparison
    assert all(row["reference_passed"] == "true" for row in comparison)
    assert any(row["source_passed"] == "false" for row in comparison)
