import json
from pathlib import Path

from pyglmy.cli import main


def test_path_cli_writes_json(tmp_path: Path) -> None:
    graph = tmp_path / "graph.json"
    output = tmp_path / "result.json"
    graph.write_text(
        json.dumps(
            {
                "vertices": [0, 1, 2],
                "edges": [[0, 1], [1, 2], [2, 0]],
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "path",
            str(graph),
            "--max-dimension",
            "1",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["betti_numbers"] == [1, 1]
