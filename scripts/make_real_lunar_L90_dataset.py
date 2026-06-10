import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "vesp.data.real_gravity",
            "--model",
            "gl0420a",
            "--n-query",
            "1024",
            "--degree-min",
            "31",
            "--degree-max",
            "90",
            "--output",
            "data/lunar_grail_gl0420a_L90_residual.csv",
            "--acceleration-output",
            "physical",
        ],
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
