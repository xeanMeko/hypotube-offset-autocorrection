#!/usr/bin/env python3
"""Compile the editable German LaTeX manual and package its source files.

Requires pdfLaTeX with German language support. Python uses only its standard
library. Run from the repository root: python scripts/build_user_manual.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
NAME = "benutzerhandbuch-de"
SOURCES = [f"{NAME}.tex", "assets/bedienoberflaeche.png", "LATEX-ANLEITUNG.txt"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="pdflatex", help="pdfLaTeX executable or path")
    args = parser.parse_args()
    engine = shutil.which(args.engine)
    if engine is None:
        parser.error("pdfLaTeX was not found. Install LaTeX with German language support; "
                     "see docs/LATEX-ANLEITUNG.txt.")

    with tempfile.TemporaryDirectory(prefix=".manual-", dir=DOCS) as folder:
        build = Path(folder)
        command = [engine, "-interaction=nonstopmode", "-halt-on-error",
                   "-file-line-error", "-no-shell-escape",
                   f"-output-directory={build}", f"{NAME}.tex"]
        for run in (1, 2):
            result = subprocess.run(command, cwd=DOCS, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise SystemExit(f"LaTeX run {run} failed:\n{result.stdout}")

        log = (build / f"{NAME}.log").read_text(errors="replace")
        problems = [line for line in log.splitlines()
                    if any(word in line for word in ("Overfull", "Warning:", "Missing character"))]
        if problems:
            print("LaTeX diagnostics:\n" + "\n".join(problems))

        archive = build / f"{NAME}-latex.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in SOURCES:
                bundle.write(DOCS / name, name)
            bundle.write(build / f"{NAME}.pdf", f"{NAME}.pdf")

        for artifact in (f"{NAME}.pdf", archive.name):
            (build / artifact).replace(DOCS / artifact)
            print(DOCS / artifact)


if __name__ == "__main__":
    main()
