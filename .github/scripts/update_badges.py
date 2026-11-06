"""Rewrite the README badge block from the LINT, DOCS and COVERAGE environment variables.

Each workflow sets only the variables it measures; badges without a variable are kept as they are.
"""

import os
import re
from pathlib import Path
from urllib.parse import quote

README = Path("README.md")
BLOCK = re.compile(r"(<!-- badges:start -->\n)(.*?)(<!-- badges:end -->)", re.DOTALL)
LABEL = re.compile(r"^\[!\[(.*?)\]")


def shield(label: str, message: str, color: str, link: str) -> str:
    """Return a markdown image link to a static shields.io badge."""
    esc = lambda s: quote(s.replace("-", "--").replace("_", "__"))  # noqa: E731
    return f"[![{label}](https://img.shields.io/badge/{esc(label)}-{esc(message)}-{color})]({link})"


def coverage_color(pct: float) -> str:
    """Map docstring or test coverage to a badge color."""
    for threshold, color in ((90, "brightgreen"), (75, "green"), (60, "yellow"), (50, "orange")):
        if pct >= threshold:
            return color
    return "red"


def main() -> None:
    """Replace the content between the badge markers in README.md."""
    run = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    new = []
    if "LINT" in os.environ:
        lint = int(os.environ["LINT"])
        new.append(shield("Ruff linting issues", "passing" if lint == 0 else f"{lint} issues",
                          "brightgreen" if lint == 0 else "yellow", run))
    if "DOCS" in os.environ:
        docs = float(os.environ["DOCS"])
        new.append(shield("Docstring coverage", f"{docs:g}%", coverage_color(docs), run))
    if "COVERAGE" in os.environ:
        cov = float(os.environ["COVERAGE"])
        new.append(shield("Test coverage", f"{cov:g}%", coverage_color(cov), run))

    text = README.read_text()
    match = BLOCK.search(text)
    if not match:
        raise SystemExit("README.md is missing the <!-- badges:start/end --> markers")
    # keyed by label: replace existing badges in place, append new ones
    badges = {}
    for line in [*match[2].splitlines(), *new]:
        if not line.strip():
            continue
        label = LABEL.match(line)
        if not label:
            raise SystemExit(f"README.md badge block has a line that is not a badge: {line!r}")
        badges[label[1]] = line
    block = "".join(f"{line}\n" for line in badges.values())
    README.write_text(f"{text[:match.start(2)]}{block}{text[match.end(2):]}")


if __name__ == "__main__":
    main()
