"""Rewrite the README badge block from the LINT and DOCS environment variables."""

import os
import re
from pathlib import Path
from urllib.parse import quote

README = Path("README.md")
BLOCK = re.compile(r"(<!-- badges:start -->\n).*?(<!-- badges:end -->)", re.DOTALL)


def shield(label: str, message: str, color: str, link: str) -> str:
    """Return a markdown image link to a static shields.io badge."""
    esc = lambda s: quote(s.replace("-", "--").replace("_", "__"))  # noqa: E731
    return f"[![{label}](https://img.shields.io/badge/{esc(label)}-{esc(message)}-{color})]({link})"


def docs_color(pct: float) -> str:
    """Map docstring coverage to a badge color."""
    for threshold, color in ((90, "brightgreen"), (75, "green"), (60, "yellow"), (50, "orange")):
        if pct >= threshold:
            return color
    return "red"


def main() -> None:
    """Replace the content between the badge markers in README.md."""
    lint = int(os.environ["LINT"])
    docs = float(os.environ["DOCS"])
    run = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"

    badges = "\n".join([
        shield("Ruff linting issues", "passing" if lint == 0 else f"{lint} issues",
               "brightgreen" if lint == 0 else "yellow", run),
        shield("Docstring coverage", f"{docs:g}%", docs_color(docs), run),
    ])

    text = README.read_text()
    if not BLOCK.search(text):
        raise SystemExit("README.md is missing the <!-- badges:start/end --> markers")
    README.write_text(BLOCK.sub(lambda m: f"{m[1]}{badges}\n{m[2]}", text, count=1))


if __name__ == "__main__":
    main()
