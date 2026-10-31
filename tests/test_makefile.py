import re
from pathlib import Path

MAKEFILE = Path(__file__).resolve().parents[1] / "Makefile"

REQUIRED_TARGETS = [
    "help",
    "setup",
    "lint",
    "test",
    "data",
    "labels",
    "features",
    "train",
    "evaluate",
    "score",
    "serve",
    "monitor",
    "all",
]


def _defined_targets(text: str) -> set[str]:
    return set(re.findall(r"^([A-Za-z_][\w-]*):", text, flags=re.MULTILINE))


def _help_lines(text: str) -> str:
    match = re.search(r"^help:\n((?:\t.*\n?)+)", text, flags=re.MULTILINE)
    assert match, "help target has no recipe"
    return match.group(1)


def test_all_pipeline_targets_are_defined() -> None:
    defined = _defined_targets(MAKEFILE.read_text(encoding="utf-8"))
    missing = [t for t in REQUIRED_TARGETS if t not in defined]
    assert not missing, f"Makefile lacks targets: {missing}"


def test_help_lists_every_target() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    help_recipe = _help_lines(text)
    not_listed = [t for t in REQUIRED_TARGETS if not re.search(rf"\b{t}\b", help_recipe)]
    assert not not_listed, f"make help does not mention: {not_listed}"


def test_all_targets_are_phony() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    phony = set()
    for line in text.splitlines():
        if line.startswith(".PHONY:"):
            phony.update(line.split(":", 1)[1].split())
    not_phony = [t for t in REQUIRED_TARGETS if t not in phony]
    assert not not_phony, f"targets not declared .PHONY: {not_phony}"
