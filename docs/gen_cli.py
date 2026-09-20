#!/usr/bin/env python3
"""Refreshes the ``ecp --help`` blocks in docs/cli.rst.

Run it from the repo root after changing any help text in ecp-cli, then run
docs/build.sh. It is deliberately not part of build.sh, since it rewrites a
tracked file.

Only the ``.. code-block:: text`` blocks are replaced. Each one is matched to
its command by the ``Usage:`` line already inside it, so the page's own prose,
headings and example commands are left untouched.
"""

import pathlib
import re
import subprocess
import sys
import textwrap

CLI_RST = pathlib.Path("docs/cli.rst")
BINARY = pathlib.Path("target/debug/ecp")
# Every help block starts with one of these lines.
ROOT_USAGE = "Usage: ecp <COMMAND>"
SUBCOMMAND_USAGE = re.compile(r"^Usage: ecp ([a-z][a-z-]*)", re.M)
# --memory-limit-gb's default is this machine's RAM, so drop just that one.
MACHINE_DEFAULT = re.compile(r"(Defaults to 80% of RAM) \[default: \d+\]")
TEXT_BLOCK = re.compile(r"(?P<head>\.\. code-block:: text\n\n)(?P<body>(?:(?: +[^\n]*)?\n)+)")


def help_text(args: list[str]) -> str:
    """The help output for a command, as it should appear in the page."""
    result = subprocess.run(
        [str(BINARY), *args, "-h" if args else "--help"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(
            f"{CLI_RST} documents `ecp {' '.join(args)}`, which the CLI does not have. "
            "Delete that section, or fix the command name in its Usage line."
        )
    out = result.stdout
    # A subcommand's description is already the page's heading and prose.
    if args:
        out = out[out.index("Usage:") :]
    return MACHINE_DEFAULT.sub(r"\1", out).rstrip() + "\n"


def replace_block(match: re.Match[str]) -> str:
    """Rewrites one help block, keeping anything that isn't help output."""
    body = match["body"]
    # The match runs to the next heading, so hand back the blank lines as they were.
    lines = body.splitlines(keepends=True)
    trailing = ""
    while lines and not lines[-1].strip():
        trailing = lines.pop() + trailing
    if ROOT_USAGE in body:
        args: list[str] = []
    else:
        found = SUBCOMMAND_USAGE.search(textwrap.dedent(body))
        if not found:
            return match[0]
        args = [found[1]]
    return match["head"] + textwrap.indent(help_text(args), "   ") + trailing


def subcommands() -> list[str]:
    """Every subcommand the CLI offers, as `ecp --help` lists them."""
    listing = help_text([]).partition("Commands:")[2].partition("\n\nOptions:")[0]
    names = [line.split()[0] for line in listing.splitlines() if line.strip()]
    return [name for name in names if name != "help"]


def main() -> None:
    if not CLI_RST.exists():
        sys.exit(f"run this from the repo root: {CLI_RST} not found")
    subprocess.run(["cargo", "build", "-q", "-p", "ecp-cli"], check=True)

    page = CLI_RST.read_text()
    updated, count = TEXT_BLOCK.subn(replace_block, page)
    CLI_RST.write_text(updated)
    print(f"refreshed {count} help blocks in {CLI_RST}")

    # A new subcommand needs a heading and prose of its own, which only a person
    # can write, so say so rather than inventing a section.
    undocumented = [name for name in subcommands() if f"Usage: ecp {name} " not in updated]
    if undocumented:
        print(f"still undocumented, add a section for: {', '.join(undocumented)}")


if __name__ == "__main__":
    main()
