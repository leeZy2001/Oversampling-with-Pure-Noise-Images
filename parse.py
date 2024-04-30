from dataclasses import dataclass
from typing import Any
import tomllib


@dataclass
class Args:
    """A container for arguments parsed with `parse_args`."""

    flags: set[str]
    keywords: dict[str, str]
    positional: list[str]

    def __getitem__(self, key):
        if isinstance(key, slice) or isinstance(key, int):
            return self.positional[key]
        if key in self.keywords:
            return self.keywords[key]
        return key in self.flags


def parse_args(args: list[str]) -> Args:
    """
       Parses a list of arguments (such as `sys.argv`) by the
       following rules:
       - Items of the form `--flag` are flags with name `flag`.
       - Items of the form `--key=value` are keyword arguments with key
         `key` and value `value`. 
       - All other items are treated as positional arguments.
       - If the entry `--` is encountered, all subsequent entries are
         treated as positional, regardless of flag-like notation. The
         first occurrence of `--` is not included (but subsequent
         occurrences of `--` would be.)
    """
    flags = set()
    kwargs = {}
    posargs = []
    forced_positional = False
    for arg in args:
        if forced_positional or not arg.startswith("--"):
            posargs.append(arg)
            continue
        if arg == "--":
            forced_positional = True
            continue
        parts = arg.split('=')
        if len(parts) == 1:
            flags.add(parts[0][2:])
            continue
        kwargs[parts[0][2:]] = '='.join(parts[1:])
    return Args(flags, kwargs, posargs)


def parse_toml(filename: str) -> dict[str, Any]:
    with open(filename, 'rb') as f:
        data = tomllib.load(f)
    return data

