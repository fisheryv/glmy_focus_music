"""Compatibility wrapper for the historical ``cli`` module."""

from focus_topology.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
