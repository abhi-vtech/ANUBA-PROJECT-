"""Test suite.

This file is load-bearing: some dependency in .venv installs a top-level
`tests` package into site-packages, and without an __init__.py ours is only a
namespace package, which loses the name to that regular package. With it, the
repo copy (reached via '' at the front of sys.path) wins under both the system
interpreter and the venv.
"""
