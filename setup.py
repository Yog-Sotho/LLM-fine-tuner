"""
setup.py — legacy shim.

All packaging metadata (dependencies, extras, packages, entry points) lives in
pyproject.toml. This file only lets tools that still call ``setup.py`` work;
do not add metadata here — setuptools overrides it with pyproject.toml anyway.
"""

from setuptools import setup

setup()
