# README

## Install

Fedora: `sudo dnf install graphviz graphviz-devel` needed before `pip install pygraphviz`

Ubuntu: `sudo apt-get install graphviz graphviz-dev` needed before `pip install pygraphviz`

## Python with Nix

> Try using Nix Python packages as much as possible.

Since we are on NixOS, we use shell hooks and `.venv/` for python package loading. REMEMBER to delete the `.venv/` directory when updating the `requirements.txt`.
