# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import sys
import os

from clang.cindex import Config, Index
# -

Config.set_library_path(r"C:\Program Files\LLVM\bin")

translation_unit = Index.create().parse("test.cpp")
