# -*- coding: UTF-8 -*-
"""
@file:__init__.py.py
@author: Wei Jie
@date: 2023/1/17
@description:
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

if __name__ == "__main__":
    print()
