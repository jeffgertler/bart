#!/usr/bin/env python


import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                                                os.path.abspath(__file__)))))
from bart import BART


if __name__ == "__main__":
    ps = BART(1.0, 1000.0, 0.0)
