import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only tests (brief §6)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash.model as dm  # noqa: E402

dm._cuda_time = time.perf_counter  # let return_stats work on CPU (wrapper, no source edit)
