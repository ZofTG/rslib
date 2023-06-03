"""pylab testing module"""

import os

# import sys

# sys.path += [os.getcwd()]
import rslib

if __name__ == "__main__":
    TDF_FILE = os.path.sep.join([os.getcwd(), "tests", "tdf_sample.tdf"])
    tdf_data = rslib.read_tdf(TDF_FILE)
    check = 1
