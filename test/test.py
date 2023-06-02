"""pylab testing module"""

import os
import pylab

if __name__ == "__main__":
    TDF_FILE = os.path.sep.join([os.getcwd(), "test", "tdf_test_file.tdf"])
    tdf_data = pylab.read_tdf(TDF_FILE)
    check = 1
