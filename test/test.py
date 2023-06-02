"""pylab testing module"""

import os
import pylab

if __name__ == "__main__":
    tdf_file = os.path.sep.join([os.getcwd(), "test", "tdf_test.tdf"])
    tdf_data = pylab.read_tdf(tdf_file)
    check = 1
