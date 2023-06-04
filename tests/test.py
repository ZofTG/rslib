"""pylab testing module"""

#! IMPORTS


import os
import rslib


#! MAIN


if __name__ == "__main__":
    TDF_FILE = os.path.sep.join([os.getcwd(), "tests", "tdf_sample.tdf"])
    tdf_data = rslib.read_tdf(TDF_FILE)
    print(tdf_data)
