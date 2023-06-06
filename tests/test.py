"""pylab testing module"""

#! IMPORTS


import os
import sys

sys.path += [os.getcwd()]
import rslib


#! MAIN


if __name__ == "__main__":
    # io
    print("TESTING TDF DATA READING")
    TDF_FILE = os.path.join(os.getcwd(), "tests", "tdf_sample.tdf")
    tdf_data = rslib.read_tdf(TDF_FILE)
    print(tdf_data)
    print("")

    print("TESTING EMT DATA READING")
    EMT_FILE = os.path.join(os.getcwd(), "tests", "emt_sample.emt")
    emt_data = rslib.read_emt(EMT_FILE)
    print(emt_data)
    print("")

    print("TESTING COSMED XLSX DATA READING")
    COSMED_FILE = os.path.join(os.getcwd(), "tests", "cosmed_sample.xlsx")
    cosmed_data, participant = rslib.read_cosmed_xlsx(COSMED_FILE)
    print(cosmed_data)
    print(participant)
