"""rslib testing module"""

#! IMPORTS


from os.path import dirname, join
import sys

sys.path += [dirname(__file__)]
sys.path += [join(dirname(dirname(__file__)), "src")]

from rslib import *


#! MAIN


if __name__ == "__main__":
    # io
    print("TESTING TDF DATA READING")
    tdf_data = read_tdf("tdf_sample.tdf")
    print(tdf_data)
    print("")

    print("TESTING EMT DATA READING")
    emt_data = read_emt("emt_sample.emt")
    print(emt_data)
    print("")

    print("TESTING COSMED XLSX DATA READING")
    cosmed_data, participant = read_cosmed_xlsx("cosmed_sample.xlsx")
    print(cosmed_data)
    print(participant)
