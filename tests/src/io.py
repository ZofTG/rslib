"""rslib.io testing module"""

#! IMPORTS


from os.path import dirname, join
import sys

sys.path += [join(dirname(dirname(dirname(__file__))), "src")]

from rslib.io import *
from numpy import isclose


__all__ = ["test_io"]


#! FUNCTION


def test_io():
    """test input-output module"""

    # setup the data path
    READ_PATH = join(dirname(dirname(__file__)), "assets", "read")
    WRITE_PATH = join(dirname(dirname(__file__)), "assets", "write")

    # read_trc
    print("\nTESTING TRC DATA READING")
    trc_rd = read_trc(join(READ_PATH, "read.trc"))
    print(trc_rd)

    # write_trc
    print("\nTESTING TRC DATA WRITING")
    TRC_FILE = join(WRITE_PATH, "write.trc")
    write_trc(TRC_FILE, trc_rd)
    trc_wt = read_trc(TRC_FILE)
    print(f"trc: read == write: {isclose(trc_wt - trc_rd, 0).all().all()}")

    # read_mot
    print("\nTESTING MOT DATA READING")
    mot_rd = read_mot(join(READ_PATH, "read.mot"))
    print(mot_rd)

    # write_mot
    print("\nTESTING MOT DATA WRITING")
    MOT_FILE = join(WRITE_PATH, "write.mot")
    write_mot(MOT_FILE, mot_rd)
    mot_wt = read_mot(MOT_FILE)
    print(f"mot: read == write: {isclose(mot_wt - mot_rd, 0).all().all()}")

    # read_cosmed_xlsx
    print("\nTESTING COSMED XLSX DATA READING")
    cosmed_data, participant = read_cosmed_xlsx(join(READ_PATH, "read.xlsx"))
    print(cosmed_data)
    print(participant)

    # read_emt
    print("\nTESTING EMT DATA READING")
    emt_rd = read_emt(join(READ_PATH, "read.emt"))
    print(emt_rd)

    # read_tdf
    print("TESTING TDF DATA READING")
    tdf_rd = read_tdf(join(READ_PATH, "read.tdf"))
    print(tdf_rd)


if __name__ == "__main__":
    test_io()
