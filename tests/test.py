"""test the rslib library"""


#! IMPORTS


from os.path import dirname
import sys

sys.path += [dirname(dirname(__file__))]
from tests import *


#! FUNCTIONS


def test_all():
    """test all rslib functionalities"""
    test_signalprocessing()
    test_regression()
    test_io()


#! MAIN


if __name__ == "__main__":
    test_all()
