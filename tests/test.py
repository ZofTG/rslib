"""test the rslib library"""


#! IMPORTS


from os.path import dirname
import sys

sys.path += [dirname(dirname(__file__))]
from tests import *


#! FUNCTIONS


def test_all():
    """test all rslib functionalities"""
    test_regression()
    test_io()
    test_signalprocessing()


#! MAIN


if __name__ == "__main__":
    test_all()
