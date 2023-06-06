"""
rslib package

A library containing helpful classes and functions to speed up the lab
data processing.

Libraries
---------
io
    a library containing functions to read specifically formatted files
    such as BtsBioengineering tdf and emt formats or Cosmed-formatted xlsx
    files

Modules
-------
signalprocessing
    a set of functions dedicated to the processing and analysis of 1D signals

utils
    module containing several utilities that can be used for multiple purposes
"""

from .io import *
from .signalprocessing import *
from .utils import *
