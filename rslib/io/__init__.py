"""
io

a library containing functions to read specifically formatted files such
as BtsBioengineering tdf and emt formats or Cosmed-formatted xlsx files

Modules
-------
btsbioengineering
    read specific BtsBioengineering file formats such as .tdf and .emt
    extensions.

cosmed
    read .xlsx files generated trough the Cosmed Omnia software.
"""

from .btsbioengineering import *
from .cosmed import *
