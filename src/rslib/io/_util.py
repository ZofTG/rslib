"""
io._util

a library containing functions to be used for reading and writing files.
These functions are thought to be used internally and not directly from the
user.

Functions
---------
check_writing_file
    check the provided filename and rename it if required.

assert_file_extension
    check the validity of the input path file to be a str with the provided
    extension.
"""


#! IMPORTS


from os.path import exists
from tkinter.messagebox import askyesno

import pandas as pd
from numpy import unique, ndarray

__all__ = ["check_entry", "check_writing_file", "assert_file_extension"]


#! FUNCTIONS


def check_entry(
    entry: object,
    mask: ndarray,
):
    """
    check_entry _summary_

    _extended_summary_

    Parameters
    ----------
    entry : object
        the object to be checked

    mask : ndarray
        the column mask to be controlled. The mask has to match all the columns
        contained by levels at index > 1.

    Raises
    ------
    TypeError
        "entry must be a pandas DataFrame."
        In case the entry is not a pandas.DataFrame.

    TypeError
        "entry columns must be a pandas MultiIndex."
        In case the entry columns are not a pandas.MultiIndex instance.

    TypeError
        "entry columns must contain {mask}."
        In case the entry columns does not match with the provided mask.

    TypeError
        "entry index must be a pandas Index."
        In case the index of the entry is not a pandas.Index
    """
    if not isinstance(entry, pd.DataFrame):
        raise TypeError("entry must be a pandas DataFrame.")
    if not isinstance(entry.columns, pd.MultiIndex):
        raise TypeError("entry columns must be a pandas MultiIndex.")
    umask = unique(mask.astype(str), axis=0)
    for lbl in unique(entry.columns.get_level_values(0)):
        imask = entry[lbl].columns.to_frame().values.astype(str)
        imask = unique(imask, axis=0)
        if not (imask == umask).all():
            raise TypeError(f"entry columns must contain {mask}.")
    if not isinstance(entry.index, pd.Index):
        raise TypeError("entry index must be a pandas Index.")


def check_writing_file(
    file: str,
):
    """
    check the provided filename and rename it if required.

    Parameters
    ----------
    file : str
        the file path

    Returns
    -------
    filename: str
        the file (renamed if required).
    """
    ext = file.rsplit(".", 1)[-1]
    filename = file
    while exists(filename):
        msg = f"The {file} file already exist.\nDo you want to replace it?"
        yes = askyesno(title="Replace", message=msg)
        if yes:
            return filename
        filename = file.replace(f".{ext}", f"_1.{ext}")
    return filename


def assert_file_extension(
    path: str,
    ext: str,
):
    """
    check the validity of the input path file to be a str with the provided
    extension.

    Parameters
    ----------
    path : str
        the object to be checked

    ext : str
        the target file extension

    Raises
    ------
    err: AsserttionError
        in case the file is not a str or it does not exist or it does not have
        the provided extension.
    """
    assert isinstance(path, str), "path must be a str object."
    msg = path + f' must have "{ext}" extension.'
    assert path.rsplit(".", 1)[-1] == f"{ext}", msg
