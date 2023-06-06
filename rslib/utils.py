"""
UTILS

module containing several utilities that can be used for multiple purposes.

Classes
-------
Participant
    an instance defining the general parameters of one subject during a test.

Functions
---------
magnitude
    get the order of magnitude of a numeric scalar value according to the
    specified base.

get_files
    get the full path of the files contained within the provided folder
    (and optionally subfolders) having the provided extension.
"""


#! IMPORTS


import os
from datetime import date, datetime

import numpy as np
import pandas as pd


__all__ = [
    "Participant",
    "magnitude",
    "get_files",
]

#! CLASSES


class Participant:
    """
    class containing all the data relevant to a participant.

    Parameters
    ----------
    surname: str | None = None
        the participant surname

    name: str | None = None
        the participant name

    gender: str | None = None
        the participant gender

    height: int | float | None = None
        the participant height

    weight: int | float | None = None
        the participant weight

    age: int | float | None = None
        the participant age

    birth_date: date | None = None
        the participant birth data
    """

    # class variables
    _name = None
    _surname = None
    _gender = None
    _height = None
    _weight = None
    _birth_date = None
    _recording_date = date  # type:ignore
    _units = {
        "fullname": "",
        "surname": "",
        "name": "",
        "gender": "",
        "height": "m",
        "weight": "kg",
        "bmi": "kg/m^2",
        "birthdate": "",
        "age": "years",
        "hrmax": "bpm",
    }

    def __init__(
        self,
        surname: str | None = None,
        name: str | None = None,
        gender: str | None = None,
        height: int | float | None = None,
        weight: int | float | None = None,
        age: int | float | None = None,
        birth_date: date | None = None,
        recording_date: date = datetime.now().date,  # type: ignore
    ):
        self.set_surname(surname)
        self.set_name(name)
        self.set_gender(gender)
        self.set_height((height / 100 if height is not None else height))
        self.set_weight(weight)
        self.set_age(age)
        self.set_birthdate(birth_date)
        self._recording_date = recording_date

    def set_surname(
        self,
        surname: str | None,
    ):
        """
        set the participant surname.

        Parameters
        ----------
        surname: str | None,
            the surname of the participant.
        """
        if surname is not None:
            assert isinstance(surname, str), "'surname' must be a string."
        self._surname = surname

    def set_name(
        self,
        name: str | None,
    ):
        """
        set the participant name.

        Parameters
        ----------
        name: str | None
            the name of the participant.
        """
        if name is not None:
            assert isinstance(name, str), "'name' must be a string."
        self._name = name

    def set_gender(
        self,
        gender: str | None,
    ):
        """
        set the participant gender.

        Parameters
        ----------
        gender: str | None
            the gender of the participant.
        """
        if gender is not None:
            assert isinstance(gender, str), "'gender' must be a string."
        self._gender = gender

    def set_height(
        self,
        height: int | float | None,
    ):
        """
        set the participant height in meters.

        Parameters
        ----------
        height: int | float | None
            the height of the participant.
        """
        if height is not None:
            txt = "'height' must be a float or int."
            assert isinstance(height, (int, float)), txt
        self._height = height

    def set_weight(
        self,
        weight: int | float | None,
    ):
        """
        set the participant weight in kg.

        Parameters
        ----------
        weight: int | float | None
            the weight of the participant.
        """
        if weight is not None:
            txt = "'weight' must be a float or int."
            assert isinstance(weight, (int, float)), txt
        self._weight = weight

    def set_age(
        self,
        age: int | float | None,
    ):
        """
        set the participant age in years.


        Parameters
        ----------
        age: int | float | None,
            the age of the participant.
        """
        if age is not None:
            txt = "'age' must be a float or int."
            assert isinstance(age, (int, float)), txt
        self._age = age

    def set_birthdate(
        self,
        birth_date: date | None,
    ):
        """
        set the participant birth_date.

        Parameters
        ----------
        birth_date: datetime.date | None
            the birth date of the participant.
        """
        if birth_date is not None:
            txt = "'BirthDate' must be a datetime.date or datetime.datetime."
            assert isinstance(birth_date, (datetime, date)), txt
            if isinstance(birth_date, datetime):
                self._birth_date = birth_date.date()
            else:
                self._birth_date = birth_date
        else:
            self._birth_date = birth_date

    @property
    def surname(self):
        """get the participant surname"""
        return self._surname

    @property
    def name(self):
        """get the participant name"""
        return self._name

    @property
    def gender(self):
        """get the participant gender"""
        return self._gender

    @property
    def height(self):
        """get the participant height in meter"""
        return self._height

    @property
    def weight(self):
        """get the participant weight in kg"""
        return self._weight

    @property
    def birthdate(self):
        """get the participant birth date"""
        return self._birth_date

    @property
    def bmi(self):
        """get the participant BMI in kg/m^2"""
        if self.height is None or self.weight is None:
            return None
        return self.weight / (self.height**2)

    @property
    def fullname(self):
        """
        get the participant full name.
        """
        return f"{self.surname} {self.name}"

    @property
    def age(self):
        """
        get the age of the participant in years
        """
        if self._age is not None:
            return self._age
        if self._birth_date is not None:
            return int((self._recording_date - self._birth_date).days // 365)  # type: ignore
        return None

    @property
    def hrmax(self):
        """
        get the maximum theoretical heart rate according to Gellish.

        References
        ----------
        Gellish RL, Goslin BR, Olson RE, McDonald A, Russi GD, Moudgil VK.
            Longitudinal modeling of the relationship between age and maximal
            heart rate.
            Med Sci Sports Exerc. 2007;39(5):822-9.
            doi: 10.1097/mss.0b013e31803349c6.
        """
        if self.age is None:
            return None
        return 207 - 0.7 * self.age

    @property
    def units(self):
        """return the unit of measurement of the stored data."""
        return self._units

    def copy(self):
        """return a copy of the object."""
        return Participant(
            name=self.name,
            surname=self.surname,
            birth_date=self.birthdate,
            height=self.height,
            weight=self.weight,
            age=self.age,
            gender=self.gender,
            recording_date=self._recording_date,  # type: ignore
        )

    @property
    def dict(self):
        """return a dict representation of self"""
        return {
            "surname": self.surname,
            "name": self.name,
            "gender": self.gender,
            "birthdate": self.birthdate,
            "age [" + self.units["age"] + "]": self.age,
            "height [" + self.units["height"] + "]": self.height,
            "weight [" + self.units["weight"] + "]": self.weight,
            "bmi [" + self.units["bmi"] + "]": self.bmi,
            "hrmax [" + self.units["hrmax"] + "]": self.hrmax,
        }

    @property
    def series(self):
        """return a pandas.Series representation of self"""
        vals = ["surname", "name", "gender", "birthdate", "age"]
        vals += ["height", "weight", "bmi", "hrmax"]
        vals = pd.MultiIndex.from_tuples([(i, self.units[i]) for i in vals])
        return pd.Series(list(self.dict.values()), index=vals)

    @property
    def dataframe(self):
        """return a pandas.DataFrame representation of self"""
        return pd.DataFrame(self.series).T

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.dataframe.__str__()


#! FUNCTIONS


def magnitude(
    value: int | float,
    base: int | float = 10,
):
    """
    return the order in the given base of the value

    Parameters
    ----------
        value: int | float
            the value to be checked

        base:int | float=10
            the base to be used to define the order of the number

    Returns
    -------
        mag float
            the number required to elevate the base to get the value
    """
    if value == 0 or base == 0:
        return 0
    else:
        return np.log(abs(value)) / np.log(base)


def get_files(
    path: str,
    extension: str = "",
    check_subfolders: bool = False,
):
    """
    list all the files having the required extension in the provided folder
    and its subfolders (if required).

    Parameters
    ----------
        path: str
            a directory where to look for the files.

        extension: str
            a str object defining the ending of the files that have to be
            listed.

        check_subfolders: bool
            if True, also the subfolders found in path are searched,
            otherwise only path is checked.

    Returns
    -------
        files: list
            a list containing the full_path to all the files corresponding
            to the input criteria.
    """

    # output storer
    out = []

    # surf the path by the os. walk function
    for root, _, files in os.walk(path):
        for obj in files:
            if obj[-len(extension) :] == extension:
                out += [os.path.join(root, obj)]

        # handle the subfolders
        if not check_subfolders:
            break

    # return the output
    return out
