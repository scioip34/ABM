"""
metarunner.py: including the main classes and methods to programatically ruzn metaprotocols of simulations, i.e.
     for some parameter search.
"""

import numpy as np
from dotenv import dotenv_values

envconf = dotenv_values(".env")


class Constant:
    """A constant parameter value for a given parameter that shall be used for simulations"""

    def __init__(self, var_name, constant):
        """defines a single variable value with name"""
        self.tunable = Tunable(var_name, values_override=[constant])

    def print(self):
        """printing method"""
        print(f"Constant {self.tunable.name} = {self.tunable.values[0]}")


class Tunable:
    """A parameter range in which we want to loop through (explore)"""

    def __init__(self, var_name, min_v=None, max_v=None, num_data_points=None, values_override=None):
        """init method of the Tunable class. We want to loop through a parameter defined with var_name from the
        min value to the max value with num_data_points number of individual parameter values between

        In case we have specific values to loop through we can pass a list of values instead of borders and number
        of datapoints."""

        if min_v is None and values_override is None:
            raise Exception("Neither value borders nor override values have been given to create Tunable!")
        elif min_v is not None and values_override is not None:
            raise Warning("Both value borders and override values are defined when creating Tunable, using override"
                          "values as default!")

        self.name = var_name
        if values_override is None:
            self.min_val = min_v
            self.max_val = max_v
            self.n_data = num_data_points
            self.generated = True
            self.values = np.linspace(self.min_val, self.max_val, num=self.n_data, endpoint=True)
        else:
            self.min_val = min(values_override)
            self.max_val = max(values_override)
            self.n_data = len(values_override)
            self.generated = False
            self.values = values_override

    def print(self):
        """printing method"""
        print(f"Tunable: {self.name} = {self.min_val}  -  -  -n={self.n_data}-  -  -  {self.max_val}")
        print(f"Values : {self.values}")


class MetaProtocol:
    """Metaprotocol class that is initialized with Tunables and runs through the desired simulations accordingly"""
    def __init__(self):
        self.default_envconf = envconf
        self.tunables = []

    def add_criterion(self, criterion):
        """Adding a criterion to the metaprotocol as a Tunable or Constant"""
        self.tunables.append(criterion)
        print("---Added new criterion to MetaProtocol: ")
        criterion.print()