"""
Wrapper class for ensemble parameter values
Most will be int or float valued
"""

class ParameterValue:
    def __init__(self, key, val):
        if isinstance(val, dict):
            self.name = val['name']
            self.value = val['values']
        else:
            self.name = key
            self.value = val

    def isMulti(self):
        return isinstance(self.value, dict)

    def __str__(self):
        if not self.isMulti():
            return str(self.value)
        else:
            return self.name


class EnsembleParameter:
    def __init__(self, key, paramData):
        self.param_key = key
        self.param_value_set = [ParameterValue(key, val) for val in paramData]

    def dir_names(self):
        return [f"{key}_{str(val)}" for key, val in self]

    """
    ChatGPT wrote this method so use with caution
    """
    def __iter__(self):
        return iter([(self.param_key, val) for val in self.param_value_set])
