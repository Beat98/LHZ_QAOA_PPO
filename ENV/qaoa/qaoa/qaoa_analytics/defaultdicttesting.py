from collections import defaultdict


class Pref(defaultdict):
    def __init__(self, initial_values_dict=None):
        if initial_values_dict is not None:
            defaultdict.__init__(self, int, initial_values_dict)
        else:
            defaultdict.__init__(self, int)

    def append(self, other):
        for k, v in other.items():
            self[k] += v
        return self

    # weird hack, no idea what is happening
    def __copy__(self):
        return Pref(self)


start_values = {'2': 1, 'sin': 1}

a = Pref()
a['2'] += 1
b = Pref(start_values)

print('a', a)
print('b', b)

from copy import deepcopy, copy
# c = deepcopy(a)
c = copy(a)
a.append(b)

print('a.append(b)', a)
print('copy(a) before append', c)
