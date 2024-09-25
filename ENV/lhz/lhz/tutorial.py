"""
Created on Mon Apr  9 13:33:26 2018

@author: kili
"""

from lhz.qutip_hdicts import Hdict_physical
import lhz.core as lhz_core
import lhz.spinglass_utility as spinglass
import numpy as np

# basic usage:
Nl = 4
Np = lhz_core.n_physical(Nl)

Jij1 = np.random.random((Nl, Nl))
Jij2 = Jij1[np.triu_indices_from(Jij1, 1)]

qd = lhz_core.qubit_dict(Nl)

cs1 = lhz_core.create_constraintlist(Nl)
cs2 = []
for c in cs1:
    cs2.append([qd[i] for i in c])

# one can either give Jij matrix as linear list (Np) or full Jij matrix in
# logical space (NlxNl matrix) and give constraints in 'base-free' format
# or with a linear index (which has to fit Jij matrix then)
Hd1 = Hdict_physical(Jij1)
Hd2 = Hdict_physical(Jij2)
Hd3 = Hdict_physical(Jij2, constraints=cs1)
Hd4 = Hdict_physical(Jij2, constraints=cs2)

state = [1, 1, 1, -1, -1, 1]
print('num of violated constraints for state', str(state), ' : ', spinglass.num_of_violated_constraints(state, cs2))

# check if all versions are identical
print(Hd1 == Hd2 == Hd3 == Hd4)  # prints True
