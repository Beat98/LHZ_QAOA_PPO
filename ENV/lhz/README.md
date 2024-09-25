## Conventions:
- spins are named by their corresponding entry in the J_ij matrix (starting with 0) and with a tuple (eg (0,1), (0,2), ...)
- for a linear numbering of the spins numpy.triu_indidices(N,1) is used (see function `qubit_dict`) 
- Nl = Number of logical spins
- Np = Number of physical spins

## Temporary usage (without installation)
```
import sys
sys.path.append('/path/to/lhz/folder') # the one containing setup.py
import lhz.lhz_core as lhz_core
import lhz.lhz_qutip as lhz_qutip
```
## Dependencies
To install the lhz you need to install some dependencies:
- pip install cython

## Installation:
- go to folder, do `pip install .` attention: once installed i think changes to source wont be updated before uninstalling/reinstalling

## Installation 2:
`python setup.py develop`
only creates a link that points to source so changes will be updated

## Deinstall:
- pip uninstall lhz