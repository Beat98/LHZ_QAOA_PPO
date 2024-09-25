from qaoa.QutipQaoa import QutipQaoa
from qaoa.legacy_qaoa_classes.QutipGateLHZQaoa import QutipGateLHZQaoa

from lhz.qutip_hdicts import Hdict_physical

nl = 4
# constraints = standard_constraints(nl)
n = nl*(nl-1)//2

Jij = [1, -0.56, 2, -1, 1, -1]*5
Jij = Jij[:n]

pstr = 'VZVZVZ'
pstr = 'ZXCVZV'
pars = [0.3, 0.3, 0.4, 0.1, 0.8, -2]

hd = Hdict_physical(Jij)
qQ = QutipQaoa(hd, pstr, Jij)
qL = QutipGateLHZQaoa(pstr, Jij)

stQ = qQ.run_qaoa(pars)
stL = qL.execute_circuit(pars)

print(stL.overlap(stQ))
print(abs(stL.overlap(stQ)))

