"""

now directly integrated into page, should become kind of obsolete

"""
from qaoa.QutipQaoa import QutipQaoa
from qaoa_analytics.analytic_commutations_for_energy_expectation_value import Line, Page, Operator, PreFactor

import sympy as sp
import numpy as np
from lhz.core import create_constraintlist, qubit_dict, n_logical
from time import time

# maybe move this to page and return a string which can be directly used in sympy
def get_string_expression_from_page(page):
    expr = ''

    for line in page.lines:
        expr += '+' + get_string_expression_from_line(line)

    return expr[1:]

def get_string_expressions_from_page(page):
    exprs = []

    for line in page.lines:
        # line.print()
        exprs.append(get_string_expression_from_line(line))

    return exprs

def get_mathematica_strings_from_page(page):
    exprs = []

    for line in page.lines:
        temp = get_string_expression_from_line(line)
        temp = sp.sympify(temp)
        exprs += [sp.printing.mathematica.mathematica_code(temp)]

    return exprs


def get_string_expression_from_line(line):
    expr = ''

    for pfkey, pfexp in zip(line.preFactor.keys, line.preFactor.exponents):
        expr += '*' + pfkey + '**' + str(pfexp)

    for op in line.Objects:
        tempstr = symbol_dict[op.typus]

        tempstr = tempstr.replace('zs', ''.join(['*Z%d' % x for x in op.includedQubits])[1:])
        tempstr = tempstr.replace('xs', ''.join(['*X%d' % x for x in op.includedQubits])[1:])  # only for testing
        tempstr = tempstr.replace('_', str(op.includedQubits)[1:-1])
        tempstr = tempstr.replace('pp', op.parameterName)

        expr += '*' + tempstr

    return expr[1:]


symbol_dict = {
    # '1': '1',
    'Uc': 'exp(-I*pp*zs)',
    'Ucd': 'exp(+I*pp*zs)',
    'C': 'zs',
    'Z': 'zs',
    'Uz': 'exp(-I*pp*J_*zs)',
    'Uzd': 'exp(+I*pp*J_*zs)'
    # 'Ux': 'exp(-I*pp*xs)',  # only for testing
    # 'Uxd': 'exp(+I*pp*xs)'  # only for testing
}


# rewrite this into:
#  a) simplified_page, wf_dag_functions, hp_funs, parameter_names = create_page_and_functions(jijs, programstring, cs = None, cstr=1.0)
#  b) expval = evaluatefunctions(wf_dag_functions, hp_funs, parameters)

def do_alles(jijs, programstring, parameters, cs=None, cstr=1.0):
    Np = len(jijs)
    Nl = n_logical(Np)

    if cs is None:
        qd = qubit_dict(Nl)
        cs0 = create_constraintlist(Nl)

        cs = [[qd[c] for c in cc] for cc in cs0]

    translation = {'Z': ('Uzd', 'Uz'), 'X': ('Uxd', 'Ux'), 'C': ('Ucd', 'Uc')}

    # problem independent part

    leftside = []

    parameternames = []
    for i, letter in enumerate(programstring):
        if letter == 'C':
            parameternames.append('a'+str(i))
        else:
            parameternames.append(('b' if letter == 'Z' else 'c') + str(i))

    for letter, pname in zip(programstring, parameternames):
        if letter == 'C':
            for constraint in cs:  # -------------------this minus is because of (1-C)--------v
                leftside.append(Operator(constraint, translation[letter][0], parameter_name='(-'+str(cstr)+')*' + pname))
        else:
            for j in range(Np):
                leftside.append(Operator([j], translation[letter][0], parameter_name=pname))

    allparamnames = ['J%d' % k for k in range(Np)] + parameternames + ['Z%d' % k for k in range(Np)]

    tt = time()
    p = Page([Line(leftside)], allparamnames)

    # p.print()
    # p.push_all_x_to_left_and_simplify()
    p.push_all_x_to_left_smart()

    for dothissmarter in range(5):
        p.push_all_x_to_left_and_simplify()
    # p.print()
    tcreatesimplify = time() - tt

    # problem dependent part
    expops = []
    for j in range(Np):
        expops.append(Line([Operator([j], 'Z', '')], PreFactor({'('+str(jijs[j])+')': '1'})))

    for constraint in cs:  # --------------------------------------------v this minus is because of (1-C)
        expops.append(Line([Operator(constraint, 'C', '')], PreFactor({'(-' + str(cstr) + ')': '1'})))

    # this line is for (1-C) (the one at the end of the function is faster)
    # expops.append(Line([Operator([], '1', '')], PreFactor({'(' + str(cstr) + ')*' + str(len(cs)): 1})))


    confparams = ['Z%d' % k for k in range(Np)]

    lines = get_string_expressions_from_page(p)

    tt = time()
    leftside_functions = []

    numl = len(p.lines)
    for il, l in enumerate(lines):
        leftside_functions.append(sp.lambdify(allparamnames, sp.sympify(l), 'numpy'))
        print(il, 'of', numl, 'done lambdify(sympifying)')

    exp_op_funs = []
    for eop in expops:
        # eop.print()
        exp_op_funs.append(sp.lambdify(confparams, sp.sympify(get_string_expression_from_line(eop)), 'numpy'))

    tsympfiylambdify = time() - tt

    # for a in exp_op_funs:
    #     print(a.__doc__)

    def all_confs(N):
        return [[2 * int(j) - 1 for j in list(np.binary_repr(i, width=N))] for i in range(2 ** N)]

    tt = time()
    wfandH = []
    wert = 0
    for ic, conf in enumerate(all_confs(Np)):

        tls = 0
        for leftside_function in leftside_functions:
            tls += leftside_function(*(jijs + parameters + conf))
        # print(tls/np.sqrt(2**Np))

        tops = 0
        for exp_op_fun in exp_op_funs:
            tops += exp_op_fun(*conf)
            # print(exp_op_fun(*conf))
        # print(tops)
        wfandH.append((tls / np.sqrt(2 ** Np), tops))

        wert += tls * tops * np.conjugate(tls)
        print('numeric evaluation %d/%d done' % (ic, 2**Np))

    wert /= 2**Np

    wert += len(cs)*cstr
    teval = time() - tt
    # print('time page: ', tcreatesimplify)
    # print('time lamb: ', tsympfiylambdify)
    # print('time eval: ', teval)

    return wert, p, wfandH, {'page': tcreatesimplify, 'lamb': tsympfiylambdify, 'eval': teval}


def do_alles_better_wo_sympy(jijs, programstring, parameters, cs=None, cstr=1.0):
    tt = time()
    p = Page(jijs=jijs, programstring=programstring, cs=cs, cstr=cstr)

    p.push_all_x_to_left_smart()

    for dothissmarter in range(5):
        # p.push_all_x_to_left_and_simplify()
        p.push_all_x_to_left_smart()
    p.push_all_x_to_left_and_simplify()
    # p.print()
    tcreatesimplify = time() - tt

    Np = len(jijs)
    Nl = n_logical(Np)

    # problem dependent part
    expops = []
    for j in range(Np):
        expops.append(Line([Operator([j], 'Z', '')], PreFactor({'(' + str(jijs[j]) + ')': '1'})))

    if cs is None:
        qd = qubit_dict(Nl)
        cs0 = create_constraintlist(Nl)

        cs = [[qd[c] for c in cc] for cc in cs0]

    for constraint in cs:  # --------------------------------------------v this minus is because of (1-C)
        expops.append(Line([Operator(constraint, 'C', '')], PreFactor({'(-' + str(cstr) + ')': '1'})))

    # this line is for (1-C) (the one at the end of the function is faster)
    # expops.append(Line([Operator([], '1', '')], PreFactor({'(' + str(cstr) + ')*' + str(len(cs)): 1})))

    confparams = ['Z%d' % k for k in range(Np)]

    tt = time()
    leftside_functions = []

    numl = len(p.lines)
    for il in range(numl):
        leftside_functions.append(p.get_lambda_function_from_line(il))
        print(il, 'of', numl, 'done directly making lambda')

    exp_op_funs = []
    for eop in expops:
        # eop.print()
        exp_op_funs.append(sp.lambdify(confparams, sp.sympify(get_string_expression_from_line(eop)), 'numpy'))

    tsympfiylambdify = time() - tt

    # for a in exp_op_funs:
    #     print(a.__doc__)

    def all_confs(N):
        return [[2 * int(j) - 1 for j in list(np.binary_repr(i, width=N))] for i in range(2 ** N)]

    tt = time()
    wfandH = []
    wert = 0
    for ic, conf in enumerate(all_confs(Np)):

        tls = 0
        for leftside_function in leftside_functions:
            tls += leftside_function(*(jijs + parameters + conf))
        # print(tls/np.sqrt(2**Np))

        tops = 0
        for exp_op_fun in exp_op_funs:
            tops += exp_op_fun(*conf)
            # print(exp_op_fun(*conf))
        # print(tops)
        wfandH.append((tls / np.sqrt(2 ** Np), tops))

        wert += tls * tops * np.conjugate(tls)
        print('numeric evaluation %d/%d done' % (ic, 2 ** Np))

    wert /= 2 ** Np

    wert += len(cs) * cstr
    teval = time() - tt
    # print('time page: ', tcreatesimplify)
    # print('time lamb: ', tsympfiylambdify)
    # print('time eval: ', teval)

    return wert, p, wfandH, {'page': tcreatesimplify, 'lamb': tsympfiylambdify, 'eval': teval}


if __name__ == '__main__':
    from lhz.qutip_hdicts import Hdict_physical

    jijs = [0.1, -0.8, 2.3, -1.1, -0.1, 0.45, 0.1, 0.7, -0.1, -1.2]
    jijs = jijs[:6]

    cstrength = 1.9
    Np = len(jijs)
    Nl = n_logical(Np)

    # pstr = 'CXZXCZX'
    pstr = 'ZXC'

    q = QutipQaoa(Hdict_physical(jijs, cstrength=cstrength), pstr, jijs)
    res = q.mc_optimization()

    ps = res.parameters
    energy = res.final_objective_value

    t0 = time()
    wert2, pu2,  _2, td2 = do_alles_better_wo_sympy(jijs, pstr, ps, cs=None, cstr=cstrength)
    td = {'page': 0, 'lamb': 0, 'eval': 0}
    wert = 0
    # wert, pu, _, td = do_alles(jijs, pstr, ps, cs=None, cstr=cstrength)

    t1 = time()

    print('time page old: %.3f, new: %.3f' % (td['page'], td2['page']))
    print('time lamb old: %.3f, new: %.3f' % (td['lamb'], td2['lamb']))
    print('time eval old: %.3f, new: %.3f' % (td['eval'], td2['eval']))

    print(wert, wert2, wert-wert2)
    print(wert2, energy, '\n --- if this is 1 its cool: ', wert2.real/energy)
    # TODO find the bug! this is not working