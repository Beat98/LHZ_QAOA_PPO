from copy import deepcopy
import numpy as np
from collections import defaultdict

"""
things to take care of:
-- if all X are taken care of check if simplifications can be done (like C Z1 Z2 C = Z1 Z2)
-- replacing _ with qubit numbers probably only works well with single qubit operators
-- simplifciation may have some bugs
-- if slow take out self.simplify in the loop in push_all_x... -> is slow, but still faster than without
-- http://book.pythontips.com/en/latest/__slots__magic.html  make faster
"""


class PreF(defaultdict):
    def __init__(self, initial_values_dict=None):
        if initial_values_dict is not None:
            defaultdict.__init__(self, int, initial_values_dict)
        else:
            defaultdict.__init__(self, int)

    def append(self, other):
        for k, v in other.items():
            self[k] += v
        return self

    def return_dict(self):
        return self


class PreFactor(object):
    """Prefactor to a specific line, to multiply two prefactors use
    the .return_dict() function in append for the other"""

    def __init__(self, exponent_dictionary=None):
        self.keys = []
        self.exponents = []

        if exponent_dictionary is not None:
            self.append(exponent_dictionary)

    def append(self, exponent_dictionary):
        for key, value in exponent_dictionary.items():
            self.__multiply(key, value)

    def __multiply(self, key, value):
        if key in self.keys:
            ind = self.keys.index(key)
            self.exponents[ind] += value
        else:
            self.keys.append(key)
            self.exponents.append(value)

    def return_dict(self):
        return dict(zip(self.keys, self.exponents))

    # def __get__(self, instance, owner):
    #     return dict(zip(self.keys, self.exponents))


PreFactor = PreF
# pl: parameter comes from original left operator, pr: ... right
# a: C, b: Z, c: X
standard_commutation_rules = {
    ('Uzd', 'Uxd'): [(PreFactor(), 'Uxd', 'Uzd'),
                     (PreFactor({'2': 1, 'sin(pr)': 1, 'sin(pl*J_)': 1}), 'X', 'Z')],
    ('Uzd', 'Ux'): [(PreFactor(), 'Ux', 'Uzd'),
                    (PreFactor({'(-1)': 1, '2': 1, 'sin(pr)': 1, 'sin(pl*J_)': 1}), 'X', 'Z')],
    ('Uz', 'Ux'): [(PreFactor(), 'Ux', 'Uz'),
                   (PreFactor({'2': 1, 'sin(pr)': 1, 'sin(pl*J_)': 1}), 'X', 'Z')],
    ('Uz', 'Uxd'): [(PreFactor(), 'Uxd', 'Uz'),
                    (PreFactor({'(-1)': 1, '2': 1, 'sin(pr)': 1, 'sin(pl*J_)': 1}), 'X', 'Z')],
    ('Ucd', 'Uxd'): [(PreFactor(), 'Uxd', 'Ucd'),
                     (PreFactor({'2': 1, 'sin(pl)': 1, 'sin(pr)': 1}), 'X', 'C')],
    ('Ucd', 'Ux'): [(PreFactor(), 'Ux', 'Ucd'),
                    (PreFactor({'(-1)': 1, '2': 1, 'sin(pl)': 1, 'sin(pr)': 1}), 'X', 'C')],
    ('Uc', 'Uxd'): [(PreFactor(), 'Uxd', 'Uc'),
                    (PreFactor({'(-1)': 1, '2': 1, 'sin(pl)': 1, 'sin(pr)': 1}), 'X', 'C')],
    ('Uc', 'Ux'): [(PreFactor(), 'Ux', 'Uc'),
                   (PreFactor({'2': 1, 'sin(pl)': 1, 'sin(pr)': 1}), 'X', 'C')],
    ('C', 'X'): [(PreFactor({'(-1)': 1}), 'X', 'C')],
    ('Z', 'Ux'): [(PreFactor(), 'Uxd', 'Z')],
    ('Z', 'Uxd'): [(PreFactor(), 'Ux', 'Z')],
    ('Ucd', 'X'): [(PreFactor(), 'X', 'Uc')],
    ('Uzd', 'X'): [(PreFactor(), 'X', 'Uz')],
    ('Uc', 'X'): [(PreFactor(), 'X', 'Ucd')],
    ('C', 'Uxd'): [(PreFactor(), 'Ux', 'C')],
    ('C', 'Ux'): [(PreFactor(), 'Uxd', 'C')],
    ('Z', 'X'): [(PreFactor({'(-1)': 1}), 'X', 'Z')],
    ('Uz', 'X'): [(PreFactor(), 'X', 'Uzd')]
}

standard_general_simplifications = {
    ('Z', 'Z'): (PreFactor(), None),
    ('X', 'X'): (PreFactor(), None),
    ('Uz', 'Uzd'): (PreFactor(), None),
    ('Uzd', 'Uz'): (PreFactor(), None),
    ('Uc', 'Ucd'): (PreFactor(), None),
    ('Ucd', 'Uc'): (PreFactor(), None),
    ('Ux', 'Uxd'): (PreFactor(), None),
    ('Uxd', 'Ux'): (PreFactor(), None),
    ('C', 'C'): (PreFactor(), None)
}

state_simplifications_totheleftonplusstate = {
    'X': PreFactor(),
    'Ux': PreFactor({'exp(-I*pp)': 1}),  # ''should'' be fine now
    'Uxd': PreFactor({'exp(+I*pp)': 1})
}

standard_symbol_dict = {
    'Uc': 'exp(-I*pp*zs)',
    'Ucd': 'exp(+I*pp*zs)',
    'C': 'zs',
    'Z': 'zs',
    'Uz': 'exp(-I*pp*J_*zs)',
    'Uzd': 'exp(+I*pp*J_*zs)'
}


class Operator(object):
    """
    careful, if you want to make a 0,0 constraint it gets fucked up by set
    """

    def __init__(self, included_qubits, typus, parameter_name):
        self.typus = typus
        self.includedQubits = set(included_qubits)
        self.parameterName = parameter_name

        if 'x' in typus or 'X' in typus:
            self.diagonalBasis = 'x'
        else:
            self.diagonalBasis = 'z'


class Line(object):
    def __init__(self, operators, prefactor=None):
        if prefactor is not None:
            self.preFactor = prefactor
        else:
            self.preFactor = PreFactor()
        self.Objects = operators

    def print(self):
        print(self.preFactor.return_dict())
        for op in self.Objects:
            print(op.typus, op.includedQubits, op.parameterName)

    def simplify_general(self, rules):
        did_something = False
        try_again = True

        while try_again:
            did_do_a_break = False
            for i_obj in range(len(self.Objects) - 1):
                if self.Objects[i_obj].includedQubits == self.Objects[i_obj + 1].includedQubits and \
                        self.Objects[i_obj].parameterName == self.Objects[i_obj + 1].parameterName:
                    # does this make it slower? but needs to be there!
                    if (self.Objects[i_obj].typus, self.Objects[i_obj + 1].typus) in rules.keys():
                        did_something = True
                        result = rules[(self.Objects[i_obj].typus, self.Objects[i_obj + 1].typus)]
                        self.preFactor.append(result[0].return_dict())
                        if result[1] is not None:
                            pass  # implement when needed, result[0] is prefactor, result[1] new operators
                        self.Objects.pop(i_obj)
                        self.Objects.pop(i_obj)
                        did_do_a_break = True
                        break
            try_again = did_do_a_break

        return did_something

    def simplify_border(self, rules):
        did_something = False

        if self.Objects[0].typus in rules.keys():
            temp_prefactor_dict = rules[self.Objects[0].typus].return_dict()
            new_pref_dict = {}

            # for k in temp_prefactor_dict.keys():
            #     if 'pp' in k:
            #         temp_prefactor_dict[k.replace('pp', str(self.Objects[0].parameterName))] = temp_prefactor_dict[k]
            #         del temp_prefactor_dict[k]
            for k in temp_prefactor_dict.keys():
                new_pref_dict[k.replace('pp', str(self.Objects[0].parameterName))] = temp_prefactor_dict[k]
                # if 'pp' in k:
                #     temp_prefactor_dict[k.replace('pp', str(self.Objects[0].parameterName))] = temp_prefactor_dict[k]
                #     del temp_prefactor_dict[k]

            self.preFactor.append(new_pref_dict)
            self.Objects.pop(0)
            did_something = True

        return did_something

    def derive_by_parameter(self):

        return None

    # make line only have Z's and spit out more lines with X, do this until only Z are left


class Page(object):
    def __init__(self, start_lines=None, parameter_names=None,
                 jijs=None, programstring=None, cs=None, cstr=1.0,
                 commutation_rules=standard_commutation_rules,
                 general_simplifications=standard_general_simplifications,
                 state_simplifications=state_simplifications_totheleftonplusstate,
                 symbol_dict=standard_symbol_dict):
        """

        :param start_lines:
        :param parameter_names: ALL parameternames, jijs + paramnames + conf
        :param jijs:
        :param programstring:
        :param cs:
        :param cstr:
        :param commutation_rules:
        :param general_simplifications:
        :param state_simplifications:
        :param symbol_dict:
        """

        # setting of start_lines and parameternames depending on case
        if start_lines is not None and parameter_names is not None:
            self.parameter_names = parameter_names
            self.lines = deepcopy(start_lines)
        elif jijs is not None and programstring is not None:
            from lhz.core import create_constraintlist, qubit_dict, n_logical
            Np = len(jijs)
            Nl = n_logical(Np)

            if cs is None:
                qd = qubit_dict(Nl)
                cs0 = create_constraintlist(Nl)

                cs = [[qd[c] for c in cc] for cc in cs0]

            translation = {'Z': ('Uzd', 'Uz'), 'X': ('Uxd', 'Ux'), 'C': ('Ucd', 'Uc')}

            # problem independent part

            self.lines = []

            self.parameter_names = []
            for i, letter in enumerate(programstring):
                if letter == 'C':
                    self.parameter_names.append('a' + str(i))
                else:
                    self.parameter_names.append(('b' if letter == 'Z' else 'c') + str(i))

            for letter, pname in zip(programstring, self.parameter_names):
                if letter == 'C':
                    for constraint in cs:  # -------------------this minus is because of (1-C)--------v
                        self.lines.append(Operator(constraint, translation[letter][0],
                                                   parameter_name='(-' + str(cstr) + ')*' + pname))
                else:
                    for j in range(Np):
                        self.lines.append(Operator([j], translation[letter][0], parameter_name=pname))

            self.lines = [Line(self.lines)]

            # update parameternames
            self.parameter_names = ['J%d' % k for k in range(Np)] + self.parameter_names + ['Z%d' % k for k in range(Np)]
        else:
            assert False, 'blub'

        # setting rest of parameters
        self.commutationRules = commutation_rules
        self.general_simplifications = general_simplifications
        self.state_simplifications = state_simplifications
        self.symbol_dict = symbol_dict
        self.namespace = {
            'I': 1j,
            'exp': np.exp,
            'sin': np.sin
        }

    def push_all_x_to_left_and_simplify(self):
        i = 0
        while self.push_x_to_left() or self.push_x_to_left():
            self.simplify_all_lines()
            i += 1
            # print(i, len(self.lines))
            # print(i)
            # self.print()

        print('simplify done, took %d steps and resulted in %d lines' % (i, len(self.lines)))

        # self.print()

        maxlinelength = 0

        for line in self.lines:
            if maxlinelength < len(line.Objects):
                maxlinelength = len(line.Objects)

        for i in range(maxlinelength):
            self.simplify_all_lines()

    def push_all_x_to_left_smart(self):
        line_index = 0
        # line_count = len(self.lines)

        while line_index < len(self.lines):
            line_done = False

            while not line_done:
                did_something = False
                for i in range(2):
                    for pos, obj in enumerate(self.lines[line_index].Objects):
                        if pos > 0 and ('x' in obj.typus or 'X' in obj.typus):
                            did_something = (did_something or self.commute_two_objects(line_index, pos))

                line_done = not did_something

            if self.state_simplifications is not None:
                self.lines[line_index].simplify_border(self.state_simplifications)
            if self.general_simplifications is not None:
                self.lines[line_index].simplify_general(self.general_simplifications)

            line_index += 1
            print(line_index, 'of', len(self.lines))
        print('simplify done, resulted in %d lines' % (len(self.lines)))

    def push_x_to_left(self):
        """
        sometimes it can happen that it does not do anything, but doing push_x_to_left again
        does something -> check when that happens
        for now always call twice
        :return:
        """
        did_something = False
        # reversed so that newly added lines don't get checked (but these can
        for i_line, line in reversed(list(enumerate(self.lines))):
            for pos, obj in enumerate(line.Objects):
                if pos > 0 and ('x' in obj.typus or 'X' in obj.typus):
                    did_something = (did_something or self.commute_two_objects(i_line, pos))

        return did_something

    def simplify_all_lines(self):
        for line in self.lines:
            if self.state_simplifications is not None:
                line.simplify_border(self.state_simplifications)
            if self.general_simplifications is not None:
                line.simplify_general(self.general_simplifications)

    def commute_two_objects(self, line_index, right_index):
        commuted = False
        left_object = self.lines[line_index].Objects[right_index - 1]
        right_object = self.lines[line_index].Objects[right_index]

        if left_object.diagonalBasis == right_object.diagonalBasis:
            # they commute, as they are diagonal in the same basis (so X with X, and Z/C with Z/C)
            self.lines[line_index].Objects[right_index - 1], self.lines[line_index].Objects[right_index] = \
                right_object, left_object

        elif len(left_object.includedQubits & right_object.includedQubits) == 0:
            # they commute, as no overlapping qubits
            self.lines[line_index].Objects[right_index - 1], self.lines[line_index].Objects[right_index] = \
                right_object, left_object
            # if left_object.diagonalBasis != right_object.diagonalBasis:
            # commuted = True
            # commuted = True  # do this so that it will commute x's out even though nothing really happens
            # or make simplifications better

        # elif left_object.diagonalBasis == right_object.diagonalBasis:
        #     # they commute, as they are diagonal in the same basis (so X with X, and Z/C with Z/C)
        #     self.lines[line_index].Objects[right_index - 1], self.lines[line_index].Objects[right_index] = \
        #         right_object, left_object
        else:
            commuted = True
            # check for type
            rule = self.commutationRules[(left_object.typus, right_object.typus)]

            original_line = self.lines.pop(line_index)

            for resulting_pref, resulting_op_left, resulting_op_right in reversed(rule):
                self.lines.insert(line_index, deepcopy(original_line))
                self.lines[line_index].Objects[right_index - 1] = \
                    Operator(right_object.includedQubits, resulting_op_left,
                             original_line.Objects[right_index].parameterName)
                self.lines[line_index].Objects[right_index] = \
                    Operator(left_object.includedQubits, resulting_op_right,
                             original_line.Objects[right_index - 1].parameterName)

                temp_prefactor_dict = resulting_pref.return_dict()
                new_prefactor_dict = {}
                # to do: replace _ with qubitnumbers in the parameters! (coming from the z-part, so always left)
                for k in temp_prefactor_dict.keys():
                    previous_key = k

                    newkey = previous_key.replace('_', str(left_object.includedQubits)[1:-1])
                    newkey = newkey.replace('pr', right_object.parameterName)
                    newkey = newkey.replace('pl', left_object.parameterName)

                    new_prefactor_dict[newkey] = temp_prefactor_dict[previous_key]

                    # old code, now should be easier without changing the dictionary
                    # if newkey != previous_key:
                    #     temp_prefactor_dict[newkey] = temp_prefactor_dict[previous_key]
                    #     del temp_prefactor_dict[previous_key]

                self.lines[line_index].preFactor.append(new_prefactor_dict)

        return commuted

    def print(self):
        for line in self.lines:
            line.print()

    def get_lambda_function_from_line(self, line_index):
        fstr = ''

        line = self.lines[line_index]

        for pfkey, pfexp in zip(line.preFactor.keys, line.preFactor.exponents):
            fstr += '*' + pfkey + '**' + str(pfexp)

        for op in line.Objects:
            tempstr = self.symbol_dict[op.typus]

            tempstr = tempstr.replace('zs', ''.join(['*Z%d' % x for x in op.includedQubits])[1:])
            tempstr = tempstr.replace('xs', ''.join(['*X%d' % x for x in op.includedQubits])[1:])  # only for testing
            tempstr = tempstr.replace('_', str(op.includedQubits)[1:-1])
            tempstr = tempstr.replace('pp', op.parameterName)

            fstr += '*' + tempstr

        fstr = 'lambda ' + ', '.join(self.parameter_names) + ':' + fstr[1:]
        # print(fstr)
        return eval(fstr, self.namespace)


if __name__ == '__main__':
    from time import time

    # l2 = Line([
    #     Operator([0, 1, 2, 3], 'Ucd', 'a'),
    #     Operator([0], 'Uzd', 'b'), Operator([1], 'Uzd', 'b'),
    #     Operator([2], 'Uzd', 'b'), Operator([3], 'Uzd', 'b'),
    #     Operator([0], 'Uxd', 'c'), Operator([1], 'Uxd', 'c'),
    #     Operator([2], 'Uxd', 'c'), Operator([3], 'Uxd', 'c'),
    #     Operator([0, 1, 2, 3], 'C', '')
    # ])
    #
    # l1 = Line([
    #     Operator([0, 1], 'Ucd', 'a'), Operator([0], 'Uxd', 'c1'), Operator([1], 'Uxd', 'c2')
    # ])

    # constraints without field
    # CX pre-part
    c_pre = lambda s: [Operator([0, 1, 3], 'Ucd', s), Operator([1, 2, 3, 4], 'Ucd', s), Operator([3, 4, 5], 'Ucd', s)]
    x_pre = lambda s: [Operator([0], 'Uxd', s), Operator([1], 'Uxd', s),
                       Operator([2], 'Uxd', s), Operator([3], 'Uxd', s),
                       Operator([4], 'Uxd', s), Operator([5], 'Uxd', s)]


    def create_startline(n, cs, c_param='g', x_param='b', z_param='a'):
        xpart = [Operator([i], 'Uxd', x_param) for i in range(n)]
        zpart = [Operator([i], 'Uzd', z_param) for i in range(n)]
        cpart = [Operator(c, 'Ucd', c_param) for c in cs]

        return Line(cpart + zpart + xpart)


    # l3_0 = Line(c_pre('a') + x_pre('c') + [Operator([0, 1, 3], 'C', 'g')])
    # l3_1 = Line(c_pre('a') + x_pre('c') + [Operator([1, 2, 3, 4], 'C', 'g')])
    # l3_2 = Line(c_pre('a') + x_pre('c') + [Operator([3, 4, 5], 'C', 'g')])

    # p = Page([l2], ['a', 'b', 'c', 'J0', 'J1', 'J2', 'J3', 'Z0', 'Z1', 'Z2', 'Z3'])
    # p = Page([l1], ['a', 'c1', 'c2', 'J0', 'J1', 'Z0', 'Z1'])
    # p = Page([l3_0, l3_1, l3_2], ['a', 'c', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5'])
    # p = Page([l3_0], ['a', 'c', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5'])
    # p = Page(
    #     start_lines=[Line([Operator([0, 1], 'Ucd', 'a1'), Operator([1, 2], 'Ucd', 'a1')]
    #                       + [Operator([0], 'Uxd', 'c1'), Operator([1], 'Uxd', 'c1'), Operator([2], 'Uxd', 'c1')] +
    #                       [Operator([0, 1], 'Ucd', 'a2'), Operator([1, 2], 'Ucd', 'a2')]
    #                       + [Operator([0], 'Uxd', 'c2'), Operator([1], 'Uxd', 'c2'), Operator([2], 'Uxd', 'c2')])],
    #     parameter_names=['a1', 'c1', 'a2', 'c2', 'Z0', 'Z1', 'Z2']
    # )
    # p = Page([Line(c_pre('a1') + x_pre('c1') + c_pre('a2') + x_pre('c2'))], ['a1', 'c1', 'a2', 'c2', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5'])
    l1 = [Line(c_pre('a') + x_pre('b'))]
    p1 = Page(start_lines=l1, parameter_names=['a', 'b', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5'])

    n_phys = 16
    constraints = [[0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7],
                   [4, 5, 8, 9], [5, 6, 9, 10], [6, 7, 10, 11],
                   [8, 9, 12, 13], [9, 10, 13, 14], [10, 11, 14, 15]]

    n_phys = 8
    constraints = [[0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7]]

    n_phys = 6
    constraints = [[0, 1, 3, 4], [1, 2, 4, 5]]

    n_phys = 4
    constraints = [[0, 1, 2, 3]]

    l2 = [create_startline(n_phys, constraints)]
    p2 = Page(start_lines=l2, parameter_names=['a', 'b', 'g'] + ['Z%d' % i for i in range(6)])
    p1.print()
    p2.print()
    print('---------')

    t0 = time()
    p2.push_all_x_to_left_and_simplify()
    t_simplify = time() - t0

    p3 = Page(start_lines=l2, parameter_names=['a', 'b', 'g'] + ['Z%d' % i for i in range(6)])
    t0 = time()
    p3.push_all_x_to_left_smart()
    t_simplify_smart = time() - t0

    # p.print()
    # test = p.get_lambda_function_from_line(1)

    # from qaoa_analytics.sympy_qaoa_evaluator import get_mathematica_strings_from_page

    # all_m_strs = get_mathematica_strings_from_page(p)
    #
    # all_m_as_sum = '+'.join(all_m_strs)
    # print(all_m_as_sum)
