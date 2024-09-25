"""
Created on Fri May 25 11:57:05 2018

@author: kili
"""
from abc import ABC, abstractmethod
import numpy as np


class ProgramInterface(ABC):
    """
    Interface which all programs have to derive from to work with the Qaoa base class
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __setitem__(self, index, value):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def zipped_program(self):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    def __repr__(self):
        strr = 'program:' + ''.join(['       '[0:(8-len(letter))] + letter for letter in self.program_string])
        strr += '\nparameters: ' + ', '.join(['%+.3f' % p for p in self.get_parameters()])
        return strr


class SimpleProgram(ProgramInterface):
    """
    Simple Program
    """

    def __init__(self, program_string='', parameters=None):
        self.program_string = program_string
        self.linearparameters = [0] * len(program_string)

        if parameters is None:
            self.set_all_parameters(0)
        else:
            assert len(program_string) == len(parameters), 'wrong number of parameters'
            self.linearparameters = parameters.copy()

    def __getitem__(self, index):
        return self.linearparameters[index]

    def __setitem__(self, index, value):
        self.linearparameters[index] = value

    def __len__(self):
        return len(self.linearparameters)

    def set_all_parameters(self, val):
        self.linearparameters = [val] * len(self)

    def get_parameters(self):
        return self.linearparameters

    def zipped_program(self):
        return zip(self.program_string, self.get_parameters())


class CombinedLetterProgram(SimpleProgram):
    """ same as simple program, angles for given letters can be kept the same"""
    def __init__(self, programstring, combined_letters=None, parameters=None):
        """

        :param programstring:
        :param combined_letters:
        :param parameters:
        """
        self.program_string = programstring
        self.combined_letters = []
        if combined_letters is not None:
            for le in combined_letters:
                if le in self.program_string and le:  # checking if its not ''
                    self.combined_letters += [le]

        self.reduced_program_string = programstring
        for letter in self.combined_letters:
            self.reduced_program_string = self.reduced_program_string.replace(letter, '')
        self.n_independent_paramters = len(self.reduced_program_string)+len(self.combined_letters)
        self.linearparameters = [0] * self.n_independent_paramters

        if parameters is None:
            self.set_all_parameters(0)
        else:
            assert self.n_independent_paramters == len(parameters), 'wrong number of parameters'
            self.linearparameters = parameters.copy()

    def get_parameters(self):
        parameters = [0] * len(self.program_string)

        count = len(self.combined_letters)
        for il, letter in enumerate(self.program_string):
            if letter in self.combined_letters:
                parameters[il] = self[self.combined_letters.index(letter)]
            else:
                parameters[il] = self[count]
                count += 1

        return parameters

    def __len__(self):
        return self.n_independent_paramters


class CommutatorProgram:
    # TODO
    def __init__(self, program_string, parameters=None):
        import re
        self.program_string = program_string

        # Extract combined sequences and single letters
        self.combined_sequences = re.findall(r'\[.*?\]', program_string)
        self.single_letters = [ch for ch in re.sub(r'\[.*?\]', '', program_string) if ch.isalpha()]

        # Unique units are either a combined sequence or a single letter
        self.unique_units = self.single_letters + self.combined_sequences

        self.n_independent_parameters = len(self.unique_units)
        self.parameters = [0] * self.n_independent_parameters if parameters is None else parameters.copy()

        assert len(self.parameters) == self.n_independent_parameters, 'wrong number of parameters'

    def get_parameters(self):
        return self.parameters

    def zipped_program(self):
        return zip(self.program_string, self.get_parameters())

    def __len__(self):
        return self.n_independent_parameters


class ConnectedLettersProgram(ProgramInterface):
    def __init__(self, programstring, connected_letter_dict, parameters=None):
        self.original_program_string = programstring
        self.replaced_program_string = programstring

        self.original_letter_dict = connected_letter_dict.copy()

        self.linearparameters = [0] * len(programstring)

        def _replacement_iterator():
            n = 97
            while True:
                yield chr(n)
                n += 1
        repl_it = _replacement_iterator()

        temp_connected_letter_dict = {}
        for k, v in connected_letter_dict.items():
            repl = next(repl_it)
            while repl in self.replaced_program_string or repl in v or repl in connected_letter_dict.keys():
                repl = next(repl_it)

            self.replaced_program_string = self.replaced_program_string.replace(k, repl)
            temp_connected_letter_dict[repl] = v

        self.connected_letters = temp_connected_letter_dict

        if parameters is None:
            self.set_all_parameters(0)
        else:
            assert len(self) == len(parameters), 'wrong number of parameters'
            self.linearparameters = parameters.copy()

    def __getitem__(self, index):
        return self.linearparameters[index]

    def __setitem__(self, index, value):
        self.linearparameters[index] = value

    def __len__(self):
        return len(self.linearparameters)

    def set_all_parameters(self, val):
        self.linearparameters = [val] * len(self)

    def get_parameters(self):
        tempparams = []

        for i in range(len(self.replaced_program_string)):
            letter = self.replaced_program_string[i]
            if letter in self.connected_letters.keys():
                tempparams += [self.linearparameters[i]] * len(self.connected_letters[letter])
            else:
                tempparams += [self.linearparameters[i]]
        return tempparams

    @property
    def program_string(self):
        str = self.replaced_program_string
        for k, v in self.connected_letters.items():
            str = str.replace(k, v)

        return str

    def zipped_program(self):
        return zip(self.program_string, self.get_parameters())


class qaoa_program(SimpleProgram):
    """
    lowercase letter is for programs that have parameters for each single atom
    uppercase for global operations
    """
    def __init__(self, program_string, n, parameters=None):
        self.program_string = program_string
        self.N = n
        
        self.nGlobal = 0
        self.nSingle = 0
        
        self.piIsGlobal = []
        
        # include checks so only x,y,z can be single
        # maybe include single constraints?
        for c in program_string:
            if c.isupper():
                self.nGlobal += 1
                self.piIsGlobal.append(True)
            else:
                self.nSingle += 1
                self.piIsGlobal.append(False)

        self.length = self.nGlobal + self.N*self.nSingle
        
        if parameters is None:
            parameters = [0]*self.length
        
        assert self.length == len(parameters), 'wrong number of parameters'
        self.linearparameters = parameters.copy()
        self.parameterIndizes = []
        
        ind = 0
        for c in program_string:
            if c.isupper():
                self.parameterIndizes.append([ind])
                ind += 1
            else:
                temp = []
                for i in range(self.N):
                    temp.append(ind)
                    ind += 1
                self.parameterIndizes.append(temp)
        
    def __len__(self):
        return self.length

    def __repr__(self):
        # would not work with the inherited function
        pass
        
    def zipped_program(self):
        temp = []
        
        for inds in self.parameterIndizes:
            t = []
            for ind in inds:
                t.append(self.linearparameters[ind])
            temp.append(t)
        
        return zip(self.program_string, self.piIsGlobal, temp)


def get_startparameters(programstring, start=0, stop=1, p_type='adiabatic'):
    p_length = len(programstring)

    if p_type == 'random':
        return np.random.random(p_length)

    elif p_type == 'adiabatic':
        p_part = np.linspace(start, stop, p_length)
        x_part = np.linspace(stop, start, p_length)

        return [x_part[i] if letter == 'X' else p_part[i] for i, letter in enumerate(programstring)]

    else:
        return np.zeros(p_length)


if __name__ == '__main__':
    pstr = 'ZXCT'
    test = ConnectedLettersProgram(pstr, {'a': 'aaT', 'T': 'AB'})
    test.linearparameters = list(range(len(pstr)))
    print(test)
