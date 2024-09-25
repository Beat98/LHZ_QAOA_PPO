import unittest
from qaoa.programs import SimpleProgram, CombinedLetterProgram, get_startparameters


class TestPrograms(unittest.TestCase):
    def test_simple_program(self):
        pass

    def test_combined_letters_program(self):
        p1 = CombinedLetterProgram('ZXCX', combined_letters=['X'])
        p1[0] = 2
        p1[1] = 1
        self.assertEqual(p1.linearparameters, [2, 1, 0])
        self.assertEqual(list(p1.zipped_program()), [('Z', 1), ('X', 2), ('C', 0), ('X', 2)])

        p2 = CombinedLetterProgram('ZXCX')
        p2[0] = 2
        p2[1] = 1
        self.assertEqual(p2.linearparameters, [2, 1, 0, 0])
        self.assertEqual(list(p2.zipped_program()), [('Z', 2), ('X', 1), ('C', 0), ('X', 0)])

        p3 = CombinedLetterProgram('XXXX', combined_letters=['X'])
        p3[0] = 2
        self.assertEqual(p3.linearparameters, [2])
        self.assertEqual(list(p3.zipped_program()), [('X', 2), ('X', 2), ('X', 2), ('X', 2)])

        p4 = CombinedLetterProgram('XZZXC', combined_letters=['X', 'Z'])
        p4[0] = 2
        p4[1] = 1
        p4[2] = -0.5
        self.assertEqual(p4.linearparameters, [2, 1, -0.5])
        self.assertEqual(list(p4.zipped_program()), [('X', 2), ('Z', 1), ('Z', 1), ('X', 2), ('C', -0.5)])

        p5 = CombinedLetterProgram('XZZXC', combined_letters=['Z', 'X'])
        p5[0] = 2
        p5[1] = 1
        p5[2] = -0.5
        self.assertEqual(p5.linearparameters, [2, 1, -0.5])
        self.assertEqual(list(p5.zipped_program()), [('X', 1), ('Z', 2), ('Z', 2), ('X', 1), ('C', -0.5)])

        p6 = CombinedLetterProgram('ZXC', combined_letters=['T', 'U', 'X'])
        p6[0] = 2
        p6[1] = 1
        self.assertEqual(p6.linearparameters, p1.linearparameters)

        p7 = CombinedLetterProgram('ZXC', combined_letters=['T', 'U', 'x', ''])
        p7[0] = 2
        p7[1] = 1
        self.assertEqual(p6.linearparameters, p7.linearparameters)
        self.assertNotEqual(list(p6.zipped_program()), list(p7.zipped_program()))

    def test_get_startparameters(self):
        pass


if __name__ == '__main__':
    unittest.main()
