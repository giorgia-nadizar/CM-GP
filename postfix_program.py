# x 2 + 2 * <end> <end>
#
# Literals         # positive
# Operators        # negative, we have a finite number of them
# Input variables  # negative, we can have many of them
# <end>            # OPERATOR_END
#
# 1. PyGAD produces numpy arrays (lists of floats), turn them into <see above>
# 2. Run that
import math

import numpy as np

import torch as th


class Operator:
    def __init__(self, name, num_operands, function):
        self.name = name
        self.num_operands = num_operands
        self.function = function

    def __str__(self):
        return self.name


OPERATORS = [
    Operator('<end>', 0, None),
    Operator('<', 2, lambda a, b: float(a < b)),
    Operator('>', 2, lambda a, b: float(a > b)),
    Operator('==', 2, lambda a, b: float(a == b)),
    Operator('!=', 2, lambda a, b: float(a != b)),
    Operator('+', 2, lambda a, b: a + b),
    Operator('-', 2, lambda a, b: a - b),
    Operator('*', 2, lambda a, b: a * b),
    Operator('/', 2, lambda a, b: a / (1.0 if b == 0.0 else b)),
    Operator('%', 2, lambda a, b: a % (1.0 if b == 0.0 else b)),
    Operator('max', 2, lambda a, b: max(a, b)),
    Operator('min', 2, lambda a, b: min(a, b)),
    Operator('trunc', 1, lambda a: float(int(a))),
    Operator('abs', 1, lambda a: abs(a)),
    Operator('neg', 1, lambda a: -a),
    Operator('sin', 1, lambda a: math.sin(a)),
    Operator('cos', 1, lambda a: math.cos(a)),
    Operator('exp', 1, lambda a: math.exp(min(a, 10.0))),
    Operator('sqrt', 1, lambda a: math.sqrt(max(a, 0.0))),
    Operator('?', 3, lambda cond, a, b: a if cond > 0.5 else b),
]
NUM_OPERATORS = len(OPERATORS)


class Program:
    def __init__(self, genome=None, size=None):
        self.size = size
        if genome is not None:
            self.genome = genome
            self.size = len(genome)
        else:
            assert size is not None, "If genome is not specified, size must be given"
            self.genome = np.ones(size)

    def __str__(self):
        return f'{self.run_program(inp=[1], do_print=True)}'

    def __call__(self, inp, len_output=None, do_print=False):

        res = self.run_program(inp, do_print=do_print)

        # If the desired output length is given, pad the result with zeroes if needed
        if len_output:
            res = np.array(res + [0.0] * len_output)
            res = res[:len_output]

        if do_print:
            return res
        else:
            return np.array(res)

    def run_program(self, inp, do_print=False):
        stack = []

        for value in self.genome:
            if value >= 0.0:
                # Literal, push it
                if do_print:
                    stack.append(str(value))
                else:
                    stack.append(value)

                continue

            value = int(value)

            if value < -NUM_OPERATORS:
                # Input variable
                input_index = -value - NUM_OPERATORS - 1

                # Silently ignore input variables beyond the end of inp
                if input_index < len(inp):
                    if do_print:
                        stack.append(f'x{input_index}')
                    else:
                        stack.append(inp[input_index])

                continue

            # Operators
            operator_index = -value - 1
            operator = OPERATORS[operator_index]

            if do_print:
                stack.append(operator.name)
            else:
                if operator.function is None:
                    # End of program
                    break

                # Pop the operands
                operands = []

                for index in range(operator.num_operands):
                    if len(stack) == 0:
                        # If the stack is empty, "pop" a 1 from it.
                        # 1 is neutral to mul, can be used for div, does something to add and sub
                        operand = 1.0
                    else:
                        operand = stack.pop()

                    operands.append(operand)

                # Run the operator and get the result back
                result = operator.function(*operands)
                stack.append(result)

        if do_print:
            return ' '.join(stack)
        else:
            return stack


if __name__ == '__main__':
    print(Program([2.0, -21.0, -6.0, -1.0, -1.0])([3.14, 6.28]))
    print(Program([-21, -7.0, -6.0, -22.0, 0.0, 0.0, -1.0, -1.0])([1, 8]))
    print(Program([5.0, -21.0, -6.0, -1.0, -1.0])([3.14, 6.28], do_print=True))
    print(Program([-2.538086, 0.334053, -17.267960, -18.188475, -18.504102, -6.522001, -6.147776, -16.242687, -1.000000, -4.448039])([1.0, 1.0], do_print=True))
