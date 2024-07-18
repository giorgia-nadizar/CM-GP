# x 2 + 2 * <end> <end>
#
# Literals         # positive
# Operators        # negative, we have a finite number of them
# Input variables  # negative, we can have many of them
# <end>            # OPERATOR_END
#
# 1. PyGAD produces numpy arrays (lists of floats). Look at them in pairs of (mean, variance).
#    sample a value from that normal distribution, and transform the sample to one
#    of the tokens listed above
# 2. Run that
#
# Format: genes are floats. We
import math
import numpy as np

class Operator:
    def __init__(self, name, num_operands, function):
        self.name = name
        self.num_operands = num_operands
        self.function = function

    def __str__(self):
        return self.name


OPERATORS = [
    Operator('abs', 1, lambda a: abs(a)),
    Operator('sin', 1, lambda a: math.sin(a)),
    Operator('cos', 1, lambda a: math.cos(a)),
    Operator('exp', 1, lambda a: math.exp(min(a, 10.0))),
    Operator('sqrt', 1, lambda a: math.sqrt(max(a, 0.0))),
    Operator('neg', 1, lambda a: -a),
    Operator('+', 2, lambda a, b: a + b),
    Operator('-', 2, lambda a, b: a - b),
    Operator('*', 2, lambda a, b: a * b),
    Operator('/', 2, lambda a, b: a / (1.0 if b == 0.0 else b)),
    Operator('%', 2, lambda a, b: a % (1.0 if b == 0.0 else b)),
    Operator('max', 2, lambda a, b: max(a, b)),
    Operator('min', 2, lambda a, b: min(a, b)),
    Operator('trunc', 1, lambda a: float(int(a))),
    Operator('<', 2, lambda a, b: float(a < b)),
    Operator('>', 2, lambda a, b: float(a > b)),
    Operator('==', 2, lambda a, b: float(a == b)),
    Operator('!=', 2, lambda a, b: float(a != b)),
    Operator('?', 3, lambda cond, a, b: a if cond > 0.5 else b),
    Operator('<end>', 0, None),
]
NUM_OPERATORS = len(OPERATORS)


class Program:
    def __init__(self, genome):
        self.genome = genome

    def __str__(self):
        return repr(self.run_program(inp=[1], do_print=True))

    def __call__(self, inp):
        res = self.run_program(inp, do_print=False)

        if len(res) == 0:
            return 0.0
        else:
            return res[-1]

    def run_program(self, inp, do_print=False):
        stack = []

        for pointer in range(0, len(self.genome), 2):
            # Sample the actual token to execute
            mean = self.genome[pointer + 0]
            log_std = self.genome[pointer + 1]

            if log_std > 10.0:
                log_std = 10.0      # Prevent exp() from overflowing

            value = np.random.normal(loc=mean, scale=math.exp(log_std))

            # Execute the token
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
                if do_print:
                    stack.append(f'x{input_index}')
                else:
                    if input_index < len(inp):
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
    print(Program([5.0, 1.0, -21.0, -2.0]).run_program([3.14, 6.28], do_print=True))
    print(Program([-17.0, 0.0]).run_program([0.0], do_print=False))
