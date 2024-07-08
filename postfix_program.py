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

class Operator:
    def __init__(self, name, num_operands, function):
        self.name = name
        self.num_operands = num_operands
        self.function = function

    def __str__(self):
        return self.name

OPERATORS = [
    Operator('<end>', 0, None),
    Operator('<',     2, lambda a, b: float(a < b)),
    Operator('>',     2, lambda a, b: float(a > b)),
    Operator('==',    2, lambda a, b: float(a == b)),
    Operator('!=',    2, lambda a, b: float(a != b)),
    Operator('+',     2, lambda a, b: a + b),
    Operator('-',     2, lambda a, b: a - b),
    Operator('*',     2, lambda a, b: a * b),
    Operator('/',     2, lambda a, b: a / b),
    Operator('%',     2, lambda a, b: a % b),
    Operator('max',   2, lambda a, b: max(a, b)),
    Operator('min',   2, lambda a, b: min(a, b)),
    Operator('trunc', 1, lambda a: float(int(a))),
    Operator('abs',   1, lambda a: abs(a)),
    Operator('sin',   1, lambda a: math.sin(a)),
    Operator('cos',   1, lambda a: math.cos(a)),
    Operator('exp',   1, lambda a: math.exp(a)),
    Operator('sqrt',  1, lambda a: math.sqrt(a)),
    Operator('?',     3, lambda cond, a, b: a if cond > 0.5 else b),
]
NUM_OPERATORS = len(OPERATORS)

def run_program(array, inp):
    stack = []

    for value in array:
        if value >= 0.0:
            # Literal, push it
            stack.append(value)
            continue

        if value < -NUM_OPERATORS:
            # Input variable
            input_index = -int(value) - NUM_OPERATORS - 1

            # Silently ignore input variables beyond the end of inp
            if input_index < len(inp):
                stack.append(inp[input_index])

            continue

        # Operators
        operator_index = -int(value) - 1
        operator = OPERATORS[operator_index]

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

    return stack

if __name__ == '__main__':
    run_program([2.0, 1.0, -6.0, -7.0, -1.0, -1.0], [3.14, 6.28])
