# x 2 + 2 * <end> <end>
#
# Literals         # positive
# Operators        # negative, we have a finite number of them
# Input variables  # negative, we can have many of them
# <end>            # OPERATOR_END
#
# 1. PyGAD produces numpy arrays (lists of floats). Look at them in pairs of (mean, variance).
#    sample a token from that normal distribution, and transform the sample to one
#    of the tokens listed above
# 2. Run that
#
# Format: genes are floats. We
import re
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
FIND_X_REGEX = re.compile('\[(\d+)\]')

class Program:
    def __init__(self, genome):
        self.tokens = []

        for pointer in range(0, len(genome), 2):
            # Sample the actual token to execute
            mean = genome[pointer + 0]
            log_std = genome[pointer + 1]

            if log_std > 10.0:
                log_std = 10.0      # Prevent exp() from overflowing

            token = np.random.normal(loc=mean, scale=math.exp(log_std))
            self.tokens.append(token)

    def __str__(self):
        return repr(self.run_program(inp=[1], do_print=True))

    def __call__(self, inp):
        return self.run_program(inp, do_print=False)

    def num_inputs_looked_at(self, inp):
        lookedat = set(FIND_X_REGEX.findall(str(self)))     # Find x'es in the representation of this program. Those are state variables actually looked at
        return len(lookedat)

    def run_program(self, inp, do_print=False):
        stack = []
        functions = {operator.name: operator.function for operator in OPERATORS}

        for token in self.tokens:
            # Literal, push it
            if token >= 0.0:
                if do_print:
                    stack.append(str(token))
                else:
                    stack.append(token)

                continue

            token = int(token)

            if token < -NUM_OPERATORS:
                # Input variable
                input_index = -token - NUM_OPERATORS - 1

                # Silently ignore input variables beyond the end of inp
                if input_index < len(inp):
                    if do_print:
                        stack.append(f'x[{input_index}]')
                    else:
                        stack.append(inp[input_index])
                else:
                    if do_print:
                        stack.append('1.0')
                    else:
                        stack.append(1.0)

                continue

            # Operators
            operator_index = -token - 1
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

            if do_print:
                # Put a string representation of the operator on the stack
                if len(operands) == 1:
                    result = f"{operator.name}({operands[0]})"
                elif operator.name in ['min', 'max']:
                    # two-operand operator that is a function call
                    result = f"{operator.name}({operands[0]}, {operands[1]})"
                elif len(operands) == 2:
                    result = f"({operands[0]} {operator.name} {operands[1]})"
                elif len(operands) == 3:
                    result = f"({operands[0]} ? {operands[1]} : {operands[2]})"

                # Simple constant propagation: if the resulting expression can be eval'd,
                # it means that it only uses operators and constants, so we can simply
                # show the program as the constant
                try:
                    result = str(eval(result, functions))
                except:
                    pass

                stack.append(result)
            else:
                # Run the operator and get the result back
                result = operator.function(*operands)
                stack.append(result)

        if len(stack) == 0:
            return 0.0
        else:
            return stack[-1]

if __name__ == '__main__':
    print(Program([5.0, 1.0, -2.0, -5.0, 18.0, 0.0, -8.0, -2.0]))
    print(Program([-17.0, 0.0]).run_program([0.0], do_print=False))
