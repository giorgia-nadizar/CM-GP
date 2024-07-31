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

def from_bool(x):
    return 1.0 if x else -1.0

OPERATORS = [
    Operator('abs', 1, lambda a: abs(a)),
    Operator('-abs', 1, lambda a: -abs(a)),
    Operator('sin', 1, lambda a: math.sin(a)),
    Operator('-sin', 1, lambda a: -math.sin(a)),
    Operator('cos', 1, lambda a: math.cos(a)),
    Operator('-cos', 1, lambda a: -math.cos(a)),
    Operator('exp', 1, lambda a: math.exp(min(a, 10.0))),
    Operator('-exp', 1, lambda a: -math.exp(min(a, 10.0))),
    Operator('sqrt', 1, lambda a: math.sqrt(max(a, 0.0))),
    Operator('-sqrt', 1, lambda a: -math.sqrt(max(a, 0.0))),
    Operator('neg', 1, lambda a: -a),
    Operator('+', 2, lambda a, b: a + b),
    Operator('-', 2, lambda a, b: a - b),
    Operator('*', 2, lambda a, b: a * b),
    Operator('/', 2, lambda a, b: a / b if abs(b) > 0.01 else np.random.normal()),
    Operator('%', 2, lambda a, b: a % b if abs(b) > 0.01 else np.random.normal()),
    Operator('max', 2, lambda a, b: max(a, b)),
    Operator('min', 2, lambda a, b: min(a, b)),
    Operator('trunc', 1, lambda a: float(int(a))),
    Operator('<', 2, lambda a, b: from_bool(a < b)),
    Operator('>', 2, lambda a, b: from_bool(a > b)),
    Operator('?', 3, lambda cond, a, b: a if cond > 0.0 else b),
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

    def to_string(self, inp):
        return repr(self.run_program(inp=inp, do_print=True))

    def __call__(self, inp):
        return self.run_program(inp, do_print=False)

    def num_inputs_looked_at(self, inp):
        lookedat = set(FIND_X_REGEX.findall(self.to_string(inp)))     # Find x'es in the representation of this program. Those are state variables actually looked at
        return len(lookedat)

    def run_program(self, inp, do_print=False):
        stack = []
        functions = {operator.name: operator.function for operator in OPERATORS}

        for token in self.tokens:
            # Literal, push it with a random sign. The program has to use the make_pos() and make_neg() operators to fix the sign
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
                        stack.append('randn()')
                    else:
                        stack.append(np.random.normal())

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
                    operand = np.random.normal()
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
            if do_print:
                stack.append('randn()')
            else:
                stack.append(np.random.normal())

        return stack[-1]

if __name__ == '__main__':
    # Compute the average output of programs
    values = []

    for l in range(20):
        for i in range(100000):
            dna = np.random.random((l*2,))
            dna[0:-1:2] *= -(NUM_OPERATORS + 1)                 # Tokens between -NUM_OPERATORS - state_dim and 0
            dna[1:-1:2] *= 3.0                                  # Log_std between 0 and 3
            p = Program(dna)
            values.append(p([]))

        print('Average output of random programs of size', l, ':', np.mean(values), '+=', np.std(values))
