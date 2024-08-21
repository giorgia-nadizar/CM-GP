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
    Operator('-abs', 1, lambda a: -abs(a)),
    Operator('sin', 1, lambda a: math.sin(a)),
    Operator('-sin', 1, lambda a: -math.sin(a)),
    Operator('cos', 1, lambda a: math.cos(a)),
    Operator('-cos', 1, lambda a: -math.cos(a)),
    Operator('exp', 1, lambda a: math.exp(min(a, 10.0))),
    Operator('-exp', 1, lambda a: -math.exp(min(a, 10.0))),
    Operator('sqrt', 1, lambda a: math.sqrt(a) if a >= 0.0 else 0.0),
    Operator('-sqrt', 1, lambda a: -math.sqrt(a) if a >= 0.0 else 0.0),
    Operator('neg', 1, lambda a: -a),
    Operator('+', 2, lambda a, b: a + b),
    Operator('-', 2, lambda a, b: a - b),
    Operator('*', 2, lambda a, b: a * b),
    Operator('/', 2, lambda a, b: a / b if abs(b) > 0.01 else 0.0),
    Operator('%', 2, lambda a, b: a % b if abs(b) > 0.01 else 0.0),
    Operator('max', 2, lambda a, b: max(a, b)),
    Operator('min', 2, lambda a, b: min(a, b)),
    Operator('trunc', 1, lambda a: float(int(a))),
    Operator('ifsmaller', 4, lambda a, b, iftrue, iffalse: iftrue if a < b else iffalse),
]
NUM_OPERATORS = len(OPERATORS)

class InvalidProgramException(Exception):
    pass

class Program:
    def __init__(self, genome, state_dim):
        self.tokens = genome
        self.state_dim = state_dim

    def to_string(self):
        def on_literal_func(stack, token):
            stack.append(f"±{token}")

        def on_input_func(stack, input_index):
            stack.append(f"x[{input_index}]")

        def on_operator_func(stack, operator, operands):
            # Put a string representation of the operator on the stack
            if len(operands) == 1:
                result = f"{operator.name}({operands[0]})"
            elif operator.name in ['min', 'max']:
                # two-operand operator that is a function call
                result = f"{operator.name}({operands[0]}, {operands[1]})"
            elif len(operands) == 2:
                result = f"({operands[0]} {operator.name} {operands[1]})"
            elif len(operands) == 4:
                result = f"({operands[0]} < {operands[1]} ? {operands[2]} : {operands[3]})"

            stack.append(result)

        return self._visit_program(
            on_literal_func=on_literal_func,
            on_input_func=on_input_func,
            on_operator_func=on_operator_func
        )

    def __call__(self, inp):
        def on_literal_func(stack, token):
            # Random sign. The program needs to wrap the literal in abs() or -abs() to set its sign
            if np.random.random() < 0.5:
                token = -token

            stack.append(token)

        def on_input_func(stack, input_index):
            stack.append(inp[input_index])

        def on_operator_func(stack, operator, operands):
            result = operator.function(*operands)
            stack.append(result)

        return self._visit_program(
            on_literal_func=on_literal_func,
            on_input_func=on_input_func,
            on_operator_func=on_operator_func
        )

    def num_inputs_looked_at(self, state_vars):
        def on_literal_func(stack, token):
            stack.append(set([]))   # Literals don't look at inputs

        def on_input_func(stack, input_index):
            stack.append(set([input_index]))    # Inputs look at inputs

        def on_operator_func(stack, operator, operands):
            looked_at = set([])

            for operand in operands:
                looked_at.update(operand)       # Operands may look at inputs

            stack.append(looked_at)

        return len(self._visit_program(
            on_literal_func=on_literal_func,
            on_input_func=on_input_func,
            on_operator_func=on_operator_func
        ))

    def _visit_program(self, on_literal_func, on_input_func, on_operator_func):
        stack = []

        for token in self.tokens:
            if token >= 0.0:
                on_literal_func(stack, token)
                continue

            # Now, cast token to an int, but with stochasticity so that a value
            # close to x.5 is always cast to x, but other values may end up on x+1 or x-1
            token = int(token + 0.498 * (np.random.random() - 0.5))

            # Input variable
            if token < -NUM_OPERATORS:
                input_index = -token - NUM_OPERATORS - 1

                if input_index >= self.state_dim:
                    raise InvalidProgramException()

                on_input_func(stack, input_index)
                continue

            # Operators
            operator_index = -token - 1
            operator = OPERATORS[operator_index]

            # Pop the operands
            operands = []

            for index in range(operator.num_operands):
                if len(stack) == 0:
                    raise InvalidProgramException()

                operands.append(stack.pop())

            on_operator_func(stack, operator, operands)

        if len(stack) == 0:
            raise InvalidProgramException()

        return stack[-1]

if __name__ == '__main__':
    # Compute the average output of programs
    values = []

    for l in range(20):
        for i in range(100000):
            dna = np.random.random((l,))
            dna[0:-1:2] *= -(NUM_OPERATORS + 1)                 # Tokens between -NUM_OPERATORS - state_dim and 0
            p = Program(dna)
            values.append(p([]))

        print('Average output of random programs of size', l, ':', np.mean(values), '+-', np.std(values))
