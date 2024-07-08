import sys

import pygad
import numpy as np
import pyrallis
from dataclasses import dataclass

import postfix_program

@dataclass
class Config:
    num_individuals: int = 100
    num_genes: int = 10

    num_generations: int = 200
    num_parents_mating: int = 30
    keep_parents: int = 10
    mutation_percent_genes: int = 10

class ProgramOptimizer:
    def __init__(self, config: Config):
        # Create the initial population
        stopping_program = [-1.0] * config.num_genes

        self.config = config
        self.initial_population = [np.array(stopping_program) for i in range(config.num_individuals)]

    def fit(self, states, actions):
        """ states is a batch of states, shape (N, state_shape)
            actions is a batch of actions, shape (N, action_shape), we assume continuous actions
        """
        self.best_solution = None
        self.best_fitness = None

        def fitness_func(ga_instance, solution, solution_idx):
            batch_size = states.shape[0]
            action_size = actions.shape[1]
            sum_error = 0.0

            for index in range(batch_size):
                action = postfix_program.run_program(solution, states[index])
                action = np.array(action + [0.0] * action_size)
                action = action[:action_size]
                desired_action = actions[index]

                print(states[index], desired_action, action)

                sum_error += np.mean((action - desired_action) ** 2)

            fitness = -(sum_error / batch_size)

            if self.best_fitness is None or fitness > self.best_fitness:
                self.best_solution = solution
                self.best_fitness = fitness

            print('F', fitness, file=sys.stderr)
            return fitness

        ga_instance = pygad.GA(num_generations=self.config.num_generations,
            num_parents_mating=self.config.num_parents_mating,
            fitness_func=fitness_func,
            initial_population=self.initial_population,
            parent_selection_type="sss",
            keep_parents=self.config.keep_parents,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=self.config.mutation_percent_genes,
            random_mutation_max_val=100.0,
            random_mutation_min_val=-100.0
        )

        print(ga_instance.run())

        # Allow the population to survive
        self.initial_population = ga_instance.population

        # Print the best individual
        print(postfix_program.run_program(self.best_solution, states[0], do_print=True))

@pyrallis.wrap()
def main(config: Config):
    optim = ProgramOptimizer(config)

    # Sample states and actions
    states = np.array([
        [1.0],
        [2.0],
        [-5.0],
        [10.0],
    ])
    actions = states * 2

    # Fit
    optim.fit(states, actions)

if __name__ == '__main__':
    main()
