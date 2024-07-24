import sys

import pygad
import numpy as np
import pyrallis
from dataclasses import dataclass

from postfix_program import Program, NUM_OPERATORS

class ProgramOptimizer:
    def __init__(self, config, state_dim):

        # Create the initial population
        # We create it so these random programs try all the operators and read all the state variables
        self.initial_population = np.random.random((config.num_individuals, config.num_genes * 2))  # Random numbers between 0 and 1
        self.initial_population *= NUM_OPERATORS + state_dim                    # Between 0 and NUM_OPERATORS + state_dim
        self.initial_population *= -1.0                                         # Between -NUM_OPERATORS -state_dim and 0

        self.best_solution = self.initial_population[0]
        self.best_fitness = None

        self.config = config

    def get_action(self, state):
        program = Program(genome=self.best_solution)
        return program(state)

    def get_best_solution_str(self):
        program = Program(genome=self.best_solution)
        return str(program)

    def _fitness_func(self, ga_instance, solution, solution_idx):
        batch_size = self.states.shape[0]
        sum_error = 0.0
        sum_lookedat = 0.0

        # Evaluate the program several times, because evaluations are stochastic
        for eval_run in range(self.config.num_eval_runs):
            for index in range(batch_size):
                # Create the Program here to sample the tokens for every eval run and every index
                program = Program(genome=solution)

                # MSE for the loss
                action = program(self.states[index])
                desired_action = self.actions[index]

                sum_error += np.mean((action - desired_action) ** 2)

                # Num input variables looked at
                sum_lookedat += program.num_inputs_looked_at(self.states[index])

        avg_error = (sum_error / (batch_size * self.config.num_eval_runs))
        avg_lookedat = (sum_lookedat / (batch_size * self.config.num_eval_runs))

        fitness = -avg_error / (avg_lookedat + 0.01) # FIXME: random equation

        return fitness

    def fit(self, states, actions):
        """ states is a batch of states, shape (N, state_shape)
            actions is a batch of actions, shape (N,), we assume continuous actions

            NOTE: One ProgramOptimizer has to be used for each action dimension
        """
        self.states = states        # picklable self._fitness_func needs these instance variables
        self.actions = actions

        self.ga_instance = pygad.GA(
            fitness_func=self._fitness_func,
            initial_population=self.initial_population,
            num_generations=self.config.num_generations,
            num_parents_mating=self.config.num_parents_mating,
            keep_parents=self.config.keep_parents,
            mutation_percent_genes=self.config.mutation_percent_genes,

            # Work with non-deterministic objective functions
            keep_elitism=0,
            save_solutions=False,
            save_best_solutions=False,

            parent_selection_type="sss",
            crossover_type="single_point",
            mutation_type="random",
            random_mutation_max_val=10,
            random_mutation_min_val=-10,
            parallel_processing=["process", None]
        )

        self.ga_instance.run()

        # Allow the population to survive
        self.initial_population = self.ga_instance.population

        # Best solution for now
        self.best_solution = self.ga_instance.best_solution()[0]
