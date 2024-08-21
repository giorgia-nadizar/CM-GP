import sys

import pygad
import numpy as np
import pyrallis
from dataclasses import dataclass

from postfix_program import Program, NUM_OPERATORS, InvalidProgramException

class ProgramOptimizer:
    def __init__(self, config, state_dim):

        # Create the initial population
        # We create it so these random programs try all the operators and read all the state variables
        self.initial_population = np.random.random((config.num_individuals, config.num_genes))  # Random numbers between 0 and 1
        self.initial_population *= -(NUM_OPERATORS + state_dim)         # Tokens between -NUM_OPERATORS - state_dim and 0

        self.best_solution = self.initial_population[0]
        self.best_fitness = None

        self.config = config
        self.state_dim = state_dim

    def get_action(self, state):
        program = Program(self.best_solution, self.state_dim)

        try:
            return program(state)
        except InvalidProgramException:
            return np.random.normal()

    def get_best_solution_str(self):
        program = Program(self.best_solution, self.state_dim)

        try:
            return program.to_string()
        except InvalidProgramException:
            return '<invalid program>'

    def _fitness_func(self, ga_instance, solution, solution_idx):
        program = Program(solution, self.state_dim)

        try:
            # Num input variables looked at
            expected_lookedat = self.states.shape[1]
            lookedat = 0.0

            for i in range(100):
                # This is a stochastic process
                lookedat += program.num_inputs_looked_at(expected_lookedat)

            looked_proportion = (lookedat / 100) / expected_lookedat

            # Evaluate the program several times, because evaluations are stochastic
            batch_size = self.states.shape[0]
            sum_error = 0.0

            for eval_run in range(self.config.num_eval_runs):
                for index in range(batch_size):
                    # MSE for the loss
                    action = program(self.states[index])
                    desired_action = self.actions[index]

                    sum_error += np.mean((action - desired_action) ** 2)

            avg_error = (sum_error / (batch_size * self.config.num_eval_runs))
            fitness = (1.0 - avg_error) * looked_proportion
        except InvalidProgramException:
            fitness = -1000.0

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
            mutation_probability=self.config.mutation_probability,

            # Work with non-deterministic objective functions
            keep_elitism=0,
            save_solutions=False,
            save_best_solutions=False,
            random_mutation_min_val=-10,
            random_mutation_max_val=10,

            parent_selection_type="rank",
            crossover_type="single_point",
            mutation_type="random",
            parallel_processing=["process", None]
        )

        self.ga_instance.run()

        # Allow the population to survive
        self.initial_population = self.ga_instance.population

        # Best solution for now
        self.best_solution = self.ga_instance.best_solution()[0]
