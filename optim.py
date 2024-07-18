import sys

import pygad
import numpy as np
import pyrallis
from dataclasses import dataclass

from postfix_program import Program, NUM_OPERATORS

class ProgramOptimizer:
    def __init__(self, config):

        # Create the initial population
        self.initial_program = [0.0] * (config.num_genes * 2)  # Mean and log_std for each gene

        self.best_solution = self.initial_program
        self.best_fitness = None

        self.config = config
        self.initial_population = [np.array(self.initial_program) for i in range(config.num_individuals)]

    def get_action(self, state):
        program = Program(genome=self.best_solution)
        return program(state)

    def _fitness_func(self, ga_instance, solution, solution_idx):
        batch_size = self.states.shape[0]
        sum_error = 0.0

        program = Program(genome=solution)

        # Evaluate the program several times, because evaluations are stochastic
        for eval_run in range(self.config.num_eval_runs):
            for index in range(batch_size):
                action = program(self.states[index])
                desired_action = self.actions[index]

                sum_error += np.mean((action - desired_action) ** 2)

        fitness = -(sum_error / (batch_size + self.config.num_eval_runs))

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
