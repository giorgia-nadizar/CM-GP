import sys

import pygad
import numpy as np
import ctypes
import pyrallis
from dataclasses import dataclass

from postfix_program import Program, NUM_OPERATORS, InvalidProgramException
from copy import deepcopy

def print_fitness(ga, fitnesses):
    print('F', fitnesses.mean(), file=sys.stderr)

class ProgramOptimizer:
    def __init__(self, config, state_space, low, high):
        self.config = config
        self.state_dim = state_space.shape[0]
        self.low = low
        self.high = high

        # Create the initial population
        # We create it so these random programs try all the operators and read all the state variables
        self.initial_population = np.random.random((config.num_individuals, config.num_genes))  # Random numbers between 0 and 1
        self.initial_population *= -(NUM_OPERATORS + self.state_dim)         # Tokens between -NUM_OPERATORS - state_dim and 0

        self.best_solution = self.initial_population[0]
        self.best_fitness = None

        self.len_episodes = np.ndarray(config.num_individuals)
        self.fitness_pop = np.ndarray(config.num_individuals)

    def get_action(self, state):
        program = Program(self.best_solution, self.state_dim, self.low, self.high)

        try:
            return program(state)
        except InvalidProgramException:
            return np.random.normal()

    def get_best_solution_str(self):
        program = Program(self.best_solution, self.state_dim, self.low, self.high)

        try:
            return program.to_string()
        except InvalidProgramException:
            return '<invalid program>'

    def _fitness_func(self, ga_instance, solution, solution_idx):
        program = Program(solution, self.state_dim, self.low, self.high)

        try:
            # Num input variables looked at
            lookedat = program.num_inputs_looked_at()
            looked_proportion = lookedat / self.state_dim

            # Evaluate the program several times, because evaluations are stochastic
            batch_size = self.states.shape[0]
            sum_error = 0.0

            for index in range(batch_size):
                # MSE for the loss
                action = program(self.states[index])
                desired_action = self.actions[index]

                sum_error += (action - desired_action) ** 2

            avg_error = (sum_error / batch_size)
            fitness = (1.0 - avg_error) * looked_proportion
        except InvalidProgramException:
            fitness = -1000.0

        return fitness

    def _fitness_func_env(self, ga_instance, solution, solution_idx):
        program = Program(solution, self.state_dim, self.low, self.high)

        fitness = 0.0
        l = 0
        terminated, truncated = False, False
        env = deepcopy(self.env)
        obs, _ = env.reset()
        while not terminated or not truncated:
            l += 1
            action = program(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            fitness += reward
            if terminated or truncated:
                break
        self.len_episodes[solution_idx] = l
        self.fitness_pop[solution_idx] = fitness
        return fitness

    def fit(self, states, actions):
        """ states is a batch of states, shape (N, state_shape)
            actions is a batch of actions, shape (N,), we assume continuous actions

            NOTE: One ProgramOptimizer has to be used for each action dimension
        """
        self.len_episodes = np.ndarray(self.config.num_individuals)
        self.fitness_pop = np.ndarray(self.config.num_individuals)
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

            parent_selection_type="sss",
            crossover_type="single_point",
            mutation_type="random",
            #parallel_processing=["process", 1],

            on_fitness=print_fitness
        )

        self.ga_instance.run()

        # Allow the population to survive
        self.initial_population = self.ga_instance.population

        # Best solution for now
        self.best_solution = self.ga_instance.best_solution()[0]
