import sys

import pygad
import numpy as np
import pyrallis
from dataclasses import dataclass

from postfix_program import Program, NUM_OPERATORS

class ProgramOptimizer:
    def __init__(self, config, action_shape):

        # Create the initial population
        self.action_shape = action_shape
        self.initial_program = [0.0] * (config.num_genes * 2 * action_shape[0])  # Mean and log_std for each gene, for each action dimension

        self.best_solution = self.initial_program
        self.best_fitness = None

        self.config = config
        self.initial_population = [np.array(self.initial_program) for i in range(config.num_individuals)]

    def get_actions_from_solution(self, solution, state):
        # One program per action dimension
        program_length = self.config.num_genes * 2
        programs = [
            Program(genome=solution[i*program_length : (i+1)*program_length])
            for i in range(self.action_shape[0])
        ]

        return np.array([p(state) for p in programs], dtype=np.float32)

    def print_best_solution(self):
        program_length = self.config.num_genes * 2

        for i in range(self.action_shape[0]):
            p = Program(genome=self.best_solution[i*program_length : (i+1)*program_length])
            print(f'a[{i}] =', p.run_program([0.0], do_print=True))

    def _fitness_func(self, ga_instance, solution, solution_idx):
        batch_size = self.states.shape[0]
        sum_error = 0.0

        # Evaluate the program several times, because evaluations are stochastic
        for eval_run in range(self.config.num_eval_runs):
            for index in range(batch_size):
                action = self.get_actions_from_solution(solution, self.states[index])
                desired_action = self.actions[index]

                sum_error += np.mean((action - desired_action) ** 2)

        fitness = -(sum_error / (batch_size + self.config.num_eval_runs))

        return fitness

    def fit(self, states, actions):
        """ states is a batch of states, shape (N, state_shape)
            actions is a batch of actions, shape (N, action_shape), we assume continuous actions
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

@dataclass
class Config:
    num_individuals: int = 1000
    num_genes: int = 10

    num_generations: int = 20
    num_parents_mating: int = 10
    keep_parents: int = 5
    mutation_percent_genes: int = 10
    keep_elites: int = 5


@pyrallis.wrap()
def main(config: Config):
    optim = ProgramOptimizer(config)

    # Sample states and actions
    #states = np.array([
    #    [1.0],
    #    [2.0],
    #    [-5.0],
    #    [10.0],
    #])

    #states = np.array([[1.0, 2.0], [2.0, 4.0]])
    #actions = np.array([[3.0], [6.0]])

    states = np.random.random_sample((10, 2))
    actions = np.sum(states, axis=1)
    actions = np.reshape(actions, (10, 1))

    #states = np.load('runs/InvertedPendulum-v4__TD3__1__1720706887/TD3.cleanrl_model_OBSERVATIONS.npy')
    #actions = np.load('runs/InvertedPendulum-v4__TD3__1__1720706887/TD3.cleanrl_model_ACTIONS.npy')

    # Fit
    optim.fit(states, actions)

    # Plot
    optim.ga_instance.plot_fitness()


if __name__ == '__main__':
    main()
