import sys

import pygad
import numpy as np
import pyrallis
from dataclasses import dataclass

from postfix_program import Program, NUM_OPERATORS

N_INPUT_VARIABLES = 2

FITNESS = []

@dataclass
class Config:
    num_individuals: int = 1000
    num_genes: int = 10

    num_generations: int = 20
    num_parents_mating: int = 10
    keep_parents: int = 5
    mutation_percent_genes: int = 10
    keep_elites: int = 5

class ProgramOptimizer:
    def __init__(self, config: Config):

        # Create the initial population
        self.initial_program = [-1.0] * config.num_genes

        self.best_solution = self.initial_program
        self.best_fitness = None

        self.config = config
        self.initial_population = [np.array(self.initial_program) for i in range(config.num_individuals)]

    def get_program(self):
        return Program(genome=self.best_solution)

    def fit(self, states, actions):
        """ states is a batch of states, shape (N, state_shape)
            actions is a batch of actions, shape (N, action_shape), we assume continuous actions
        """
        #self.best_solution = None
        #self.best_fitness = None

        def fitness_func(ga_instance, solution, solution_idx):
            batch_size = states.shape[0]
            action_size = actions.shape[1]
            sum_error = 0.0

            for index in range(batch_size):
                program = self.get_program()
                action = program(states[index], len_output=action_size)
                #action = np.array(action + [0.0] * action_size)
                #action = action[:action_size]
                desired_action = actions[index]

                #print(states[index], desired_action, action)

                sum_error += np.mean((action - desired_action) ** 2)

            fitness = -(sum_error / batch_size)
            #FITNESS.append(fitness)

            # !!!!
            if self.best_fitness is None or fitness > self.best_fitness:
                self.best_solution = solution
                self.best_fitness = fitness

            #print('F', fitness, file=sys.stderr)
            return fitness

        self.ga_instance = pygad.GA(num_generations=self.config.num_generations,
            parallel_processing=8,
            save_solutions=True,
            save_best_solutions=True,
            num_parents_mating=self.config.num_parents_mating,
            fitness_func=fitness_func,
            initial_population=self.initial_population,
            #sol_per_pop=500,
            parent_selection_type="sss",
            keep_parents=self.config.keep_parents,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=self.config.mutation_percent_genes,
            random_mutation_max_val=10,
            random_mutation_min_val=-10,
            gene_space={
                'low': -NUM_OPERATORS - N_INPUT_VARIABLES,
                'high': 1.0
                },
            keep_elitism=10,
        )

        #print(ga_instance.run())
        self.ga_instance.run()

        # Allow the population to survive
        self.initial_population = self.ga_instance.population

        # Print the best individual
        program = self.get_program()
        #print(program(states[0], do_print=True))

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

    states = np.random.random_sample((10,2))
    actions = np.sum(states, axis=1)
    actions = np.reshape(actions, (10, 1))

    #states = np.load('runs/InvertedPendulum-v4__TD3__1__1720706887/TD3.cleanrl_model_OBSERVATIONS.npy')
    #actions = np.load('runs/InvertedPendulum-v4__TD3__1__1720706887/TD3.cleanrl_model_ACTIONS.npy')

    # Fit
    optim.fit(states, actions)

    # Plot
    import matplotlib.pyplot as plt
    #plt.plot(FITNESS)
    #plt.show()
    optim.ga_instance.plot_fitness()
    print('done')

if __name__ == '__main__':
    main()