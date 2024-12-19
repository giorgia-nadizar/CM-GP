import copy
import multiprocessing
import sys

import jax.numpy as jnp
from jax import random

from ggpax.run_utils import compute_masks, compute_genome_transformation_function, compile_parents_selection, \
    compile_mutation, compile_survival_selection, update_config_with_data
from ggpax.standard import individual
from ggpax.standard.encoding import genome_to_cgp_program, genome_to_lgp_program
from ggpax.utils import cgp_expression_from_genome, lgp_expression_from_genome


def print_fitness(fitnesses):
    print('F', fitnesses.mean(), file=sys.stderr)


class GraphProgramOptimizer:
    def __init__(self, ggp_config, state_space, action_space):

        self.states = None
        self.actions = None

        # add data info to the config
        self.config = copy.deepcopy(ggp_config)
        update_config_with_data(self.config, state_space.shape[0], action_space.shape[0])
        self.state_dim = state_space.shape[0]
        # TODO for now assume in (-1,1)
        self.low = action_space.low[0]
        self.high = action_space.high[0]
        self.rnd_key = random.PRNGKey(self.config["seed"])

        # Create structures that are needed during the evolution
        genome_mask, mutation_mask = compute_masks(self.config)
        genome_transformation_function = compute_genome_transformation_function(self.config)
        self.genome_encoder = genome_to_cgp_program if self.config["solver"] == "cgp" else genome_to_lgp_program
        self.select_parents = compile_parents_selection(self.config)
        self.mutate_genomes = compile_mutation(self.config, genome_mask, mutation_mask, genome_transformation_function)
        self.select_survivals = compile_survival_selection(self.config)

        # Create the initial population of only identities
        self.rnd_key, genome_key = random.split(self.rnd_key, 2)
        self.population = individual.generate_population(pop_size=self.config["n_individuals"],
                                                         genome_mask=genome_mask, rnd_key=genome_key,
                                                         genome_transformation_function=genome_transformation_function)
        self.best_solution = self.population[0]
        self.best_fitness = None
        self.len_episodes = jnp.zeros(self.config["n_individuals"])
        self.fitness_pop = jnp.zeros(self.config["n_individuals"])

    def get_program_reset_state(self):
        program_state_key = "buffer_size" if self.config["solver"] == "cgp" else "n_registers"
        return jnp.zeros(self.config[program_state_key])

    def get_actions(self, state):
        best_program = self.genome_encoder(self.best_solution, self.config)
        if state.ndim == 1:
            _, actions = best_program(state, self.get_program_reset_state())
        else:
            actions_stack = []
            for s in state:
                _, acts = best_program(s, self.get_program_reset_state())
                actions_stack.append(acts)
            actions = jnp.vstack(actions_stack)
        return actions

    def get_best_solution_str(self):
        expression_from_genome_fn = cgp_expression_from_genome if self.config[
                                                                      "solver"] == "cgp" else lgp_expression_from_genome
        return expression_from_genome_fn(self.best_solution, self.config)

    def _fitness_func(self, solution):
        program = self.genome_encoder(solution, self.config)

        # Evaluate the program several times, because evaluations are stochastic
        # MSE for the loss
        # [!2] Retrieve proposed actions and calculate the distance with more optimal actions
        predicted_actions = []
        for current_state in self.states:
            _, actions = program(current_state, self.get_program_reset_state())
            predicted_actions.append(actions)

        avg_error = jnp.mean((jnp.vstack(predicted_actions) - self.actions) ** 2).item()
        return 1. - avg_error

    def _fitness_func_env(self, solution, solution_idx):
        program = self.genome_encoder(solution, self.config)

        fitness = 0.0
        l = 0
        terminated, truncated = False, False
        obs, _ = self.env.reset()
        while not terminated or not truncated:
            l += 1
            _, action = program(obs, self.get_program_reset_state())
            obs, reward, terminated, truncated, info = self.env.step(action)
            fitness += reward
            if terminated or truncated:
                break
        self.len_episodes[solution_idx].append(l)
        self.fitness_pop[solution_idx].append(fitness)
        return fitness

    def ggp_ga(self):
        # exec_pool = multiprocessing.Pool(processes=20)
        genomes = self.population
        for _generation in range(self.config["num_generations"]):
            # evaluate the fitness of the population on the regression problem
            # fitness_values = exec_pool.map(self._fitness_func, genomes)
            fitness_values = jnp.asarray([self._fitness_func(g) for g in genomes])
            # select parents
            self.rnd_key, select_key = random.split(self.rnd_key, 2)
            parents = self.select_parents(genomes, fitness_values, select_key)
            # perform mutation
            self.rnd_key, mutate_key = random.split(self.rnd_key, 2)
            mutate_keys = random.split(mutate_key, len(parents))
            offspring_matrix = self.mutate_genomes(parents, mutate_keys)
            offspring = jnp.reshape(offspring_matrix, (-1, offspring_matrix.shape[-1]))
            # select survivals
            self.rnd_key, survival_key = random.split(self.rnd_key, 2)
            survivals = parents if self.select_survivals is None else self.select_survivals(genomes, fitness_values,
                                                                                            survival_key)
            # update population
            assert len(genomes) == len(survivals) + len(offspring)
            genomes = jnp.concatenate((survivals, offspring))

        # fitness_values = exec_pool.map(self._fitness_func, genomes)
        fitness_values = jnp.asarray([self._fitness_func(g) for g in genomes])

        return genomes, fitness_values

    def fit(self, states, actions):
        """ states is a batch of states, shape (N, state_shape)
            actions is a batch of actions, shape (N,), we assume continuous actions
        """
        self.len_episodes = [[] for _ in range(self.config["n_individuals"])]
        self.fitness_pop = [[] for _ in range(self.config["n_individuals"])]
        self.states = states  # picklable self._fitness_func needs these instance variables
        self.actions = actions

        self.population, self.fitness_pop = self.ggp_ga()
        self.best_fitness = jnp.max(self.fitness_pop)
        self.best_solution = self.population[jnp.argmax(self.fitness_pop)]
