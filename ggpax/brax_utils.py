from functools import partial
from brax import envs
from brax.envs import ant
from brax.envs.wrappers import EpisodeWrapper
from typing import List, Dict, Callable

from jax import vmap, jit

from ggpax.control_evaluation import evaluate_cgp_genome, evaluate_cgp_genome_n_times, evaluate_lgp_genome, \
    evaluate_lgp_genome_n_times
from ggpax.weighted import encoding_weighted


def init_environment(env_name: str, episode_length: int, terminate_when_unhealthy: bool = True) -> EpisodeWrapper:
    if env_name == "miniant":
        env = partial(ant.Ant, use_contact_forces=False)(terminate_when_unhealthy=terminate_when_unhealthy)
    else:
        try:
            env = envs.get_environment(env_name=env_name, terminate_when_unhealthy=terminate_when_unhealthy)
        except TypeError:
            env = envs.get_environment(env_name=env_name)
    env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
    return env


def init_environment_from_config(config: Dict) -> EpisodeWrapper:
    return init_environment(config["problem"]["environment"], config["problem"]["episode_length"],
                            config.get("unhealthy_termination", True))


def init_environments(config: Dict) -> List[Dict]:
    n_steps = config["problem"]["incremental_steps"]
    min_duration = config["problem"]["min_length"]
    step_size = (config["problem"]["episode_length"] - min_duration) / (n_steps - 1)
    gen_step_size = int(config["n_generations"] / n_steps)
    return [
        {
            "start_gen": gen_step_size * n,
            "env": init_environment(env_name=config["problem"]["environment"],
                                    episode_length=(min_duration + int(step_size * n))
                                    ),
            "fitness_scaler": config["problem"]["episode_length"] / (min_duration + int(step_size * n)),
            "duration": (min_duration + int(step_size * n))
        }
        for n in range(n_steps)
    ]


def compile_genome_evaluation(config: Dict, env, episode_length: int) -> Callable:
    if config["solver"] == "cgp":
        eval_func, eval_n_times_func = evaluate_cgp_genome, evaluate_cgp_genome_n_times
        w_encoding_func = encoding_weighted.genome_to_cgp_program
    else:
        eval_func, eval_n_times_func = evaluate_lgp_genome, evaluate_lgp_genome_n_times
        w_encoding_func = encoding_weighted.genome_to_lgp_program
    if config["n_evals_per_individual"] == 1:
        partial_eval_genome = partial(eval_func, config=config, env=env, episode_length=episode_length)
    else:
        partial_eval_genome = partial(eval_n_times_func, config=config, env=env,
                                      n_times=config["n_evals_per_individual"], episode_length=episode_length)

    if config.get("weighted_connections", False):
        partial_eval_genome = partial(partial_eval_genome, genome_encoder=w_encoding_func)

    vmap_evaluate_genome = vmap(partial_eval_genome, in_axes=(0, 0))
    return jit(vmap_evaluate_genome)
