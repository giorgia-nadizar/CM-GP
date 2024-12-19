import functools
from brax import envs
from brax.envs import ant
from brax.envs.wrappers import EpisodeWrapper
from typing import List, Dict


def init_environment(env_name: str, episode_length: int, terminate_when_unhealthy: bool = True) -> EpisodeWrapper:
    if env_name == "miniant":
        env = functools.partial(ant.Ant, use_contact_forces=False)(terminate_when_unhealthy=terminate_when_unhealthy)
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
