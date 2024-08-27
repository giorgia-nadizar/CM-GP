import os
import random
import time
from dataclasses import dataclass
import pyrallis

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

from optim import ProgramOptimizer
from postfix_program import Program, NUM_OPERATORS, InvalidProgramException
import envs

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "direct_GP"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "SimpleGoal-v0"
    """the id of the environment"""
    total_timesteps: int = 1_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.1
    """the scale of policy noise"""
    learning_starts: int = 1
    """timestep to start learning"""
    policy_frequency: int = 128
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    # Parameters for the program optimizer
    num_individuals: int = 100
    num_genes: int = 5

    num_generations: int = 10
    num_parents_mating: int = 8
    mutation_probability: float = 0.1

def make_env(env_id, seed, idx, capture_video, run_name):
    if capture_video and idx == 0:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env

def get_state_actions(program_optimizers, obs, env, args):
    program_actions = []

    for i, o in enumerate(obs):
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        for action_index in range(env.action_space.shape[0]):
            action[action_index] = program_optimizers[action_index].get_action(o)

        program_actions.append(action)

    return np.array(program_actions)

@pyrallis.wrap()
def run_synthesis(args: Args):
    N_INTERACTIONS = 0

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    # Actor is a learnable program
    program_optimizers = [ProgramOptimizer(
        args,
        env.observation_space,
        env.action_space.low[i],
        env.action_space.high[i]
    ) for i in range(env.action_space.shape[0])]


    # Add env reference to the ProgramOptimizer (dirty)
    for p in program_optimizers:
        p.env = env
        p._fitness_func = p._fitness_func_env

    for action_index in range(env.action_space.shape[0]):
        print(f"a[{action_index}] = {program_optimizers[action_index].get_best_solution_str()}")

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)


    for global_step in range(args.total_timesteps):

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = get_state_actions(program_optimizers, obs[None, :], env, args)[0]
                action = np.random.normal(loc=action, scale=args.policy_noise)
                print('ACTION', action)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(action)
        N_INTERACTIONS += 1

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if 'episode' in info:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], N_INTERACTIONS)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        rb.add(obs, real_next_obs, action, reward, termination, info)

        # RESET
        if termination or truncation:
            next_obs, _ = env.reset()

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            # Optimize the program
            if global_step % args.policy_frequency == 0:
                orig_program_actions = get_state_actions(program_optimizers, data.observations.detach().numpy(), env,
                                                         args)
                cur_program_actions = np.copy(orig_program_actions)

                # Fit the program optimizers on all the action dimensions
                states = data.observations.detach().numpy()
                actions = cur_program_actions

                print('Best program:')

                for action_index in range(env.action_space.shape[0]):
                    program_optimizers[action_index].fit(states, actions[:, action_index])
                    print(f"a[{action_index}] = {program_optimizers[action_index].get_best_solution_str()}")
                    # Add interactions during optimization
                    for pf, le in zip(program_optimizers[action_index].fitness_pop, program_optimizers[action_index].len_episodes):
                        N_INTERACTIONS += sum(le)
                        #print(N_INTERACTIONS)
                        writer.add_scalar("charts/episodic_return", sum(pf)/(args.num_generations + 1), N_INTERACTIONS)
                        #writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


    env.close()
    writer.close()

if __name__ == "__main__":
    run_synthesis()