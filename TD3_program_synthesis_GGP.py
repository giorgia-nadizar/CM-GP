 # docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
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

import envs


RES = []

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
    wandb_project_name: str = "cleanRL_program_synth"
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
    env_id: str = "SimpleGoalSpeed-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000000
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
    learning_starts: int = 1000
    """timestep to start learning"""
    policy_frequency: int = 128
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    # Parameters for the program optimizer
    num_individuals: int = 100
    num_genes: int = 20

    num_generations: int = 30
    num_parents_mating: int = 80
    mutation_probability: float = 0.05

def make_env(env_id, seed, idx, capture_video, run_name):
    if capture_video and idx == 0:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


# [!1] This is a helper function to retrieve proposed actions for each action variable, given a list of observations.
# A program is in our implementation is responsible for only 1 action variable. So, to optimize the actor for an action
# space of size two, 2 separate optimizers are used, each having a population to be optimized for the respective
# variable.

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

    for action_index in range(env.action_space.shape[0]):
        print(f"a[{action_index}] = {program_optimizers[action_index].get_best_solution_str()}")

    qf1 = QNetwork(env).to(device)
    qf2 = QNetwork(env).to(device)
    qf1_target = QNetwork(env).to(device)
    qf2_target = QNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)

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

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if 'episode' in info:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()

        # [!1] Buffer is filled with experiences, also used for the genetic algorithm
        rb.add(obs, real_next_obs, action, reward, termination, info)

        # RESET
        if termination or truncation:
            next_obs, _ = env.reset()

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                )

                # Go over all observations the buffer provides
                next_state_actions = get_state_actions(program_optimizers, data.next_observations.detach().numpy(), env, args)
                next_state_actions = torch.tensor(next_state_actions)
                next_state_actions = (next_state_actions + clipped_noise).clamp(
                    env.action_space.low[0], env.action_space.high[0]).float()

                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            #print(f'Loss critic: {qf1_loss}')

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # Optimize the program
            if global_step % args.policy_frequency == 0:
                # [!1] Use helper function to retrieve proposed actions from the optimizers
                orig_program_actions = get_state_actions(program_optimizers, data.observations.detach().numpy(), env, args)
                cur_program_actions = np.copy(orig_program_actions)
                print('BEFORE ACTIONS', orig_program_actions[0])

                # [!1][!2] The calculation of a more optimal action, given the proposed action of the actor
                for i in range(500):
                    program_actions = torch.tensor(cur_program_actions, requires_grad=True)

                    program_objective_1 = qf1(data.observations, program_actions).mean()
                    program_objective_2 = qf2(data.observations, program_actions).mean()
                    program_objective = (program_objective_1 + program_objective_2) * 0.5
                    program_objective.backward()

                    with torch.no_grad():
                        cur_program_actions += program_actions.grad.numpy() # [!2] Actual calculation using the gradient

                    if np.abs(cur_program_actions - orig_program_actions).mean() > 0.5:
                        break

                print('    TARGET', cur_program_actions[0])
                RES.append(cur_program_actions[0])

                # Fit the program optimizers on all the action dimensions
                states = data.observations.detach().numpy()
                actions = cur_program_actions

                print('Best program:')
                writer.add_scalar("losses/program_objective", program_objective.item(), global_step)

                # [!1] Each action variable has an optimizer to generate a program controlling that variable
                for action_index in range(env.action_space.shape[0]):
                    # [!2] The fitting is the actual optimization process, where the genetic algorithm iterates on the
                    # population of candidates inside each optimizer (--> see ./optim.py).
                    # Given actions to the fit method are the ones more calculated with the gradients above. Inside the
                    # optimizer, the program proposed actions are retrieved for all states of the states argument.
                    program_optimizers[action_index].fit(states, actions[:, action_index])
                    print(f"a[{action_index}] = {program_optimizers[action_index].get_best_solution_str()}")

            # update the target network
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 10 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    env.close()
    writer.close()

    import matplotlib.pyplot as plt
    plt.plot(RES)
    plt.show()


if __name__ == "__main__":
    run_synthesis()
