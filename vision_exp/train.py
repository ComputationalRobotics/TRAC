# Adapted from https://github.com/joonleesky/train-procgen-pytorch

from common.env.procgen_wrappers import *
from common.env.atari_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.policy import CategoricalPolicy
from common.model import ImpalaModel
from agents.ppo import PPO as AGENT
from common import set_global_seeds, set_global_log_levels

import os, time, argparse
from procgen import ProcgenEnv
import random
import torch
from pathlib import Path

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',         type=str, default = 'procgen', help='experiment name, either procgen or atari')
    parser.add_argument('--env_name',         type=str, default = 'starpilot', help='environment ID')
    parser.add_argument('--seed',             type=int, default = random.randint(0,9999), help='Random generator seed')
    parser.add_argument('--log_level',        type=int, default = int(5), help='[10,20,30,40]')
    parser.add_argument('--optimizer',        type=str, default = 'TRAC', help='optimizer to use')
    parser.add_argument('--env_seed',         type=int, default = int(20), help='Environment')
    parser.add_argument('--level_steps',      type=int, default = int(2000000), help='Number of steps per task')
    parser.add_argument('--replay_ratio',     type=int, default = int(1000), help='Number of steps per update')
    parser.add_argument('--max_levels',       type=int, default = int(45), help='Number of distribution shifts')
    parser.add_argument('--storage_r',        type=int, default = int(1), help='s1 or s2')
    parser.add_argument('--warmstart_step',  type=int, default = int(0), help='warmstart step')
    args = parser.parse_args()
    
    exp_name = args.exp_name
    env_name = args.env_name
    start_level = args.env_seed
    num_levels = args.max_levels
    seed = args.seed
    log_level = args.log_level
    optimizer = args.optimizer
    env_seed = args.env_seed
    level_steps = args.level_steps
    update_steps = args.replay_ratio
    max_levels = args.max_levels
    num_timesteps = level_steps * max_levels
    storage_r = True if args.storage_r == 1 else 0
    warmstart_step = args.warmstart_step
    set_global_seeds(seed)
    set_global_log_levels(log_level)

    # define for atari key: ["Qbert", ...]
    env_set = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: " + str(device))
    torch.set_num_threads(1)
    
    if exp_name == 'procgen':
        env = ProcgenEnv(num_envs=1,
                        env_name=env_name,
                        start_level=0,
                        num_levels=1,
                        distribution_mode='hard')
        env = VecExtractDictObs(env, "rgb")
        env = VecNormalize(env, ob=False)
        env = TransposeFrame(env)
        env = ScaledFloatFrame(env)
    else:
        env = gym.make(f'ALE/{env_set[env_name][0]}-ram-v5', render_mode="rgb_array", obs_type="rgb")    
        obs = env.reset()
        obs = np.transpose(obs[0], (2, 0, 1))
        
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    if exp_name == "atari":
        observation_shape = obs.shape
    in_channels = observation_shape[0]
    action_space = env.action_space

    root = f'{args.exp_name}/logs/{args.env_name}/{level_steps}/{update_steps}/{storage_r}/'
    if warmstart_step:
        logdir = f'{root}{optimizer}_warmstarted/{env_seed}/seed_{seed}_{time.strftime("%d-%m-%Y_%H-%M-%S")}'
    else:
        logdir = f'{root}{optimizer}/{env_seed}/seed_{seed}_{time.strftime("%d-%m-%Y_%H-%M-%S")}'
    Path(logdir).mkdir(parents=True, exist_ok=True)
    optimizer_log_file_path = os.path.join(logdir, f"sinit_{seed}.txt")
    logger = Logger(1, logdir)
    if args.exp_name == 'procgen':
        model = ImpalaModel(in_channels=in_channels)
    elif args.exp_name == 'atari':
        model = ImpalaModel(in_channels=in_channels, in_features=32*18*10*3)
    action_size = action_space.n
    policy = CategoricalPolicy(model, action_size)
    policy.to(device)

    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, update_steps, 1, device)
    
    agent = AGENT(env, policy, logger, storage, device, 1, update_steps)
    if exp_name == 'procgen':
        env_generator = AGENT.environment_generator_procgen(env_name, max_levels, seed=env_seed)
    else:
        env_generator = AGENT.environment_generator_atari(env_set[env_name])

    print('START TRAINING with ' + str(optimizer) + '...')
    agent.train_seq(env_generator, optimizer_log_file_path=optimizer_log_file_path, exp_name=exp_name,total_steps_per_level=level_steps, optimizer=optimizer, storage_r=storage_r, warmstart=warmstart_step)
    