from .base_agent import BaseAgent
from common.env.procgen_wrappers import *
import torch
import numpy as np
from procgen import ProcgenEnv
from trac import start_trac


class PPO(BaseAgent):
    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 n_steps=1000,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=2048,
                 gamma=0.999,
                 lmbda=0.95,
                 learning_rate=0.001,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 **kwargs):

        super(PPO, self).__init__(env, policy, logger, storage, device, n_checkpoints)

        self.n_steps = n_steps
        self.n_envs = 1
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = None
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae

    def predict(self, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1-done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()
    
    def environment_generator_procgen(env_name, total_levels, seed=0, n_envs=1, distribution_mode="hard"):
        for current_level in range(total_levels):
            current_level +=seed
            env = ProcgenEnv(num_envs=n_envs, env_name=env_name, start_level=current_level, num_levels=1, distribution_mode=distribution_mode)
            env = VecExtractDictObs(env, "rgb")
            env = VecNormalize(env, ob=False)
            env = TransposeFrame(env)
            env = ScaledFloatFrame(env)
            yield env
    def environment_generator_atari(env_set):
        for env_name in env_set:
            env = gym.make('ALE/' + env_name + '-ram-v5', render_mode='rgb_array', obs_type='rgb')
            yield env
    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for _ in range(self.epoch):
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size)
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                mask_batch = (1-done_batch)
                dist_batch, value_batch, _ = self.policy(obs_batch, hidden_state_batch, mask_batch)

                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                entropy_loss = dist_batch.entropy().mean()
                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss.backward()

                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(pi_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list)}
        return summary
        
    def train_seq(self, env_generator, optimizer_log_file_path, exp_name, total_steps_per_level=1000000, optimizer="TRAC", storage_r=True, warmstart=0):
        steps_per_optimization = self.n_steps
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)
        expand = True if exp_name == "atari" else False
        # first we set opt to BASE
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)
        
        current_level = 0
        global_step = 0
        truncated = False
        for self.env in env_generator:
            current_level +=1
            level_steps = 0
            obs = self.env.reset()
            if expand:
                obs = np.expand_dims(np.transpose(obs[0], (2, 0, 1)), axis=0)
            while level_steps < total_steps_per_level:
                if global_step == warmstart:
                    if optimizer == "TRAC":
                        self.optimizer = start_trac(log_file=optimizer_log_file_path,Base=torch.optim.Adam)(self.policy.parameters(), lr=self.learning_rate)
                    else:
                        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)
                obs = self.env.reset()
                if expand:
                    obs = np.expand_dims(np.transpose(obs[0], (2, 0, 1)), axis=0)
                self.policy.eval()
                for _ in range(steps_per_optimization):
                    act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done)
                    if expand:
                        next_obs, rew, done, truncated, info = self.env.step(act[0])
                        self.storage.store(obs, hidden_state, act, np.array([rew]), np.array([done]), np.array([info]), log_prob_act, value)
                        obs = np.expand_dims(np.transpose(next_obs, (2, 0, 1)), axis=0)
                    else:
                        next_obs, rew, done, info = self.env.step(act)
                        self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
                        obs = next_obs
                    hidden_state = next_hidden_state
                    level_steps += 1
                    global_step += 1
                    if done or truncated:
                        obs = self.env.reset()
                        if expand:
                            obs = np.expand_dims(np.transpose(obs[0], (2, 0, 1)), axis=0)

                _, _, last_val, hidden_state = self.predict(obs, hidden_state, done)
                self.storage.store_last(obs, hidden_state, last_val)
                self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)
                summary = self.optimize()
                self.t += self.n_steps * self.n_envs
                rew_batch, done_batch = self.storage.fetch_log_data()
                self.logger.feed(rew_batch, done_batch)
                self.logger.write_summary(summary)
                self.logger.dump()
                if storage_r:
                    self.storage.reset()
            if not storage_r:
                self.storage.reset()
            print("LEVEL SWITCHED")
            
        self.env.close()
