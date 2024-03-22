# pylint: disable=all
# PufferLib's customized CleanRL PPO + LSTM implementation
from pdb import set_trace as T

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import collections
from typing import Tuple
from tqdm import tqdm

import pufferlib
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.policy_pool
import pufferlib.policy_ranker
import pufferlib.utils
import pufferlib.vectorization
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        # Define the architecture of the DQN network here. 
        # It typically has a few fully connected layers.
        self.layers = nn.Sequential(
            nn.Linear(observation_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )
    
    def forward(self, x):
        return self.layers(x)

class DQNAgent:
    def __init__(self, observation_space, action_space):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
        self.policy_net = DQN(observation_space, action_space).to(self.device)
        self.target_net = DQN(observation_space, action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set the target network to eval mode
        self.is_recurrent = False
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.004)
        
        self.action_space = action_space
    
    def select_action(self, state, epsilon):
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state.unsqueeze(0))
                return q_values.max(1)[1].view(1, 1).item()
        else:
            return random.randrange(self.action_space)
    
    def optimize_model(self, gamma, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(-1))

        # Compute V(s_{t+1}) for all next states using the target network.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) * (1 - done_batch) + reward_batch

        # Compute Huber loss
        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v

class ReplayBuffer:

    def __init__(self):
        self.buffer = collections.deque()
        self.l1 = 1
    def add(self, obs: torch.Tensor, actions: torch.Tensor, values: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor):
        obs = obs.detach().cpu().numpy()
        actions = actions.detach().cpu().numpy()
        values = values.detach().cpu().numpy()
        advantages = advantages.detach().cpu().numpy()
        returns = returns.detach().cpu().numpy()
        self.buffer.append((obs, actions, values, advantages,returns))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, values, advantages, returns = zip(*batch)
        return np.array(states), np.array(actions), values[0], advantages[0], np.array(returns)
    def __len__(self):
        return len(self.buffer)

@dataclass
class CleanPuffeRLDQN:
    env_creator: callable = None
    env_creator_kwargs: dict = None
    agent: nn.Module = None
    agent_creator: callable = None
    agent_kwargs: dict = None

    exp_name: str = os.path.basename(__file__)

    data_dir: str = 'data'
    record_loss: bool = False
    checkpoint_interval: int = 1
    seed: int = 1
    torch_deterministic: bool = True
    vectorization: ... = pufferlib.vectorization.Serial
    device: str = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    num_buffers: int = 1
    num_envs: int = 8
    num_cores: int = psutil.cpu_count(logical=False)
    cpu_offload: bool = True
    verbose: bool = True
    batch_size: int = 2**14
    policy_store: pufferlib.policy_store.PolicyStore = None
    policy_ranker: pufferlib.policy_ranker.PolicyRanker = None

    policy_pool: pufferlib.policy_pool.PolicyPool = None
    policy_selector: pufferlib.policy_ranker.PolicySelector = None

    # Wandb
    wandb_entity: str = None
    wandb_project: str = None
    wandb_extra_data: dict = None

    # Selfplay
    selfplay_learner_weight: float = 1.0
    selfplay_num_policies: int = 1
    

    def __post_init__(self, *args, **kwargs):
        self.start_time = time.time()

        # If data_dir is provided, load the resume state
        resume_state = {}
        if self.data_dir is not None:
          path = os.path.join(self.data_dir, f"trainer.pt")
          if os.path.exists(path):
            print(f"Loaded checkpoint from {path}")
            resume_state = torch.load(path)
            print(f"Resuming from update {resume_state['update']} "
                  f"with policy {resume_state['policy_checkpoint_name']}")

        self.wandb_run_id = resume_state.get("wandb_run_id", None)
        self.learning_rate = resume_state.get("learning_rate", self.learning_rate)

        self.global_step = resume_state.get("global_step", 0)
        self.agent_step = resume_state.get("agent_step", 0)
        self.update = resume_state.get("update", 0)

        self.total_updates = self.total_timesteps // self.batch_size
        self.envs_per_worker = self.num_envs // self.num_cores
        assert self.num_cores * self.envs_per_worker == self.num_envs

        # Seed everything
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.seed is not None:
            torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

        # Create environments
        self.process = psutil.Process()
        allocated = self.process.memory_info().rss
        self.buffers = [
            self.vectorization(
                self.env_creator,
                env_kwargs=self.env_creator_kwargs,
                num_workers=self.num_cores,
                envs_per_worker=self.envs_per_worker,
            )
            for _ in range(self.num_buffers)
        ]
        self.num_agents = self.buffers[0].num_agents
        print("buffer obs",*self.buffers[0].single_observation_space.shape, "action",*self.buffers[0].single_action_space.shape)
        
        # If an agent_creator is provided, use envs (=self.buffers[0]) to create the agent
        self.agent = DQNAgent(*self.buffers[0].single_observation_space.shape, *self.buffers[0].single_action_space.shape)
        if self.verbose:
            print(
                "Allocated %.2f MB to environments. Only accurate for Serial backend."
                % ((self.process.memory_info().rss - allocated) / 1e6)
            )

        self.epsilon = 0.82
        # Create policy store: omitted

        # Create policy ranker: omitted

        # Setup agent: omitted

        # TODO: this can be cleaned up
        #self.agent.is_recurrent = hasattr(self.agent, "lstm")
        #self.agent = self.agent.to(self.device)

        # Setup policy pool ...

        # Setup policy selector ...

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.agent.policy_net.parameters(), lr=self.learning_rate, eps=1e-5
        )
        if "optimizer_state_dict" in resume_state:
          self.optimizer.load_state_dict(resume_state["optimizer_state_dict"])

        self.replay_buffer = ReplayBuffer()
        # Prepare model and optimizer for training
        self.policy_net = self.agent.policy_net
        self.target_net = self.agent.target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        ### Allocate Storage:?
        next_obs, next_done, next_lstm_state = [], [], []
        for i, envs in enumerate(self.buffers):
            envs.async_reset(self.seed + i)
            next_done.append(
                torch.zeros((self.num_envs * self.num_agents,)).to(self.device)
            )
            next_obs.append([])

            # if self.agent.is_recurrent:
            #     shape = (
            #         self.agent.lstm.num_layers,
            #         self.num_envs * self.num_agents,
            #         self.agent.lstm.hidden_size,
            #     )
            #     next_lstm_state.append(
            #         (
            #             torch.zeros(shape).to(self.device),
            #             torch.zeros(shape).to(self.device),
            #         )
            #     )
            # else:
            #     next_lstm_state.append(None)

        allocated_torch = torch.cuda.memory_allocated(self.device)
        allocated_cpu = self.process.memory_info().rss
        self.data = SimpleNamespace(
            buf=0,
            sort_keys=[],
            next_obs=next_obs,
            next_done=next_done,
            next_lstm_state=next_lstm_state,
            obs=torch.zeros(
                self.batch_size + 1, *self.buffers[0].single_observation_space.shape
            ).to("cpu" if self.cpu_offload else self.device),
            actions=torch.zeros(
                self.batch_size + 1, *self.buffers[0].single_action_space.shape, dtype=int
            ).to(self.device),
            logprobs=torch.zeros(self.batch_size + 1).to(self.device),
            rewards=torch.zeros(self.batch_size + 1).to(self.device),
            dones=torch.zeros(self.batch_size + 1).to(self.device),
            values=torch.zeros(self.batch_size + 1).to(self.device),
        )
        #???_
        allocated_torch = torch.cuda.memory_allocated(self.device) - allocated_torch
        allocated_cpu = self.process.memory_info().rss - allocated_cpu
        if self.verbose:
            print(
                "Allocated to storage - Pytorch: %.2f GB, System: %.2f GB"
                % (allocated_torch / 1e9, allocated_cpu / 1e9)
            )

        if self.record_loss and self.data_dir is not None:
            self.loss_file = os.path.join(self.data_dir, "loss.txt")
            with open(self.loss_file, "w") as f:
                pass
            self.action_file = os.path.join(self.data_dir, "actions.txt")
            with open(self.action_file, "w") as f:
                pass

        if self.wandb_entity is not None:
            self.wandb_run_id = self.wandb_run_id or wandb.util.generate_id()

            wandb.init(
                id=self.wandb_run_id,
                project=self.wandb_project,
                entity=self.wandb_entity,
                config=self.wandb_extra_data or {},
                sync_tensorboard=True,
                name=self.exp_name,
                monitor_gym=True,
                save_code=True,
                resume="allow",
            )

    @pufferlib.utils.profile
    def evaluate(self, show_progress=False):
        # Pick new policies for the policy pool
        # TODO: find a way to not switch mid-stream
        # self.policy_pool.update_policies({
        #     p.name: p.policy(
        #         policy_args=[self.buffers[0]],
        #         device=self.device,
        #     ) for p in self.policy_store.select_policies(self.policy_selector)
        # })

        allocated_torch = torch.cuda.memory_allocated(self.device)
        allocated_cpu = self.process.memory_info().rss
        ptr = env_step_time = inference_time = agent_steps_collected = 0
        padded_steps_collected = 0

        step = 0
        infos = defaultdict(lambda: defaultdict(list))
        stats = defaultdict(lambda: defaultdict(list))
        performance = defaultdict(list)
        progress_bar = tqdm(total=self.batch_size, disable=not show_progress)

        data = self.data
        total_rewards = []
        actions, logprob, value = [], [], []
        ##obs = self.buffers[buf].reset() ###
        while True:
            buf = data.buf

            step += 1
            if ptr == self.batch_size + 1:
                break
            
            start = time.time()
            
            done = False
            episode_reward = 0
            obs, r, d, i = self.buffers[buf].recv()
            obs = torch.Tensor(obs)
            if not self.cpu_offload:
                obs = obs.to(self.device)

            r = torch.Tensor(r).float().to(self.device).view(-1)

            if len(d) != 0 and len(data.next_done[buf]) != 0:
                alive_mask = (data.next_done[buf].cpu() + torch.Tensor(d)) != 2
                data.next_done[buf] = torch.Tensor(d).to(self.device)
            else:
                alive_mask = [1 for _ in range(len(obs))]

            agent_steps_collected += sum(alive_mask)
            padded_steps_collected += len(alive_mask)

            # ALGO LOGIC: action logic
            while not done:
                action = self.agent.select_action(state, self.epsilon)
                next_state, reward, done, _ = self.buffers[buf].step(action)
                episode_reward += reward
                state = next_state
                actions.append(action.cpu().numpy())

            actions = np.array(actions)    
            #total_rewards.append(episode_reward)

            self.buffers[buf].send(actions, None)
            env_step_time += time.time() - start
            data.buf = (data.buf + 1) % self.num_buffers

            # Index alive mask with policy pool idxs...
            # TODO: Find a way to avoid having to do this
            if self.selfplay_learner_weight > 0:
              alive_mask = np.array(alive_mask) * self.policy_pool.learner_mask

            for idx in np.where(alive_mask)[0]:
                if ptr == self.batch_size + 1:
                    break

                data.obs[ptr] = obs[idx]
                #data.values[ptr] = value[idx]
                data.actions[ptr] = actions[idx]
                #data.logprobs[ptr] = logprob[idx]
                data.sort_keys.append((buf, idx, step))

                if len(d) != 0:
                    data.rewards[ptr] = r[idx]
                    data.dones[ptr] = d[idx]

                ptr += 1
                progress_bar.update(1)


        env_sps = int(agent_steps_collected / env_step_time)
        inference_sps = int(padded_steps_collected / inference_time)

        progress_bar.set_description(
            "Eval: "
            + ", ".join(
                [
                    f"Env SPS: {env_sps}",
                    f"Inference SPS: {inference_sps}",
                    f"Agent Steps: {agent_steps_collected}",
                    *[f"{k}: {np.mean(v):.2f}" for k, v in stats['learner'].items()],
                ]
            )
        )

        self.global_step += self.batch_size

        if self.wandb_entity:
            wandb.log(
                {
                    "performance/env_time": env_step_time,
                    "performance/env_sps": env_sps,
                    "performance/inference_time": inference_time,
                    "performance/inference_sps": inference_sps,
                    **{
                        f"performance/env/{k}": np.mean(v)
                        for k, v in performance.items()
                    },
                    **{f"charts/{k}": np.mean(v) for k, v in stats['learner'].items()},
                    "charts/reward": float(torch.mean(data.rewards)),
                    "agent_steps": self.global_step,
                    "global_step": self.global_step,
                }
            )

        allocated_torch = torch.cuda.memory_allocated(self.device) - allocated_torch
        allocated_cpu = self.process.memory_info().rss - allocated_cpu
        if self.verbose:
            print(
                "Allocated during evaluation - Pytorch: %.2f GB, System: %.2f GB"
                % (allocated_torch / 1e9, allocated_cpu / 1e9)
            )

        uptime = timedelta(seconds=int(time.time() - self.start_time))
        print(
            f"Epoch: {self.update} - {self.global_step // 1000}K steps - {uptime} Elapsed\n"
            f"\tSteps Per Second: Env={env_sps}, Inference={inference_sps}"
        )

        progress_bar.close()
        return data, stats, infos

    @pufferlib.utils.profile
    def train_depre(
        self,
        batch_rows=32,
        update_epochs=4,
        bptt_horizon=16,
        gamma=0.99,
        gae_lambda=0.95,
        anneal_lr=True,
        norm_adv=True,
        clip_coef=0.1,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        # ADD replay_buffer
        replay_buffer = ReplayBuffer(),
        batch_cff = 2
    ):
        if self.done_training():
            raise RuntimeError(
                f"Trying to train for more than max_updates={self.total_updates} updates"
            )

        # assert self.num_steps % bptt_horizon == 0, "num_steps must be divisible by bptt_horizon"
        allocated_torch = torch.cuda.memory_allocated(self.device)
        allocated_cpu = self.process.memory_info().rss

        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (self.update - 1.0) / self.total_updates
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        # Sort here
        data = self.data
        idxs = sorted(range(len(data.sort_keys)), key=data.sort_keys.__getitem__)
        data.sort_keys = []

        num_minibatches = self.batch_size // bptt_horizon // batch_rows
        b_idxs = (
            torch.Tensor(idxs)
            .long()[:-1]
            .reshape(batch_rows, num_minibatches, bptt_horizon)
            .transpose(0, 1)
        )

        # bootstrap value if not done
        ## TO CHANGE
        with torch.no_grad():
            advantages = torch.zeros(self.batch_size, device=self.device)
            lastgaelam = 0
            for t in reversed(range(self.batch_size)):
                i, i_nxt = idxs[t], idxs[t + 1]
                nextnonterminal = 1.0 - data.dones[i_nxt]

                #nextvalues = data.values[i_nxt]
                # delta = (
                #     data.rewards[i_nxt]
                #     #+ gamma * nextvalues * nextnonterminal
                #     - data.values[i]
                # )
                # advantages[t] = lastgaelam = (
                #     delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                # )

        # Flatten the batch
        data.b_obs = b_obs = data.obs[b_idxs]
        b_actions = data.actions[b_idxs]
        #b_logprobs = data.logprobs[b_idxs]
        b_dones = data.dones[b_idxs]
        b_values = data.values[b_idxs]
        b_advantages = advantages.reshape(
            batch_rows, num_minibatches, bptt_horizon
        ).transpose(0, 1)
        b_returns = b_advantages + b_values

        # Optimizing the policy and value network
        train_time = time.time()
        clipfracs = []
        for epoch in range(update_epochs):
            lstm_state = None
            for mb in range(num_minibatches):
                mb_obs = b_obs[mb].to(self.device)
                mb_actions = b_actions[mb].contiguous()
                mb_values = b_values[mb].reshape(-1)
                mb_advantages = b_advantages[mb].reshape(-1)
                mb_returns = b_returns[mb].reshape(-1)
            
                replay_buffer.add(mb_obs,mb_actions,mb_values,mb_advantages,mb_returns)
                if len(replay_buffer) > batch_cff * batch_rows * bptt_horizon:
                    # sample from batchs
                    mb_obs,mb_actions,mb_values,mb_advantages,mb_returns,mb_actions = replay_buffer.sample(1)
                    mb_obs = mb_obs.to(self.device)
                    mb_actions = mb_actions.contiguous()
                    mb_values = mb_values.reshape(-1)
                    mb_advantages = mb_advantages.reshape(-1)
                    mb_returns = mb_returns.reshape(-1)

                if self.agent.is_recurrent:
                    (
                        _,
                        newlogprob,
                        entropy,
                        newvalue,
                        lstm_state,
                    ) = self.agent.get_action_and_value(
                        mb_obs, state=lstm_state, done=b_dones[mb], action=mb_actions
                    )
                    lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                else:
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        mb_obs.reshape(
                            -1, *self.buffers[0].single_observation_space.shape
                        ),
                        action=mb_actions,
                    )

                # logratio = newlogprob - b_logprobs[mb].reshape(-1)
                # ratio = logratio.exp()

                # with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    # approx_kl = ((ratio - 1) - logratio).mean()
                    # clipfracs += [
                    #     ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    # ]

                mb_advantages = mb_advantages.reshape(-1)
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                # pg_loss1 = -mb_advantages * ratio
                # pg_loss2 = -mb_advantages * torch.clamp(
                #     ratio, 1 - clip_coef, 1 + clip_coef
                # )
                pg_loss = 0#torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                if self.record_loss:
                    with open(self.loss_file, "a") as f:
                        print(f"# mini batch ({epoch}, {mb}) -- pg_loss:{pg_loss.item():.4f}, value_loss:{v_loss.item():.4f}, " + \
                              f"entropy:{entropy_loss.item():.4f}, approx_kl: {approx_kl.item():.4f}",
                                file=f)
                    with open(self.action_file, "a") as f:
                        print(f"# mini batch ({epoch}, {mb}) -- pg_loss:{pg_loss.item():.4f}, value_loss:{v_loss.item():.4f}, " + \
                              f"entropy:{entropy_loss.item():.4f}, approx_kl: {approx_kl.item():.4f}",
                                file=f)
                        atn_list = mb_actions.cpu().numpy().tolist()
                        for atns in atn_list:
                            for atn in atns:
                                print(f"{atn}", file=f)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
                self.optimizer.step()

            # if target_kl is not None:
            #     if approx_kl > target_kl:
            #         break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TIMING: performance metrics to evaluate cpu/gpu usage
        train_time = time.time() - train_time
        train_sps = int(self.batch_size / train_time)
        self.update += 1

        print(f"\tTrain={train_sps}\n")

        allocated_torch = torch.cuda.memory_allocated(self.device) - allocated_torch
        allocated_cpu = self.process.memory_info().rss - allocated_cpu
        if self.verbose:
            print(
                "Allocated during training - Pytorch: %.2f GB, System: %.2f GB"
                % (allocated_torch / 1e9, allocated_cpu / 1e9)
            )

        if self.record_loss:
            with open(self.loss_file, "a") as f:
                print(f"Epoch -- policy_loss:{pg_loss.item():.4f}, value_loss:{v_loss.item():.4f}, ",
                      f"entropy:{entropy_loss.item():.4f}, approx_kl:{approx_kl.item():.4f}",
                      f"clipfrac:{np.mean(clipfracs):.4f}, explained_var:{explained_var:.4f}",
                      file=f)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if self.wandb_entity:
            wandb.log(
                {
                    "performance/train_sps": train_sps,
                    "performance/train_time": train_time,
                    "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "agent_steps": self.global_step,
                    "global_step": self.global_step,
                }
            )

        if self.update % self.checkpoint_interval == 1 or self.done_training():
           self._save_checkpoint()

    @pufferlib.utils.profile
    def train(
        self,
        num_episodes,
        max_steps_per_episode,
        batch_size,
        gamma=0.99,  # Discount factor for future rewards
        target_update_freq=10,  # How often to update the target network
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995):
        epsilon = epsilon_start
        print("DQN now")
        for episode in range(num_episodes):
            state = self.buffers[0].reset()# buf idx
            episode_rewards = 0
            
            for step in range(max_steps_per_episode):
                action = [self.agent.select_action(state[agent_id], self.epsilon) for agent_id in range(self.num_agents)]
                self.buffers[0].send(action)
                next_state, reward, done, _ = self.buffers[0].recv()

                for agent_id in range(self.num_agents):
                    self.agent.replay_buffer.add(state[agent_id], action[agent_id], reward[agent_id], next_state[agent_id], done[agent_id])

                    if len(self.agent.replay_buffer) > batch_size:
                        transitions = self.agent.replay_buffer.sample(batch_size)
                        self.agent.optimize_model(transitions, gamma)

                    state = next_state  # Update the state

                    if done[agent_id]:
                        break
            # Train the policy network using a batch of experiences from the replay buffer
            if episode % target_update_freq == 0:
                self.agent.update_target_net()
            
            # Update epsilon
            epsilon = max(epsilon_end, epsilon_decay * epsilon)
                
            # Print the total rewards obtained in the episode
            print(f"Episode {episode}: Total Reward = {episode_rewards}")
            
            # Update the target network with weights from the policy network
            if self.wandb_entity:
                wandb.log({
                "episode": episode,
                "epsilon": epsilon,
                "episode_reward": episode_rewards
            })

    def done_training(self):
        return self.update >= self.total_updates

    def close(self):
        for envs in self.buffers:
            envs.close()

        if self.wandb_entity:
            wandb.finish()

    def _save_checkpoint(self):
        if self.data_dir is None:
            return

        policy_name = f"{self.exp_name}.{self.update:06d}"
        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "agent_step": self.agent_step,
            "update": self.update,
            "learning_rate": self.learning_rate,
            "policy_checkpoint_name": policy_name,
            "wandb_run_id": self.wandb_run_id,
        }
        path = os.path.join(self.data_dir, f"trainer.pt")
        tmp_path = path + ".tmp"
        torch.save(state, tmp_path)
        os.rename(tmp_path, path)

        # NOTE: as the agent_creator has args internally, the policy args are not passed
        self.policy_store.add_policy(policy_name, self.agent)

        if self.policy_ranker:
            self.policy_ranker.add_policy_copy(
                policy_name, self.policy_pool._learner_name
            )
