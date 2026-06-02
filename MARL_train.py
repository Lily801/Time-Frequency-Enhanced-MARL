import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import random
from typing import Dict, List, Tuple, Any, Optional
import json
import warnings
import time
import pickle
from matplotlib.colors import to_rgba
import re
warnings.filterwarnings('ignore')

# 环境设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# 导入环境
from andes_marl import (

    make_matd3_env,
    make_mappo_env,
    make_masac_env,
    make_wse_matd3_env,
    make_nsga2_matd3_mix_env
)

# 定义每个图片的生成函数
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
GPU = True
DEVICE_IDX = 0
if GPU and torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{DEVICE_IDX}")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

def save_config_to_json(path, env_config, algo_config):
    """保存配置到JSON文件 - 修复版本"""

    def convert_config_for_json(config):
        """将配置转换为可JSON序列化的格式 - 修复版本"""

        def _convert(obj):
            # 处理 torch.device
            if isinstance(obj, torch.device):
                return str(obj)
            # 处理 torch.dtype
            elif isinstance(obj, torch.dtype):
                return str(obj)
            # 处理 torch.nn.Module
            elif isinstance(obj, torch.nn.Module):
                return f"torch.nn.Module({type(obj).__name__})"
            # 处理 torch.Tensor（移动到CPU并转换为列表）
            elif isinstance(obj, torch.Tensor):
                # 如果张量在GPU上，移动到CPU
                if obj.is_cuda:
                    obj = obj.cpu()
                # 转换为Python列表
                return obj.tolist()
            # 处理字典
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            # 处理列表
            elif isinstance(obj, list):
                return [_convert(item) for item in obj]
            # 处理元组
            elif isinstance(obj, tuple):
                return tuple(_convert(item) for item in obj)
            # 处理 numpy 数组
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # 处理 numpy 标量
            elif isinstance(obj, np.generic):
                return obj.item()
            # 处理无法序列化的类型
            elif hasattr(obj, '__dict__'):
                # 尝试转换为字典
                try:
                    return _convert(obj.__dict__)
                except:
                    return str(obj)
            # 基本类型直接返回
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                # 其他类型转换为字符串
                try:
                    return str(obj)
                except:
                    return f"Unserializable object of type {type(obj)}"

        return _convert(config)

    # 转换配置以便JSON序列化
    env_config_json = convert_config_for_json(env_config)
    algo_config_json = convert_config_for_json(algo_config)

    combined_config = {
        'env_config': env_config_json,
        'algo_config': algo_config_json
    }

    config_file = os.path.join(path, 'config.json')
    try:
        with open(config_file, 'w') as f:
            json.dump(combined_config, f, indent=2, default=str)
        print(f"配置保存成功: {config_file}")
    except Exception as e:
        print(f"保存配置失败: {e}")
        # 尝试简化保存
        simplified_config = {
            'env_config': {k: str(v) for k, v in env_config.items()},
            'algo_config': {k: str(v) for k, v in algo_config.items()}
        }
        with open(config_file, 'w') as f:
            json.dump(simplified_config, f, indent=2)


class ReplayBuffer:
    """经验回放缓冲区 - 修复版本"""

    def __init__(self, capacity, obs_dims, act_dims, n_agents):
        self.capacity = capacity
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.n_agents = n_agents

        # 为每个智能体创建独立的缓冲区
        self.obs_buffers = [np.zeros((capacity, obs_dims[i]), dtype=np.float32) for i in range(n_agents)]
        self.next_obs_buffers = [np.zeros((capacity, obs_dims[i]), dtype=np.float32) for i in range(n_agents)]
        self.act_buffers = [np.zeros((capacity, act_dims[i]), dtype=np.float32) for i in range(n_agents)]
        self.rew_buffers = [np.zeros((capacity, 1), dtype=np.float32) for i in range(n_agents)]
        self.done_buffers = [np.zeros((capacity, 1), dtype=np.float32) for i in range(n_agents)]

        self.pointer = 0
        self.size = 0

    def store(self, obs, actions, rewards, next_obs, dones):
        for i in range(self.n_agents):
            self.obs_buffers[i][self.pointer] = obs[i]
            self.act_buffers[i][self.pointer] = actions[i]
            self.rew_buffers[i][self.pointer] = rewards[i]
            self.next_obs_buffers[i][self.pointer] = next_obs[i]
            self.done_buffers[i][self.pointer] = dones[i]

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        batch = {
            'obs': [self.obs_buffers[i][indices] for i in range(self.n_agents)],
            'act': [self.act_buffers[i][indices] for i in range(self.n_agents)],
            'rew': [self.rew_buffers[i][indices] for i in range(self.n_agents)],
            'next_obs': [self.next_obs_buffers[i][indices] for i in range(self.n_agents)],
            'done': [self.done_buffers[i][indices] for i in range(self.n_agents)],
        }

        return batch


class ActorNetwork(nn.Module):
    """Actor网络"""

    def __init__(self, obs_dim, act_dim, hidden_dim=256, init_w=3e-3):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)

        # 参数初始化
        self.init_weights(init_w)

    def init_weights(self, init_w):
        # Xavier初始化
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

        # 最后一层使用较小的初始化
        nn.init.uniform_(self.fc3.weight, -init_w, init_w)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, obs):
        # 添加NaN检查
        if torch.isnan(obs).any():
            print("Warning: Actor input contains NaN")
            obs = torch.nan_to_num(obs, nan=0.0)

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) #* 0.5

        # 输出检查
        if torch.isnan(action).any():
            print("Warning: Actor output contains NaN")
            action = torch.nan_to_num(action, nan=0.0)
        return action

class CriticNetwork(nn.Module):
    """Critic网络"""

    def __init__(self, obs_dim, act_dim, hidden_dim=256, init_w=3e-3):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # 参数初始化
        self.init_weights(init_w)

    def init_weights(self, init_w):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

        nn.init.uniform_(self.fc3.weight, -init_w, init_w)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, obs, act):
        # NaN检查
        if torch.isnan(obs).any() or torch.isnan(act).any():
            print("Warning: Critic input contains NaN")
            obs = torch.nan_to_num(obs, nan=0.0)
            act = torch.nan_to_num(act, nan=0.0)

        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)

        if torch.isnan(q_value).any():
            print("Warning: Critic output contains NaN")
            q_value = torch.nan_to_num(q_value, nan=0.0)

        return q_value


class MultiAgentCriticNetwork(nn.Module):
    """多智能体Critic网络（集中式训练）"""

    def __init__(self, total_obs_dim, total_act_dim, hidden_dim=256):
        super().__init__()
        self.total_obs_dim = total_obs_dim
        self.total_act_dim = total_act_dim

        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, act):
        # obs和act应该已经拼接好了
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


# 2. MATD3 完整实现 - 修复版本
class MATD3:
    """MATD3算法完整实现（双Critic） - 修复版本"""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.n_agents = len(env.possible_agents)
        self.agent_names = env.possible_agents

        # 获取每个智能体的观测和动作维度
        self.obs_dims = [env.observation_spaces[agent].shape[0] for agent in self.agent_names]
        self.act_dims = [env.action_spaces[agent].shape[0] for agent in self.agent_names]

        print(f"MATD3初始化: {self.n_agents}个智能体")
        print(f"观测维度: {self.obs_dims}")
        print(f"动作维度: {self.act_dims}")

        # 计算Critic的总输入维度
        self.total_obs_dim = sum(self.obs_dims)
        self.total_act_dim = sum(self.act_dims)

        # 创建网络
        self.actors = [ActorNetwork(self.obs_dims[i], self.act_dims[i]).to(config['device'])
                       for i in range(self.n_agents)]
        self.actor_targets = [ActorNetwork(self.obs_dims[i], self.act_dims[i]).to(config['device'])
                              for i in range(self.n_agents)]

        # 双Critic网络
        self.critics1 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                         for _ in range(self.n_agents)]
        self.critics2 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                         for _ in range(self.n_agents)]
        self.critic_targets1 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                                for _ in range(self.n_agents)]
        self.critic_targets2 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                                for _ in range(self.n_agents)]

        # 复制参数
        for i in range(self.n_agents):
            self.actor_targets[i].load_state_dict(self.actors[i].state_dict())
            self.critic_targets1[i].load_state_dict(self.critics1[i].state_dict())
            self.critic_targets2[i].load_state_dict(self.critics2[i].state_dict())

        # 优化器
        self.actor_optimizers = [torch.optim.Adam(self.actors[i].parameters(),
                                                  lr=config['actor_lr'])
                                 for i in range(self.n_agents)]
        self.critic_optimizers1 = [torch.optim.Adam(self.critics1[i].parameters(),
                                                    lr=config['critic_lr'])
                                   for i in range(self.n_agents)]
        self.critic_optimizers2 = [torch.optim.Adam(self.critics2[i].parameters(),
                                                    lr=config['critic_lr'])
                                   for i in range(self.n_agents)]

        # 经验回放
        self.buffer = ReplayBuffer(config['buffer_size'], self.obs_dims, self.act_dims, self.n_agents)

        # 训练参数
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.noise_std = config['noise_std']
        self.noise_clip = config['noise_clip']
        self.policy_noise = config['policy_noise']
        self.policy_freq = config['policy_freq']
        self.batch_size = config['batch_size']
        self.device = config['device']

        self.total_it = 0

        # 训练记录
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.noise_std_initial = self.noise_std  # 保存初始噪声标准差，用于退火

    def select_action(self, obs, explore=True):
        """选择动作"""
        actions = {}
        for i, agent in enumerate(self.agent_names):
            obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0).to(self.device)
            action = self.actors[i](obs_tensor).cpu().detach().numpy()[0]

            if explore:
                noise = np.random.normal(0, self.noise_std, size=action.shape)
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
                action = action + noise

            action = np.clip(action, -1.0, 1.0)
            actions[agent] = action

        return actions

    def update(self):
        """更新网络参数 - 移除NaN清洗，改为显式报错"""
        self.total_it += 1

        if self.buffer.size < self.batch_size:
            return 0.0, 0.0

        batch = self.buffer.sample(self.batch_size)

        # 转换为 tensor
        obs = [torch.FloatTensor(batch['obs'][i]).to(self.device) for i in range(self.n_agents)]
        act = [torch.FloatTensor(batch['act'][i]).to(self.device) for i in range(self.n_agents)]
        rew = [torch.FloatTensor(batch['rew'][i]).to(self.device) for i in range(self.n_agents)]
        next_obs = [torch.FloatTensor(batch['next_obs'][i]).to(self.device) for i in range(self.n_agents)]
        done = [torch.FloatTensor(batch['done'][i]).to(self.device) for i in range(self.n_agents)]

        # 输入张量有限性检查（已有，保留）
        for i in range(self.n_agents):
            if not torch.all(torch.isfinite(obs[i])):
                raise ValueError(f"[MATD3] 观测张量包含非有限值，智能体索引 {i}，观测值: {obs[i].cpu().numpy()}")
            if not torch.all(torch.isfinite(act[i])):
                raise ValueError(f"[MATD3] 动作张量包含非有限值，智能体索引 {i}，动作值: {act[i].cpu().numpy()}")
            if not torch.all(torch.isfinite(rew[i])):
                raise ValueError(f"[MATD3] 奖励张量包含非有限值，智能体索引 {i}，奖励值: {rew[i].cpu().numpy()}")
            if not torch.all(torch.isfinite(next_obs[i])):
                raise ValueError(
                    f"[MATD3] 下一观测张量包含非有限值，智能体索引 {i}，下一观测值: {next_obs[i].cpu().numpy()}")
            if not torch.all(torch.isfinite(done[i])):
                raise ValueError(f"[MATD3] 终止标志张量包含非有限值，智能体索引 {i}，终止值: {done[i].cpu().numpy()}")

        # 拼接所有智能体的观测和动作
        obs_cat = torch.cat(obs, dim=1)
        act_cat = torch.cat(act, dim=1)
        next_obs_cat = torch.cat(next_obs, dim=1)

        # 计算Critic损失
        critic_losses1 = []
        critic_losses2 = []

        for i in range(self.n_agents):
            with torch.no_grad():
                # 目标网络的下一个动作
                next_actions = []
                for j in range(self.n_agents):
                    next_action = self.actor_targets[j](next_obs[j])
                    # 添加噪声
                    noise = torch.randn_like(next_action) * self.policy_noise
                    noise = noise.clamp(-self.noise_clip, self.noise_clip)
                    next_action = (next_action + noise).clamp(-1.0, 1.0)

                    # 【新增】检查目标动作的有限性
                    if not torch.all(torch.isfinite(next_action)):
                        raise ValueError(
                            f"[MATD3] 目标动作包含非有限值，智能体索引 {j}，动作值: {next_action.cpu().numpy()}")

                    next_actions.append(next_action)

                next_act_cat = torch.cat(next_actions, dim=1)

                # 计算目标Q值
                target_q1 = self.critic_targets1[i](next_obs_cat, next_act_cat)
                target_q2 = self.critic_targets2[i](next_obs_cat, next_act_cat)

                # 【新增】检查目标Q值的有限性
                if not torch.all(torch.isfinite(target_q1)):
                    raise ValueError(f"[MATD3] 目标Q1包含非有限值，智能体索引 {i}，Q值: {target_q1.cpu().numpy()}")
                if not torch.all(torch.isfinite(target_q2)):
                    raise ValueError(f"[MATD3] 目标Q2包含非有限值，智能体索引 {i}，Q值: {target_q2.cpu().numpy()}")

                target_q = torch.min(target_q1, target_q2)
                target_q = rew[i] + (1 - done[i]) * self.gamma * target_q

                # 【新增】检查最终目标值的有限性
                if not torch.all(torch.isfinite(target_q)):
                    raise ValueError(
                        f"[MATD3] 目标值（reward+gamma*Q）包含非有限值，智能体索引 {i}，值: {target_q.cpu().numpy()}")

            # 当前Q值
            current_q1 = self.critics1[i](obs_cat, act_cat)
            current_q2 = self.critics2[i](obs_cat, act_cat)

            # 【新增】检查当前Q值的有限性
            if not torch.all(torch.isfinite(current_q1)):
                raise ValueError(f"[MATD3] 当前Q1包含非有限值，智能体索引 {i}，Q值: {current_q1.cpu().numpy()}")
            if not torch.all(torch.isfinite(current_q2)):
                raise ValueError(f"[MATD3] 当前Q2包含非有限值，智能体索引 {i}，Q值: {current_q2.cpu().numpy()}")

            # Critic损失
            critic_loss1 = F.mse_loss(current_q1, target_q)
            critic_loss2 = F.mse_loss(current_q2, target_q)

            critic_losses1.append(critic_loss1)
            critic_losses2.append(critic_loss2)

        # 更新所有Critic
        total_critic_loss = 0.0
        for i in range(self.n_agents):
            self.critic_optimizers1[i].zero_grad()
            critic_losses1[i].backward(retain_graph=True if i < self.n_agents - 1 else False)
            torch.nn.utils.clip_grad_norm_(self.critics1[i].parameters(), 0.5)
            self.critic_optimizers1[i].step()

            self.critic_optimizers2[i].zero_grad()
            critic_losses2[i].backward()
            torch.nn.utils.clip_grad_norm_(self.critics2[i].parameters(), 0.5)
            self.critic_optimizers2[i].step()

            total_critic_loss += (critic_losses1[i].item() + critic_losses2[i].item()) / 2

        # 延迟策略更新
        actor_loss = 0.0
        if self.total_it % self.policy_freq == 0:
            # 计算所有Actor的损失
            actor_losses = []

            # 首先计算所有Actor的当前动作
            current_actions = []
            for j in range(self.n_agents):
                current_action = self.actors[j](obs[j])

                # 【新增】检查当前动作的有限性
                if not torch.all(torch.isfinite(current_action)):
                    raise ValueError(
                        f"[MATD3] 当前动作包含非有限值，智能体索引 {j}，动作值: {current_action.cpu().numpy()}")

                current_actions.append(current_action)
            current_act_cat = torch.cat(current_actions, dim=1)

            for i in range(self.n_agents):
                q_value = self.critics1[i](obs_cat, current_act_cat)

                # 【新增】检查Q值的有限性
                if not torch.all(torch.isfinite(q_value)):
                    raise ValueError(
                        f"[MATD3] Actor损失计算中的Q值包含非有限值，智能体索引 {i}，Q值: {q_value.cpu().numpy()}")

                actor_loss = -q_value.mean()
                actor_losses.append(actor_loss)

            # 更新所有Actor
            for i in range(self.n_agents):
                self.actor_optimizers[i].zero_grad()

            for i in range(self.n_agents):
                actor_losses[i].backward(retain_graph=True if i < self.n_agents - 1 else False)
                actor_loss += actor_losses[i].item()

            for i in range(self.n_agents):
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                self.actor_optimizers[i].step()

            # 软更新目标网络
            for i in range(self.n_agents):
                for param, target_param in zip(self.critics1[i].parameters(), self.critic_targets1[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critics2[i].parameters(), self.critic_targets2[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actors[i].parameters(), self.actor_targets[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return float(actor_loss / self.n_agents if self.total_it % self.policy_freq == 0 else 0.0), float(
            total_critic_loss / self.n_agents)

    def train(self, n_episodes, save_path):
        """训练MATD3算法 - 修复数据保存问题"""

        os.makedirs(save_path, exist_ok=True)

        # 训练数据记录
        data = {
            'episode': [],
            'reward': [],
            'success': [],
            'avg_damping_ratio': [],
            'avg_oscillation_freq': [],
            'freq_deviation': [],
            'voltage_deviation': [],
            'actor_loss': [],
            'critic_loss': [],
            'action_magnitude': []
        }

        for episode in range(n_episodes):
            # 【新增】噪声退火：随 episode 增加线性减小噪声
            progress = episode / max(1, n_episodes - 1)
            self.noise_std = max(0.01, self.noise_std_initial * (1 - progress))

            obs, info = self.env.reset()
            episode_reward = {agent: 0.0 for agent in self.agent_names}
            episode_success = False

            # 收集数据
            freq_deviations = []
            voltage_deviations = []
            damping_ratios = []
            osc_freqs = []
            action_magnitudes = []
            episode_actor_loss = 0.0
            episode_critic_loss = 0.0
            update_count = 0

            step = 0
            while step < self.env.max_steps:
                try:
                    actions = self.select_action(obs, explore=True)
                    next_obs, rewards, dones, truncs, infos = self.env.step(actions)

                    # 存储经验
                    obs_list = [obs[agent] for agent in self.agent_names]
                    next_obs_list = [next_obs[agent] for agent in self.agent_names]
                    act_list = [actions[agent] for agent in self.agent_names]
                    rew_list = [rewards[agent] for agent in self.agent_names]
                    done_list = [dones[agent] for agent in self.agent_names]

                    self.buffer.store(obs_list, act_list, rew_list, next_obs_list, done_list)

                    for agent in self.agent_names:
                        episode_reward[agent] += rewards[agent]

                    # 收集系统状态数据
                    if hasattr(self.env, 'wcoi_idx'):
                        try:
                            wcoi = self.env.sim_case.dae.y[self.env.wcoi_idx].astype(np.float32)
                            if len(wcoi) > 0:
                                freq_dev = np.mean(np.abs(wcoi - 1.0))
                                freq_deviations.append(freq_dev)
                        except:
                            pass

                    if hasattr(self.env, 'voltage_idx'):
                        try:
                            voltage = self.env.sim_case.dae.y[self.env.voltage_idx].astype(np.float32)
                            if len(voltage) > 0:
                                volt_dev = np.mean(np.abs(voltage - 1.0))
                                voltage_deviations.append(volt_dev)
                        except:
                            pass

                    # 收集相域信息
                    if hasattr(self.env, 'prony_analyzers'):
                        for agent in self.agent_names:
                            analyzer = self.env.prony_analyzers[agent]
                            if analyzer and analyzer.last_analysis:
                                damping_ratios.append(analyzer.last_analysis['damping_ratio'])
                                osc_freqs.append(analyzer.last_analysis['oscillation_freq'])

                    # 收集动作幅度
                    for agent in self.agent_names:
                        if agent in actions:
                            action_magnitudes.append(np.mean(np.abs(actions[agent])))

                    # 更新网络
                    if self.buffer.size >= self.batch_size:
                        actor_loss, critic_loss = self.update()
                        episode_actor_loss += actor_loss
                        episode_critic_loss += critic_loss
                        update_count += 1

                    obs = next_obs
                    step += 1

                    if all(dones.values()) or all(truncs.values()):
                        episode_success = not any(infos[agent].get('sim_crashed', False) for agent in self.agent_names)
                        break

                except Exception as e:
                    print(f"Episode {episode}, Step {step} 出错: {e}")
                    break

            # 计算统计数据
            total_reward = sum(episode_reward.values())
            self.episode_rewards.append(total_reward)

            avg_freq_dev = np.mean(freq_deviations) if freq_deviations else 0.0
            avg_volt_dev = np.mean(voltage_deviations) if voltage_deviations else 0.0
            avg_damping = np.mean(damping_ratios) if damping_ratios else 0.0
            avg_osc_freq = np.mean(osc_freqs) if osc_freqs else 0.0
            avg_action_mag = np.mean(action_magnitudes) if action_magnitudes else 0.0
            avg_actor_loss = episode_actor_loss / max(1, update_count)
            avg_critic_loss = episode_critic_loss / max(1, update_count)

            self.actor_losses.append(avg_actor_loss)
            self.critic_losses.append(avg_critic_loss)

            # ===== 修复：确保所有数据都是Python标量 =====
            data['episode'].append(int(episode))
            data['reward'].append(float(total_reward))
            data['success'].append(int(episode_success))
            data['avg_damping_ratio'].append(float(avg_damping))
            data['avg_oscillation_freq'].append(float(avg_osc_freq))
            data['freq_deviation'].append(float(avg_freq_dev))
            data['voltage_deviation'].append(float(avg_volt_dev))
            data['actor_loss'].append(float(avg_actor_loss))
            data['critic_loss'].append(float(avg_critic_loss))
            data['action_magnitude'].append(float(avg_action_mag))

            # 定期保存
            if episode % 10 == 0 or episode == n_episodes - 1:
                try:
                    # 保存训练数据
                    df = pd.DataFrame({
                        'episode': [int(x) for x in data['episode']],
                        'reward': [float(x) for x in data['reward']],
                        'success': [int(x) for x in data['success']],
                        'avg_damping_ratio': [float(x) for x in data['avg_damping_ratio']],
                        'avg_oscillation_freq': [float(x) for x in data['avg_oscillation_freq']],
                        'freq_deviation': [float(x) for x in data['freq_deviation']],
                        'voltage_deviation': [float(x) for x in data['voltage_deviation']],
                        'actor_loss': [float(x) for x in data['actor_loss']],
                        'critic_loss': [float(x) for x in data['critic_loss']],
                        'action_magnitude': [float(x) for x in data['action_magnitude']]
                    })
                    df.to_csv(os.path.join(save_path, 'training_data.csv'), index=False)

                    # 保存模型
                    self._save_model(save_path, episode)

                    print(f"MATD3 - Episode {episode}, "
                          f"Reward: {total_reward:.2f}, "
                          f"Success: {episode_success}, "
                          f"Damping: {avg_damping:.4f}, "
                          f"Actor Loss: {avg_actor_loss:.4f}, "
                          f"Critic Loss: {avg_critic_loss:.4f}")

                except Exception as e:
                    print(f"保存数据失败: {e}")

        # 最终保存
        try:
            self._save_model(save_path, 'final')

            # 最终保存训练数据
            df = pd.DataFrame({
                'episode': [int(x) for x in data['episode']],
                'reward': [float(x) for x in data['reward']],
                'success': [int(x) for x in data['success']],
                'avg_damping_ratio': [float(x) for x in data['avg_damping_ratio']],
                'avg_oscillation_freq': [float(x) for x in data['avg_oscillation_freq']],
                'freq_deviation': [float(x) for x in data['freq_deviation']],
                'voltage_deviation': [float(x) for x in data['voltage_deviation']],
                'actor_loss': [float(x) for x in data['actor_loss']],
                'critic_loss': [float(x) for x in data['critic_loss']],
                'action_magnitude': [float(x) for x in data['action_magnitude']]
            })
            df.to_csv(os.path.join(save_path, 'training_data.csv'), index=False)

            # 保存配置
            self._save_config(save_path)

            print(f"MATD3训练完成！数据保存到: {save_path}")

        except Exception as e:
            print(f"最终保存失败: {e}")

        return data

    def _save_model(self, path, episode):
        """保存模型 - 修复版本"""
        try:
            model_dict = {
                'actors': [self.actors[i].state_dict() for i in range(self.n_agents)],
                'critics1': [self.critics1[i].state_dict() for i in range(self.n_agents)],
                'critics2': [self.critics2[i].state_dict() for i in range(self.n_agents)],
                'actor_targets': [self.actor_targets[i].state_dict() for i in range(self.n_agents)],
                'critic_targets1': [self.critic_targets1[i].state_dict() for i in range(self.n_agents)],
                'critic_targets2': [self.critic_targets2[i].state_dict() for i in range(self.n_agents)],
                'actor_optimizers': [self.actor_optimizers[i].state_dict() for i in range(self.n_agents)],
                'critic_optimizers1': [self.critic_optimizers1[i].state_dict() for i in range(self.n_agents)],
                'critic_optimizers2': [self.critic_optimizers2[i].state_dict() for i in range(self.n_agents)],
                'obs_dims': self.obs_dims,
                'act_dims': self.act_dims,
                'total_obs_dim': self.total_obs_dim,
                'total_act_dim': self.total_act_dim,
                'agent_names': self.agent_names,
                'episode': episode,
                'episode_rewards': [float(x) for x in self.episode_rewards],
                'actor_losses': [float(x) for x in self.actor_losses],
                'critic_losses': [float(x) for x in self.critic_losses],
                'config': {
                    'buffer_size': self.config.get('buffer_size'),
                    'batch_size': self.config.get('batch_size'),
                    'actor_lr': float(self.config.get('actor_lr', 0.0)),
                    'critic_lr': float(self.config.get('critic_lr', 0.0)),
                    'gamma': float(self.config.get('gamma', 0.0)),
                    'tau': float(self.config.get('tau', 0.0)),
                    'noise_std': float(self.config.get('noise_std', 0.0)),
                    'noise_clip': float(self.config.get('noise_clip', 0.0)),
                    'policy_noise': float(self.config.get('policy_noise', 0.0)),
                    'policy_freq': int(self.config.get('policy_freq', 0)),
                    'device': str(self.config.get('device', 'cpu'))
                }
            }

            model_path = os.path.join(path, f'model_{episode}.pth')
            torch.save(model_dict, model_path)
            print(f"模型保存成功: {model_path}")

        except Exception as e:
            print(f"保存模型失败: {e}")

    def _save_config(self, path):
        """保存配置 - 修复版本"""
        try:
            env_config = {
                'tf': float(self.env.tf) if hasattr(self.env, 'tf') else 15.0,
                'tstep': float(self.env.tstep) if hasattr(self.env, 'tstep') else 1 / 30,
                'max_steps': int(self.env.max_steps) if hasattr(self.env, 'max_steps') else 28,
                'include_prony': bool(self.env.include_prony) if hasattr(self.env, 'include_prony') else True,
                'shared_observations': bool(self.env.shared_observations) if hasattr(self.env,
                                                                                     'shared_observations') else True,
                'prony_coordination': bool(self.env.prony_coordination) if hasattr(self.env,
                                                                                   'prony_coordination') else True,
                'algorithm_type': str(self.env.algorithm_type) if hasattr(self.env, 'algorithm_type') else 'MATD3'
            }

            # 简化算法配置
            algo_config = {
                'buffer_size': int(self.config.get('buffer_size', 100000)),
                'batch_size': int(self.config.get('batch_size', 256)),
                'actor_lr': float(self.config.get('actor_lr', 1e-4)),
                'critic_lr': float(self.config.get('critic_lr', 1e-3)),
                'gamma': float(self.config.get('gamma', 0.99)),
                'tau': float(self.config.get('tau', 0.01)),
                'noise_std': float(self.config.get('noise_std', 0.1)),
                'noise_clip': float(self.config.get('noise_clip', 0.5)),
                'policy_noise': float(self.config.get('policy_noise', 0.2)),
                'policy_freq': int(self.config.get('policy_freq', 2)),
                'device': str(self.config.get('device', 'cpu'))
            }

            save_config_to_json(path, env_config, algo_config)

        except Exception as e:
            print(f"保存配置失败: {e}")

class MAPPO:
    """MAPPO算法完整实现 - 修复版本"""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.n_agents = len(env.possible_agents)
        self.agent_names = env.possible_agents

        # 获取每个智能体的观测和动作维度
        self.obs_dims = [env.observation_spaces[agent].shape[0] for agent in self.agent_names]
        self.act_dims = [env.action_spaces[agent].shape[0] for agent in self.agent_names]

        print(f"MAPPO初始化: {self.n_agents}个智能体")
        print(f"观测维度: {self.obs_dims}")
        print(f"动作维度: {self.act_dims}")

        # Actor网络
        self.actors = [self._create_actor(self.obs_dims[i], self.act_dims[i]).to(config['device'])
                       for i in range(self.n_agents)]

        # 计算Critic的总输入维度
        self.total_obs_dim = sum(self.obs_dims)
        self.total_act_dim = sum(self.act_dims)

        # Critic网络（集中式）
        self.critic = MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])

        # 优化器
        self.actor_optimizers = [torch.optim.Adam(self.actors[i].parameters(),
                                                  lr=config['actor_lr'])
                                 for i in range(self.n_agents)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=config['critic_lr'])

        # PPO参数
        self.clip_epsilon = config['clip_epsilon']
        self.entropy_coef = config['entropy_coef']
        self.value_coef = config['value_coef']
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.ppo_epochs = config['ppo_epochs']
        self.batch_size = config['batch_size']
        self.device = config['device']

        # 训练记录
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
    def _create_actor(self, obs_dim, act_dim):
        """创建Actor网络 - 修复版本"""

        class StochasticActor(nn.Module):
            def __init__(self, obs_dim, act_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, 256),
                    nn.Tanh(),
                    nn.Linear(256, 256),
                    nn.Tanh(),
                )
                self.mean_layer = nn.Linear(256, act_dim)
                self.log_std_layer = nn.Linear(256, act_dim)

                # 初始化
                for layer in self.net:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=0.01)
                        nn.init.constant_(layer.bias, 0.0)
                nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
                nn.init.constant_(self.mean_layer.bias, 0.0)
                nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
                nn.init.constant_(self.log_std_layer.bias, 0.0)

            def forward(self, x):
                x = self.net(x)
                mean = self.mean_layer(x)
                log_std = self.log_std_layer(x)
                log_std = torch.clamp(log_std, -20, 2)
                std = torch.exp(log_std)
                return mean, std

            def sample(self, x):
                mean, std = self.forward(x)
                normal = Normal(mean, std)
                action = normal.rsample()  # 重参数化
                log_prob = normal.log_prob(action).sum(-1, keepdim=True)

                # 限制动作范围 [-0.5, 0.5]
                action = torch.tanh(action) * 0.5

                # 修正log概率
                log_prob -= torch.log(1 - torch.tanh(action).pow(2) + 1e-6).sum(-1, keepdim=True)

                return action, log_prob

            def evaluate(self, x, action):
                mean, std = self.forward(x)
                normal = Normal(mean, std)

                # 将动作转换回原始空间
                raw_action = torch.atanh(torch.clamp(action / 0.5, -0.99, 0.99))
                log_prob = normal.log_prob(raw_action).sum(-1, keepdim=True)
                entropy = normal.entropy().sum(-1, keepdim=True)

                return log_prob, entropy

        return StochasticActor(obs_dim, act_dim)

    def select_action(self, obs, explore=True):
        """选择动作 - 修复版本"""
        actions = {}
        log_probs = {}
        values = {}

        obs_list = []
        for agent in self.agent_names:
            obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0).to(self.device)
            obs_list.append(obs_tensor)

        # 拼接所有智能体的观测作为Critic输入
        obs_cat = torch.cat(obs_list, dim=1)

        with torch.no_grad():
            # 计算集中式Critic值
            dummy_actions = [torch.zeros(1, self.env.action_spaces[agent].shape[0]).to(self.device)
                             for agent in self.agent_names]
            dummy_act_cat = torch.cat(dummy_actions, dim=1)
            value = self.critic(obs_cat, dummy_act_cat).cpu().numpy()[0]

            for i, agent in enumerate(self.agent_names):
                if explore:
                    action, log_prob = self.actors[i].sample(obs_list[i])
                    action = action.cpu().numpy()[0]
                    log_prob = log_prob.cpu().numpy()[0]
                else:
                    # 确定性策略
                    mean, std = self.actors[i](obs_list[i])
                    action = mean.cpu().numpy()[0]
                    log_prob = 0.0

                actions[agent] = action
                log_probs[agent] = log_prob
                values[agent] = value

        return actions, log_probs, values

    def compute_gae(self, rewards, values, dones, next_value):
        """计算GAE"""
        gae = 0
        advantages = []

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_values = next_value
            else:
                next_values = values[t + 1]

            delta = rewards[t] + self.gamma * next_values * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return np.array(advantages)

    def train_epoch(self, samples):
        """训练一个epoch - 修复版本（正确处理随机Actor输出）"""
        obs_list = samples['obs']
        actions_list = samples['actions']
        log_probs_list = samples['log_probs']
        advantages_list = samples['advantages']
        returns_list = samples['returns']

        if len(obs_list) == 0 or len(obs_list[0]) == 0:
            return 0.0, 0.0

        n_steps = len(obs_list[0])

        # 将数据展平
        for i in range(self.n_agents):
            if len(obs_list[i]) > 0:
                obs_list[i] = obs_list[i].reshape(-1, self.obs_dims[i])
                actions_list[i] = actions_list[i].reshape(-1, self.act_dims[i])
                log_probs_list[i] = log_probs_list[i].reshape(-1, 1)
                advantages_list[i] = advantages_list[i].reshape(-1, 1)
                returns_list[i] = returns_list[i].reshape(-1, 1)
            else:
                obs_list[i] = np.zeros((0, self.obs_dims[i]))
                actions_list[i] = np.zeros((0, self.act_dims[i]))
                log_probs_list[i] = np.zeros((0, 1))
                advantages_list[i] = np.zeros((0, 1))
                returns_list[i] = np.zeros((0, 1))

        total_samples = len(obs_list[0])
        if total_samples < 4:
            return 0.0, 0.0

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        # 训练多个epoch
        for epoch in range(self.ppo_epochs):
            indices = np.arange(total_samples)
            np.random.shuffle(indices)

            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                if end <= start:
                    continue

                batch_indices = indices[start:end]

                batch_obs = []
                batch_actions = []
                batch_log_probs_old = []
                batch_advantages = []
                batch_returns = []

                for i in range(self.n_agents):
                    obs_data = obs_list[i][batch_indices]
                    act_data = actions_list[i][batch_indices]
                    log_prob_data = log_probs_list[i][batch_indices]
                    adv_data = advantages_list[i][batch_indices]
                    ret_data = returns_list[i][batch_indices]

                    batch_obs.append(torch.FloatTensor(obs_data).to(self.device))
                    batch_actions.append(torch.FloatTensor(act_data).to(self.device))
                    batch_log_probs_old.append(torch.FloatTensor(log_prob_data).to(self.device))
                    batch_advantages.append(torch.FloatTensor(adv_data).to(self.device))
                    batch_returns.append(torch.FloatTensor(ret_data).to(self.device))

                batch_obs_cat = torch.cat(batch_obs, dim=1)
                batch_act_cat = torch.cat(batch_actions, dim=1)

                # 更新Critic
                values = self.critic(batch_obs_cat, batch_act_cat)
                if torch.isnan(values).any():
                    values = torch.nan_to_num(values, nan=0.0)
                critic_loss = F.mse_loss(values, batch_returns[0])  # 使用第一个智能体的returns（可根据需求调整）
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                total_critic_loss += critic_loss.item()

                # 更新每个Actor
                epoch_actor_loss = 0.0
                actor_losses = []

                for i in range(self.n_agents):
                    # 获取当前策略的均值和标准差
                    mean, log_std = self.actors[i](batch_obs[i])
                    # 检查NaN
                    if torch.isnan(mean).any():
                        mean = torch.nan_to_num(mean, nan=0.0)
                    if torch.isnan(log_std).any():
                        log_std = torch.nan_to_num(log_std, nan=0.0)
                    # 确保 log_std 形状与 mean 匹配
                    std = torch.exp(torch.clamp(log_std, -20, 2))

                    # 处理动作：将动作从[-0.5,0.5]转换回原始空间用于概率计算
                    scaled_actions = torch.clamp(batch_actions[i], -0.5 + 1e-6, 0.5 - 1e-6)
                    raw_actions = torch.atanh(scaled_actions * 2)  # 缩放到tanh域

                    # 计算新log概率
                    normal = Normal(mean, std)
                    new_log_prob = normal.log_prob(raw_actions).sum(-1, keepdim=True)

                    # 计算熵
                    entropy = normal.entropy().mean()

                    # 计算比率
                    ratio = torch.exp(new_log_prob - batch_log_probs_old[i])
                    ratio = torch.clamp(ratio, 0.1, 10.0)

                    # PPO损失
                    surr1 = ratio * batch_advantages[i]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages[i]
                    actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                    if torch.isnan(actor_loss):
                        actor_loss = torch.tensor(0.0, device=self.device)

                    actor_losses.append(actor_loss)

                # 更新所有Actor
                for i in range(self.n_agents):
                    self.actor_optimizers[i].zero_grad()
                for i in range(self.n_agents):
                    actor_losses[i].backward(retain_graph=(i < self.n_agents - 1))
                    epoch_actor_loss += actor_losses[i].item()
                for i in range(self.n_agents):
                    torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                    self.actor_optimizers[i].step()

                total_actor_loss += epoch_actor_loss / self.n_agents

        avg_actor_loss = total_actor_loss / self.ppo_epochs if self.ppo_epochs > 0 else 0.0
        avg_critic_loss = total_critic_loss / self.ppo_epochs if self.ppo_epochs > 0 else 0.0

        return float(avg_actor_loss), float(avg_critic_loss)

    def train(self, n_episodes, save_path):
        """训练MAPPO算法 - 修复奖励格式化错误并强制数据有效性检查"""

        os.makedirs(save_path, exist_ok=True)

        data = {
            'episode': [], 'reward': [], 'success': [],
            'avg_damping_ratio': [], 'avg_oscillation_freq': [],
            'freq_deviation': [], 'voltage_deviation': [],
            'actor_loss': [], 'critic_loss': [], 'action_magnitude': []
        }

        for episode in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = {agent: 0.0 for agent in self.agent_names}
            episode_success = False

            # 收集轨迹数据
            obs_trajectory = [[] for _ in range(self.n_agents)]
            action_trajectory = [[] for _ in range(self.n_agents)]
            reward_trajectory = [[] for _ in range(self.n_agents)]
            log_prob_trajectory = [[] for _ in range(self.n_agents)]
            value_trajectory = [[] for _ in range(self.n_agents)]
            done_trajectory = [[] for _ in range(self.n_agents)]

            # 收集指标数据
            freq_deviations = []
            voltage_deviations = []
            damping_ratios = []
            osc_freqs = []
            action_magnitudes = []

            step = 0
            while step < self.env.max_steps:
                try:
                    actions, log_probs, values = self.select_action(obs, explore=True)
                    next_obs, rewards, dones, truncs, infos = self.env.step(actions)

                    # ===== 关键修改：使用 enumerate 遍历智能体，并存储轨迹数据 =====
                    for i, agent in enumerate(self.agent_names):
                        reward_val = rewards[agent]
                        # 如果奖励是 NumPy 数组，提取标量
                        if isinstance(reward_val, np.ndarray):
                            if reward_val.size == 1:
                                reward_val = reward_val.item()
                            else:
                                raise ValueError(
                                    f"Reward for agent {agent} is an array with size {reward_val.size} (expected scalar)")
                        elif not np.isscalar(reward_val):
                            raise ValueError(f"Reward for agent {agent} is not scalar (type: {type(reward_val)})")
                        # 存储标量奖励
                        reward_trajectory[i].append(reward_val)
                        episode_reward[agent] += reward_val

                        # 存储其他轨迹数据
                        obs_trajectory[i].append(obs[agent].copy())
                        action_trajectory[i].append(actions[agent].copy())
                        log_prob_trajectory[i].append(log_probs[agent])
                        value_trajectory[i].append(values[agent])
                        done_trajectory[i].append(dones[agent])

                    # 收集系统状态数据（保持不变）
                    if not hasattr(self.env, 'wcoi_idx'):
                        raise RuntimeError("Environment missing 'wcoi_idx' attribute")
                    wcoi = self.env.sim_case.dae.y[self.env.wcoi_idx].astype(np.float32)
                    if len(wcoi) == 0:
                        raise RuntimeError("wcoi array is empty")
                    freq_dev = np.mean(np.abs(wcoi - 1.0))
                    if not np.isfinite(freq_dev):
                        raise RuntimeError(f"freq_dev is NaN/inf: {freq_dev}")
                    freq_deviations.append(freq_dev)

                    if not hasattr(self.env, 'voltage_idx'):
                        raise RuntimeError("Environment missing 'voltage_idx' attribute")
                    voltage = self.env.sim_case.dae.y[self.env.voltage_idx].astype(np.float32)
                    if len(voltage) == 0:
                        raise RuntimeError("voltage array is empty")
                    volt_dev = np.mean(np.abs(voltage - 1.0))
                    if not np.isfinite(volt_dev):
                        raise RuntimeError(f"volt_dev is NaN/inf: {volt_dev}")
                    voltage_deviations.append(volt_dev)

                    if not hasattr(self.env, 'prony_analyzers'):
                        raise RuntimeError("Environment missing 'prony_analyzers'")
                    for agent in self.agent_names:
                        analyzer = self.env.prony_analyzers.get(agent)
                        if analyzer is None or analyzer.last_analysis is None:
                            raise RuntimeError(f"Prony analyzer for agent {agent} not available")
                        res = analyzer.last_analysis
                        if not res.get('valid', False):
                            raise RuntimeError(f"Prony analysis invalid for agent {agent}")
                        damping_ratios.append(res['damping_ratio'])
                        osc_freqs.append(res['oscillation_freq'])

                    for agent in self.agent_names:
                        if agent in actions:
                            mag = np.mean(np.abs(actions[agent]))
                            if not np.isfinite(mag):
                                raise RuntimeError(f"Action magnitude for {agent} is NaN/inf")
                            action_magnitudes.append(mag)
                        else:
                            raise RuntimeError(f"Action missing for agent {agent}")

                    obs = next_obs
                    step += 1

                    if all(dones.values()) or all(truncs.values()):
                        episode_success = not any(infos[agent].get('sim_crashed', False) for agent in self.agent_names)
                        break

                except Exception as e:
                    raise RuntimeError(f"Episode {episode}, Step {step} 出错: {e}") from e

            # 计算总奖励（此时 episode_reward 中已全是标量）
            total_reward = float(sum(episode_reward.values()))
            self.episode_rewards.append(total_reward)

            # 计算GAE和returns（仅当有足够数据）
            actor_loss = 0.0
            critic_loss = 0.0
            if len(obs_trajectory[0]) > 0:
                try:
                    # 获取下一个状态的价值
                    with torch.no_grad():
                        next_obs_list = []
                        for agent in self.agent_names:
                            obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0).to(self.device)
                            next_obs_list.append(obs_tensor)
                        next_obs_cat = torch.cat(next_obs_list, dim=1)
                        dummy_actions = [torch.zeros(1, self.env.action_spaces[agent].shape[0]).to(self.device)
                                         for agent in self.agent_names]
                        dummy_act_cat = torch.cat(dummy_actions, dim=1)
                        next_value = self.critic(next_obs_cat, dummy_act_cat).cpu().numpy()[0]

                    # 准备训练数据
                    samples = {
                        'obs': [],
                        'actions': [],
                        'log_probs': [],
                        'advantages': [],
                        'returns': []
                    }

                    for i in range(self.n_agents):
                        if len(reward_trajectory[i]) > 0:
                            rewards = np.array(reward_trajectory[i])
                            values = np.array(value_trajectory[i])
                            dones = np.array(done_trajectory[i])

                            advantages = self.compute_gae(rewards, values, dones, next_value)
                            returns = advantages + values

                            samples['obs'].append(np.array(obs_trajectory[i]))
                            samples['actions'].append(np.array(action_trajectory[i]))
                            samples['log_probs'].append(np.array(log_prob_trajectory[i]))
                            samples['advantages'].append(advantages)
                            samples['returns'].append(returns)

                    # 训练网络
                    if len(samples['obs']) > 0:
                        actor_loss, critic_loss = self.train_epoch(samples)
                        self.actor_losses.append(actor_loss)
                        self.critic_losses.append(critic_loss)

                except Exception as e:
                    raise RuntimeError(f"训练过程出错: {e}") from e

            # 计算统计数据（所有值转为Python标量）
            avg_freq_dev = float(np.mean(freq_deviations)) if freq_deviations else 0.0
            avg_volt_dev = float(np.mean(voltage_deviations)) if voltage_deviations else 0.0
            avg_damping = float(np.mean(damping_ratios)) if damping_ratios else 0.0
            avg_osc_freq = float(np.mean(osc_freqs)) if osc_freqs else 0.0
            avg_action_mag = float(np.mean(action_magnitudes)) if action_magnitudes else 0.0
            actor_loss = float(actor_loss)
            critic_loss = float(critic_loss)

            # 记录数据
            data['episode'].append(episode)
            data['reward'].append(total_reward)
            data['success'].append(1 if episode_success else 0)
            data['avg_damping_ratio'].append(avg_damping)
            data['avg_oscillation_freq'].append(avg_osc_freq)
            data['freq_deviation'].append(avg_freq_dev)
            data['voltage_deviation'].append(avg_volt_dev)
            data['actor_loss'].append(actor_loss)
            data['critic_loss'].append(critic_loss)
            data['action_magnitude'].append(avg_action_mag)

            # 定期保存
            if episode % 10 == 0 or episode == n_episodes - 1:
                try:
                    df = pd.DataFrame(data)
                    df.to_csv(os.path.join(save_path, 'training_data.csv'), index=False)
                    self._save_model(save_path, episode)
                    print(
                        f"MAPPO - Episode {episode}, Reward: {total_reward:.2f}, Success: {episode_success}, Damping: {avg_damping:.4f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
                except Exception as e:
                    print(f"保存数据失败: {e}")

        # 最终保存
        try:
            self._save_model(save_path, 'final')
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(save_path, 'training_data.csv'), index=False)
            self._save_config(save_path)
            print(f"MAPPO训练完成！数据保存到: {save_path}")
        except Exception as e:
            print(f"最终保存失败: {e}")

        return data

    def _save_model(self, path, episode):
        """保存模型 - 修复版本"""
        try:
            model_dict = {
                'actors': [self.actors[i].state_dict() for i in range(self.n_agents)],
                'critic': self.critic.state_dict(),
                'actor_optimizers': [self.actor_optimizers[i].state_dict() for i in range(self.n_agents)],
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'obs_dims': self.obs_dims,
                'act_dims': self.act_dims,
                'total_obs_dim': self.total_obs_dim,
                'total_act_dim': self.total_act_dim,
                'agent_names': self.agent_names,
                'episode': episode,
                'episode_rewards': [float(x) for x in self.episode_rewards],
                'actor_losses': [float(x) for x in self.actor_losses],
                'critic_losses': [float(x) for x in self.critic_losses],
                'config': {
                    'actor_lr': float(self.config.get('actor_lr', 0.0)),
                    'critic_lr': float(self.config.get('critic_lr', 0.0)),
                    'gamma': float(self.config.get('gamma', 0.0)),
                    'gae_lambda': float(self.config.get('gae_lambda', 0.0)),
                    'clip_epsilon': float(self.config.get('clip_epsilon', 0.0)),
                    'ppo_epochs': int(self.config.get('ppo_epochs', 0)),
                    'batch_size': int(self.config.get('batch_size', 0)),
                    'entropy_coef': float(self.config.get('entropy_coef', 0.0)),
                    'value_coef': float(self.config.get('value_coef', 0.0)),
                    'device': str(self.config.get('device', 'cpu'))
                }
            }

            model_path = os.path.join(path, f'model_{episode}.pth')
            torch.save(model_dict, model_path)
            print(f"模型保存成功: {model_path}")

        except Exception as e:
            print(f"保存模型失败: {e}")

    def _save_config(self, path):
        """保存配置 - 修复版本"""
        try:
            env_config = {
                'tf': float(self.env.tf) if hasattr(self.env, 'tf') else 15.0,
                'tstep': float(self.env.tstep) if hasattr(self.env, 'tstep') else 1 / 30,
                'max_steps': int(self.env.max_steps) if hasattr(self.env, 'max_steps') else 28,
                'include_prony': bool(self.env.include_prony) if hasattr(self.env, 'include_prony') else True,
                'shared_observations': bool(self.env.shared_observations) if hasattr(self.env,
                                                                                     'shared_observations') else True,
                'prony_coordination': bool(self.env.prony_coordination) if hasattr(self.env,
                                                                                   'prony_coordination') else True,
                'algorithm_type': str(self.env.algorithm_type) if hasattr(self.env, 'algorithm_type') else 'MAPPO'
            }

            # 简化算法配置
            algo_config = {
                'actor_lr': float(self.config.get('actor_lr', 3e-4)),
                'critic_lr': float(self.config.get('critic_lr', 3e-4)),
                'gamma': float(self.config.get('gamma', 0.99)),
                'gae_lambda': float(self.config.get('gae_lambda', 0.95)),
                'clip_epsilon': float(self.config.get('clip_epsilon', 0.2)),
                'ppo_epochs': int(self.config.get('ppo_epochs', 10)),
                'batch_size': int(self.config.get('batch_size', 64)),
                'entropy_coef': float(self.config.get('entropy_coef', 0.01)),
                'value_coef': float(self.config.get('value_coef', 0.5)),
                'device': str(self.config.get('device', 'cpu'))
            }

            save_config_to_json(path, env_config, algo_config)

        except Exception as e:
            print(f"保存配置失败: {e}")

# 4. MASAC 完整实现 - 修复版本
class MASAC:
    """多智能体软演员-评论家算法完整实现 - 修复版本"""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.n_agents = len(env.possible_agents)
        self.agent_names = env.possible_agents

        # 获取每个智能体的观测和动作维度
        self.obs_dims = [env.observation_spaces[agent].shape[0] for agent in self.agent_names]
        self.act_dims = [env.action_spaces[agent].shape[0] for agent in self.agent_names]

        print(f"MASAC初始化: {self.n_agents}个智能体")
        print(f"观测维度: {self.obs_dims}")
        print(f"动作维度: {self.act_dims}")

        # Actor网络（输出均值和标准差）
        self.actors = [self._create_sac_actor(self.obs_dims[i], self.act_dims[i]).to(config['device'])
                       for i in range(self.n_agents)]

        # 计算Critic的总输入维度
        self.total_obs_dim = sum(self.obs_dims)
        self.total_act_dim = sum(self.act_dims)

        # Critic网络（双Q网络）
        self.critics1 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                         for _ in range(self.n_agents)]
        self.critics2 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                         for _ in range(self.n_agents)]

        # 目标Critic网络
        self.critic_targets1 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                                for _ in range(self.n_agents)]
        self.critic_targets2 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                                for _ in range(self.n_agents)]

        # 复制参数
        for i in range(self.n_agents):
            self.critic_targets1[i].load_state_dict(self.critics1[i].state_dict())
            self.critic_targets2[i].load_state_dict(self.critics2[i].state_dict())

        # 优化器
        self.actor_optimizers = [torch.optim.Adam(self.actors[i].parameters(),
                                                  lr=config['actor_lr'])
                                 for i in range(self.n_agents)]
        self.critic_optimizers1 = [torch.optim.Adam(self.critics1[i].parameters(),
                                                    lr=config['critic_lr'])
                                   for i in range(self.n_agents)]
        self.critic_optimizers2 = [torch.optim.Adam(self.critics2[i].parameters(),
                                                    lr=config['critic_lr'])
                                   for i in range(self.n_agents)]

        # 温度参数
        self.log_alpha = torch.zeros(1, requires_grad=True, device=config['device'])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config['alpha_lr'])
        self.target_entropy = -np.mean(self.act_dims)  # 平均动作维度作为目标熵

        # 经验回放
        self.buffer = ReplayBuffer(config['buffer_size'], self.obs_dims, self.act_dims, self.n_agents)

        # 训练参数
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.batch_size = config['batch_size']
        self.device = config['device']

        # 训练记录
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_values = []

    def _create_sac_actor(self, obs_dim, act_dim):
        """创建SAC的Actor网络 - 修复形状问题"""

        class SquashedGaussianPolicy(nn.Module):
            def __init__(self, obs_dim, act_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                )
                self.mean_layer = nn.Linear(256, act_dim)
                self.log_std_layer = nn.Linear(256, act_dim)

                # 初始化
                for layer in self.net:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.01)
                        nn.init.zeros_(layer.bias)
                nn.init.xavier_uniform_(self.mean_layer.weight, gain=0.01)
                nn.init.zeros_(self.mean_layer.bias)
                nn.init.xavier_uniform_(self.log_std_layer.weight, gain=0.01)
                nn.init.zeros_(self.log_std_layer.bias)

            def forward(self, obs):
                x = self.net(obs)
                mean = self.mean_layer(x)
                log_std = self.log_std_layer(x)
                log_std = torch.clamp(log_std, -20, 2)
                return mean, log_std

            def sample(self, obs):
                mean, log_std = self.forward(obs)
                std = log_std.exp()
                normal = Normal(mean, std)
                x_t = normal.rsample()  # 重参数化
                action = torch.tanh(x_t) * 0.5  # 缩放到[-0.5, 0.5]

                # 计算log概率 - 修复形状问题
                log_prob = normal.log_prob(x_t)
                # 修正log概率（由于tanh变换）
                log_prob -= torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(-1, keepdim=True)

                return action, log_prob

        return SquashedGaussianPolicy(obs_dim, act_dim)

    def select_action(self, obs, explore=True):
        """选择动作"""
        actions = {}
        for i, agent in enumerate(self.agent_names):
            obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0).to(self.device)

            if explore:
                action, _ = self.actors[i].sample(obs_tensor)
                action = action.detach().cpu().numpy()[0]
            else:
                with torch.no_grad():
                    mean, log_std = self.actors[i](obs_tensor)
                    std = log_std.exp()
                    normal = Normal(mean, std)
                    x_t = normal.rsample()
                    action = torch.tanh(x_t) * 0.5
                    action = action.cpu().numpy()[0]

            actions[agent] = action

        return actions

    def update(self):
        """更新网络参数 - 修复梯度问题"""
        if self.buffer.size < self.batch_size:
            return 0.0, 0.0, 0.0

        batch = self.buffer.sample(self.batch_size)

        obs = [torch.FloatTensor(batch['obs'][i]).to(self.device) for i in range(self.n_agents)]
        act = [torch.FloatTensor(batch['act'][i]).to(self.device) for i in range(self.n_agents)]
        rew = [torch.FloatTensor(batch['rew'][i]).to(self.device) for i in range(self.n_agents)]
        next_obs = [torch.FloatTensor(batch['next_obs'][i]).to(self.device) for i in range(self.n_agents)]
        done = [torch.FloatTensor(batch['done'][i]).to(self.device) for i in range(self.n_agents)]

        # 拼接所有智能体的观测和动作
        obs_cat = torch.cat(obs, dim=1)
        act_cat = torch.cat(act, dim=1)
        next_obs_cat = torch.cat(next_obs, dim=1)

        alpha = torch.exp(self.log_alpha)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_alpha_loss = 0.0

        for i in range(self.n_agents):
            # 更新Critic
            with torch.no_grad():
                # 下一个状态的动作和log概率
                next_actions = []
                next_log_probs = []
                for j in range(self.n_agents):
                    next_action, next_log_prob = self.actors[j].sample(next_obs[j])
                    next_actions.append(next_action)
                    next_log_probs.append(next_log_prob)

                next_act_cat = torch.cat(next_actions, dim=1)
                # 目标Q值
                target_q1 = self.critic_targets1[i](next_obs_cat, next_act_cat)
                target_q2 = self.critic_targets2[i](next_obs_cat, next_act_cat)
                target_q = torch.min(target_q1, target_q2)
                target_q = target_q - alpha * next_log_probs[i]  # 减去熵项
                target_q = rew[i] + (1 - done[i]) * self.gamma * target_q

            # 当前Q值
            current_q1 = self.critics1[i](obs_cat, act_cat)
            current_q2 = self.critics2[i](obs_cat, act_cat)

            # Critic损失
            critic_loss1 = F.mse_loss(current_q1, target_q)
            critic_loss2 = F.mse_loss(current_q2, target_q)

            # 更新Critic - 分别更新
            self.critic_optimizers1[i].zero_grad()
            critic_loss1.backward()
            torch.nn.utils.clip_grad_norm_(self.critics1[i].parameters(), 0.5)
            self.critic_optimizers1[i].step()

            self.critic_optimizers2[i].zero_grad()
            critic_loss2.backward()
            torch.nn.utils.clip_grad_norm_(self.critics2[i].parameters(), 0.5)
            self.critic_optimizers2[i].step()

            # 重新采样当前动作和log概率用于Actor更新
            current_actions = []
            current_log_probs = []
            for j in range(self.n_agents):
                if j == i:
                    action, log_prob = self.actors[j].sample(obs[j])
                    current_actions.append(action)
                    current_log_probs.append(log_prob)
                else:
                    action, log_prob = self.actors[j].sample(obs[j])
                    current_actions.append(action.detach())
                    current_log_probs.append(log_prob.detach())

            current_act_cat = torch.cat(current_actions, dim=1)

            # 计算Q值
            q1 = self.critics1[i](obs_cat, current_act_cat)
            q2 = self.critics2[i](obs_cat, current_act_cat)
            q = torch.min(q1, q2)

            actor_loss = (alpha * current_log_probs[i] - q).mean()

            # 更新Actor
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward(retain_graph=(i < self.n_agents - 1))
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()

            # 更新温度参数
            alpha_loss = -(self.log_alpha * (current_log_probs[i].detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.critics1[i].parameters(),
                                           self.critic_targets1[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critics2[i].parameters(),
                                           self.critic_targets2[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            total_actor_loss += actor_loss.item()
            total_critic_loss += (critic_loss1.item() + critic_loss2.item()) / 2
            total_alpha_loss += alpha_loss.item()

        return (total_actor_loss / self.n_agents,
                total_critic_loss / self.n_agents,
                total_alpha_loss / self.n_agents)

    def train(self, n_episodes, save_path):
        """训练MASAC算法"""

        os.makedirs(save_path, exist_ok=True)

        # 训练数据记录
        data = {
            'episode': [],
            'reward': [],
            'success': [],
            'avg_damping_ratio': [],
            'avg_oscillation_freq': [],
            'freq_deviation': [],
            'voltage_deviation': [],
            'actor_loss': [],
            'critic_loss': [],
            'alpha_loss': [],
            'alpha_value': [],
            'action_magnitude': []
        }

        for episode in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = {agent: 0.0 for agent in self.agent_names}
            episode_success = False

            # 收集数据
            freq_deviations = []
            voltage_deviations = []
            damping_ratios = []
            osc_freqs = []
            action_magnitudes = []
            episode_actor_loss = 0.0
            episode_critic_loss = 0.0
            episode_alpha_loss = 0.0
            update_count = 0

            step = 0
            while step < self.env.max_steps:
                actions = self.select_action(obs, explore=True)
                next_obs, rewards, dones, truncs, infos = self.env.step(actions)

                # 存储经验
                obs_list = [obs[agent] for agent in self.agent_names]
                next_obs_list = [next_obs[agent] for agent in self.agent_names]
                act_list = [actions[agent] for agent in self.agent_names]
                rew_list = [rewards[agent] for agent in self.agent_names]
                done_list = [dones[agent] for agent in self.agent_names]

                self.buffer.store(obs_list, act_list, rew_list, next_obs_list, done_list)

                for agent in self.agent_names:
                    episode_reward[agent] += rewards[agent]

                # 收集系统状态数据
                if hasattr(self.env, 'wcoi_idx'):
                    try:
                        wcoi = self.env.sim_case.dae.y[self.env.wcoi_idx].astype(np.float32)
                        if len(wcoi) > 0:
                            freq_dev = np.mean(np.abs(wcoi - 1.0))
                            freq_deviations.append(freq_dev)
                    except:
                        pass

                if hasattr(self.env, 'voltage_idx'):
                    try:
                        voltage = self.env.sim_case.dae.y[self.env.voltage_idx].astype(np.float32)
                        if len(voltage) > 0:
                            volt_dev = np.mean(np.abs(voltage - 1.0))
                            voltage_deviations.append(volt_dev)
                    except:
                        pass

                # 收集相域信息
                if hasattr(self.env, 'prony_analyzers'):
                    for agent in self.agent_names:
                        analyzer = self.env.prony_analyzers[agent]
                        if analyzer.last_analysis and analyzer.last_analysis['valid']:
                            damping_ratios.append(analyzer.last_analysis['damping_ratio'])
                            osc_freqs.append(analyzer.last_analysis['oscillation_freq'])

                # 收集动作幅度
                for agent in self.agent_names:
                    if agent in actions:
                        action_magnitudes.append(np.mean(np.abs(actions[agent])))

                # 更新网络
                if self.buffer.size >= self.batch_size:
                    actor_loss, critic_loss, alpha_loss = self.update()
                    episode_actor_loss += actor_loss
                    episode_critic_loss += critic_loss
                    episode_alpha_loss += alpha_loss
                    update_count += 1

                obs = next_obs
                step += 1

                if all(dones.values()) or all(truncs.values()):
                    episode_success = not any(infos[agent].get('sim_crashed', False) for agent in self.agent_names)
                    break

            # 计算统计数据
            total_reward = sum(episode_reward.values())
            self.episode_rewards.append(total_reward)

            avg_freq_dev = np.mean(freq_deviations) if freq_deviations else 0.0
            avg_volt_dev = np.mean(voltage_deviations) if voltage_deviations else 0.0
            avg_damping = np.mean(damping_ratios) if damping_ratios else 0.0
            avg_osc_freq = np.mean(osc_freqs) if osc_freqs else 0.0
            avg_action_mag = np.mean(action_magnitudes) if action_magnitudes else 0.0
            avg_actor_loss = episode_actor_loss / max(1, update_count)
            avg_critic_loss = episode_critic_loss / max(1, update_count)
            avg_alpha_loss = episode_alpha_loss / max(1, update_count)
            alpha_value = torch.exp(self.log_alpha).item()

            self.actor_losses.append(avg_actor_loss)
            self.critic_losses.append(avg_critic_loss)
            self.alpha_values.append(alpha_value)

            # 记录数据
            data['episode'].append(episode)
            data['reward'].append(total_reward)
            data['success'].append(1 if episode_success else 0)
            data['avg_damping_ratio'].append(avg_damping)
            data['avg_oscillation_freq'].append(avg_osc_freq)
            data['freq_deviation'].append(avg_freq_dev)
            data['voltage_deviation'].append(avg_volt_dev)
            data['actor_loss'].append(avg_actor_loss)
            data['critic_loss'].append(avg_critic_loss)
            data['alpha_loss'].append(avg_alpha_loss)
            data['alpha_value'].append(alpha_value)
            data['action_magnitude'].append(avg_action_mag)

            # 定期保存
            if episode % 10 == 0 or episode == n_episodes - 1:
                # 保存训练数据
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(save_path, 'training_data.csv'), index=False)

                # 保存模型
                self._save_model(save_path, episode)

                print(f"MASAC - Episode {episode}, "
                      f"Reward: {total_reward:.2f}, "
                      f"Success: {episode_success}, "
                      f"Damping: {avg_damping:.4f}, "
                      f"Actor Loss: {avg_actor_loss:.4f}, "
                      f"Critic Loss: {avg_critic_loss:.4f}, "
                      f"Alpha: {alpha_value:.4f}")

        # 最终保存
        self._save_model(save_path, 'final')
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_path, 'training_data.csv'), index=False)

        # 保存配置
        self._save_config(save_path)

        print(f"MASAC训练完成！数据保存到: {save_path}")
        return data

    def _save_model(self, path, episode):
        """保存模型"""
        model_dict = {
            'actors': [self.actors[i].state_dict() for i in range(self.n_agents)],
            'critics1': [self.critics1[i].state_dict() for i in range(self.n_agents)],
            'critics2': [self.critics2[i].state_dict() for i in range(self.n_agents)],
            'critic_targets1': [self.critic_targets1[i].state_dict() for i in range(self.n_agents)],
            'critic_targets2': [self.critic_targets2[i].state_dict() for i in range(self.n_agents)],
            'actor_optimizers': [self.actor_optimizers[i].state_dict() for i in range(self.n_agents)],
            'critic_optimizers1': [self.critic_optimizers1[i].state_dict() for i in range(self.n_agents)],
            'critic_optimizers2': [self.critic_optimizers2[i].state_dict() for i in range(self.n_agents)],
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'obs_dims': self.obs_dims,
            'act_dims': self.act_dims,
            'total_obs_dim': self.total_obs_dim,
            'total_act_dim': self.total_act_dim,
            'agent_names': self.agent_names,
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'alpha_values': self.alpha_values,
            'config': self.config
        }
        torch.save(model_dict, os.path.join(path, f'model_{episode}.pth'))

    def _save_config(self, path):
        """保存配置"""
        env_config = {
            'tf': self.env.tf,
            'tstep': self.env.tstep,
            'max_steps': self.env.max_steps,
            'include_prony': self.env.include_prony,
            'shared_observations': self.env.shared_observations,
            'prony_coordination': self.env.prony_coordination,
            'algorithm_type': self.env.algorithm_type
        }
        save_config_to_json(path, env_config, self.config)

class WSE_MATD3:
    """NSGA2与MATD3真正混合的多目标算法 - 完整修复版本"""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.n_agents = len(env.possible_agents)
        self.agent_names = env.possible_agents

        # 获取每个智能体的观测和动作维度
        self.obs_dims = [env.observation_spaces[agent].shape[0] for agent in self.agent_names]
        self.act_dims = [env.action_spaces[agent].shape[0] for agent in self.agent_names]

        print(f"WSE_MATD3初始化: {self.n_agents}个智能体")
        print(f"观测维度: {self.obs_dims}")
        print(f"动作维度: {self.act_dims}")

        # 计算Critic的总输入维度
        self.total_obs_dim = sum(self.obs_dims)
        self.total_act_dim = sum(self.act_dims)

        # 创建Actor网络
        self.actors = [self._create_actor(self.obs_dims[i], self.act_dims[i]).to(config['device'])
                       for i in range(self.n_agents)]
        self.actor_targets = [self._create_actor(self.obs_dims[i], self.act_dims[i]).to(config['device'])
                              for i in range(self.n_agents)]

        # 使用 MultiAgentCriticNetwork 创建双Critic网络
        self.critics1 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                         for _ in range(self.n_agents)]
        self.critics2 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                         for _ in range(self.n_agents)]
        self.critic_targets1 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                                for _ in range(self.n_agents)]
        self.critic_targets2 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                                for _ in range(self.n_agents)]

        # 复制参数
        for i in range(self.n_agents):
            self.actor_targets[i].load_state_dict(self.actors[i].state_dict())
            self.critic_targets1[i].load_state_dict(self.critics1[i].state_dict())
            self.critic_targets2[i].load_state_dict(self.critics2[i].state_dict())

        # 优化器
        self.actor_optimizers = [torch.optim.Adam(self.actors[i].parameters(),
                                                  lr=config.get('actor_lr', 1e-4))
                                 for i in range(self.n_agents)]
        self.critic_optimizers1 = [torch.optim.Adam(self.critics1[i].parameters(),
                                                    lr=config.get('critic_lr', 1e-3))
                                   for i in range(self.n_agents)]
        self.critic_optimizers2 = [torch.optim.Adam(self.critics2[i].parameters(),
                                                    lr=config.get('critic_lr', 1e-3))
                                   for i in range(self.n_agents)]

        # 经验回放
        self.buffer_size = config.get('buffer_size', 50000)
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_dims, self.act_dims, self.n_agents)

        # MATD3训练参数
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.01)
        self.noise_std = config.get('noise_std', 0.05)
        self.noise_std_initial = self.noise_std  # 新增：保存初始噪声标准差，用于退火
        self.noise_clip = config.get('noise_clip', 0.5)
        self.policy_noise = config.get('policy_noise', 0.2)
        self.policy_freq = config.get('policy_freq', 2)
        self.batch_size = config.get('batch_size', 256)
        self.device = config['device']

        # 崩溃惩罚（用于奖励计算）
        self.crash_penalty = config.get('crash_penalty', -10.0)  # 新增

        # NSGA2多目标参数
        self.nsga2_config = {
            'population_size': config.get('nsga2_population_size', 20),
            'crossover_rate': config.get('nsga2_crossover_rate', 0.8),
            'mutation_rate': config.get('nsga2_mutation_rate', 0.05),
            'mutation_std': config.get('nsga2_mutation_std', 0.05),
            'n_objectives': 4  # 频率偏差、电压偏差、阻尼比、控制代价
        }

        # 多目标权重向量（Pareto前沿）
        self.weight_vectors = self._initialize_weight_vectors()
        self.current_weight_idx = 0
        self.weight_update_freq = config.get('weight_update_freq', 20)

        # 训练记录
        self.episode_rewards = []
        self.pareto_front_history = []
        self.multi_objective_values = []  # 存储多目标值，供NSGA2更新使用
        self.training_data = []  # 用于记录训练数据

        # 目标函数归一化参数
        # 目标函数归一化参数（增加count字段）
        self.objective_stats = {
            'freq_dev': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'volt_dev': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'damping': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'control_cost': {'mean': 0.0, 'std': 1.0, 'count': 0}
        }
        self.warmup_steps = config.get('warmup_steps', 200)  # 预热步数

        self.total_it = 0

        print(f"算法配置: gamma={self.gamma}, tau={self.tau}, batch_size={self.batch_size}")
        print(
            f"多目标配置: 种群大小={self.nsga2_config['population_size']}, 目标数={self.nsga2_config['n_objectives']}")

    def _create_actor(self, obs_dim, act_dim):
        return nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )

    def _create_critic(self, obs_dim, act_dim):
        """创建Critic网络"""
        return nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _initialize_weight_vectors(self):
        """初始化多目标权重向量"""
        population_size = self.nsga2_config['population_size']
        n_objectives = self.nsga2_config['n_objectives']
        weight_vectors = []
        for _ in range(population_size):
            weights = np.random.dirichlet([1.0] * n_objectives)
            weight_vectors.append(weights)
        return weight_vectors

    def select_action(self, obs, explore=True):
        """选择动作"""
        actions = {}
        for i, agent in enumerate(self.agent_names):
            obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0).to(self.device)
            if torch.isnan(obs_tensor).any():
                raise ValueError(f"观测包含NaN: {obs[agent]}")

            with torch.no_grad():
                action = self.actors[i](obs_tensor).cpu().numpy()[0]

            if explore:
                noise = np.random.normal(0, self.noise_std, size=action.shape)
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
                action = action + noise

            action = np.clip(action, -1.0, 1.0)
            actions[agent] = action

        return actions

    def _compute_multi_objective_reward(self, rewards_dict, obs_dict, actions_dict):
        """计算多目标奖励，添加严格的数据有效性检查"""
        if not self.weight_vectors:
            return 0.0, {}

        current_weights = self.weight_vectors[self.current_weight_idx]

        # 提取多目标值
        freq_dev = 0.0
        volt_dev = 0.0
        damping = 0.0
        control_cost = 0.0

        try:
            # === 频率偏差 ===
            if hasattr(self.env, 'wcoi_idx'):
                wcoi = self.env.sim_case.dae.y[self.env.wcoi_idx].astype(np.float32)
                if len(wcoi) == 0:
                    raise ValueError("wcoi array is empty")
                freq_dev = np.mean(np.abs(wcoi - 1.0))
                if np.isnan(freq_dev) or np.isinf(freq_dev):
                    raise ValueError(f"freq_dev is NaN/inf: {freq_dev}")
            else:
                raise ValueError("env has no wcoi_idx")

            # === 电压偏差 ===
            if hasattr(self.env, 'voltage_idx'):
                voltage = self.env.sim_case.dae.y[self.env.voltage_idx].astype(np.float32)
                if len(voltage) == 0:
                    raise ValueError("voltage array is empty")
                volt_dev = np.mean(np.abs(voltage - 1.0))
                if np.isnan(volt_dev) or np.isinf(volt_dev):
                    raise ValueError(f"volt_dev is NaN/inf: {volt_dev}")
            else:
                raise ValueError("env has no voltage_idx")

            # === 阻尼比（Prony分析结果） ===
            if hasattr(self.env, 'prony_analyzers'):
                damping_vals = []
                for agent in self.agent_names:
                    analyzer = self.env.prony_analyzers[agent]
                    if analyzer and analyzer.last_analysis:
                        res = analyzer.last_analysis
                        if not res.get('valid', False):
                            raise ValueError(f"Prony analysis invalid for {agent}")
                        damping_vals.append(res['damping_ratio'])
                    else:
                        raise ValueError(f"No Prony analysis result for {agent}")
                if len(damping_vals) == 0:
                    raise ValueError("No damping ratios collected")
                damping = np.mean(damping_vals)
                if np.isnan(damping) or np.isinf(damping):
                    raise ValueError(f"damping is NaN/inf: {damping}")
            else:
                raise ValueError("env has no prony_analyzers")

            # === 控制代价（动作幅值） ===
            control_costs = []
            for agent in self.agent_names:
                if agent in actions_dict:
                    cost = np.mean(np.abs(actions_dict[agent]))
                    if np.isnan(cost) or np.isinf(cost):
                        raise ValueError(f"control cost for {agent} is NaN/inf: {cost}")
                    control_costs.append(cost)
                else:
                    raise ValueError(f"Action missing for {agent}")
            if len(control_costs) == 0:
                raise ValueError("No control costs collected")
            control_cost = np.mean(control_costs)

        except Exception as e:
            # 直接抛出原始异常，不允许静默失败
            raise RuntimeError(f"Error computing multi-objective reward: {e}") from e

        # 更新统计和归一化（保持不变）
        self._update_objective_stats(freq_dev, volt_dev, damping, control_cost)

        if self.objective_stats['freq_dev']['count'] > self.warmup_steps:
            freq_norm = (freq_dev - self.objective_stats['freq_dev']['mean']) / (
                        self.objective_stats['freq_dev']['std'] + 1e-6)
            volt_norm = (volt_dev - self.objective_stats['volt_dev']['mean']) / (
                        self.objective_stats['volt_dev']['std'] + 1e-6)
            damp_norm = (damping - self.objective_stats['damping']['mean']) / (
                        self.objective_stats['damping']['std'] + 1e-6)
            cost_norm = (control_cost - self.objective_stats['control_cost']['mean']) / (
                        self.objective_stats['control_cost']['std'] + 1e-6)
        else:
            freq_norm = freq_dev
            volt_norm = volt_dev
            damp_norm = damping
            cost_norm = control_cost

        # 加权奖励（负号表示最小化，阻尼比取正）
        weighted = (-current_weights[0] * freq_norm * 2.0 +
                    -current_weights[1] * volt_norm * 1.0 +
                    current_weights[2] * damp_norm * 10.0 +
                    -current_weights[3] * cost_norm * 0.5)

        scaled_reward = np.tanh(weighted) * 5.0

        return scaled_reward, {
            'freq_dev': freq_dev,
            'volt_dev': volt_dev,
            'damping': damping,
            'control_cost': control_cost,
            'weights': current_weights
        }

    def _update_objective_stats(self, freq_dev, volt_dev, damping, control_cost):
        """更新目标函数统计信息（带预热机制）"""
        alpha = 0.1  # 移动平均系数
        for key, val in zip(['freq_dev', 'volt_dev', 'damping', 'control_cost'],
                            [freq_dev, volt_dev, damping, control_cost]):
            stat = self.objective_stats[key]
            stat['count'] += 1
            if stat['count'] <= self.warmup_steps:
                # 预热阶段：直接累积平均
                if stat['count'] == 1:
                    stat['mean'] = val
                else:
                    stat['mean'] = (stat['mean'] * (stat['count'] - 1) + val) / stat['count']
                stat['std'] = 1.0  # 初始标准差设为1，避免除零
            else:
                # 正常运行阶段：使用指数移动平均
                old_mean = stat['mean']
                stat['mean'] = (1 - alpha) * old_mean + alpha * val
                stat['std'] = (1 - alpha) * stat['std'] + alpha * np.abs(val - old_mean)

    def update(self):
        """更新网络参数"""
        self.total_it += 1

        if self.buffer.size < self.batch_size:
            return None, None

        batch = self.buffer.sample(self.batch_size)

        # 转换为tensor
        obs = [torch.FloatTensor(batch['obs'][i]).to(self.device) for i in range(self.n_agents)]
        act = [torch.FloatTensor(batch['act'][i]).to(self.device) for i in range(self.n_agents)]
        rew = [torch.FloatTensor(batch['rew'][i]).to(self.device) for i in range(self.n_agents)]
        next_obs = [torch.FloatTensor(batch['next_obs'][i]).to(self.device) for i in range(self.n_agents)]
        done = [torch.FloatTensor(batch['done'][i]).to(self.device) for i in range(self.n_agents)]

        # 检查NaN
        for i in range(self.n_agents):
            if torch.isnan(obs[i]).any():
                raise ValueError(f"观测包含NaN，智能体索引 {i}, 观测: {obs[i]}")
            if torch.isnan(act[i]).any():
                raise ValueError(f"动作包含NaN，智能体索引 {i}, 动作: {act[i]}")
            if torch.isnan(rew[i]).any():
                raise ValueError(f"奖励包含NaN，智能体索引 {i}, 奖励: {rew[i]}")
            if torch.isnan(next_obs[i]).any():
                raise ValueError(f"下一观测包含NaN，智能体索引 {i}, 下一观测: {next_obs[i]}")

        obs_cat = torch.cat(obs, dim=1)
        act_cat = torch.cat(act, dim=1)
        next_obs_cat = torch.cat(next_obs, dim=1)

        critic_losses = []
        actor_losses = []

        for i in range(self.n_agents):
            # 目标Q值
            with torch.no_grad():
                next_actions = []
                for j in range(self.n_agents):
                    next_a = self.actor_targets[j](next_obs[j])
                    noise = torch.randn_like(next_a) * self.policy_noise
                    noise = noise.clamp(-self.noise_clip, self.noise_clip)
                    next_a = (next_a + noise).clamp(-1.0, 1.0)
                    next_actions.append(next_a)
                next_act_cat = torch.cat(next_actions, dim=1)

                target_q1 = self.critic_targets1[i](next_obs_cat, next_act_cat)
                target_q2 = self.critic_targets2[i](next_obs_cat, next_act_cat)
                target_q = torch.min(target_q1, target_q2)
                target_q = rew[i] + (1 - done[i]) * self.gamma * target_q

            # 当前Q值
            current_q1 = self.critics1[i](obs_cat, act_cat)
            current_q2 = self.critics2[i](obs_cat, act_cat)

            critic_loss1 = F.mse_loss(current_q1, target_q)
            critic_loss2 = F.mse_loss(current_q2, target_q)

            # 更新Critic1
            self.critic_optimizers1[i].zero_grad()
            critic_loss1.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critics1[i].parameters(), 0.5)#保持梯度裁剪阈值为 0.5 不变
            self.critic_optimizers1[i].step()

            # 更新Critic2
            self.critic_optimizers2[i].zero_grad()
            critic_loss2.backward()
            torch.nn.utils.clip_grad_norm_(self.critics2[i].parameters(), 0.5)
            self.critic_optimizers2[i].step()

            critic_losses.append((critic_loss1.item() + critic_loss2.item()) / 2)

            # 延迟策略更新
            if self.total_it % self.policy_freq == 0:
                # 当前动作
                current_actions = [self.actors[j](obs[j]) for j in range(self.n_agents)]
                current_act_cat = torch.cat(current_actions, dim=1)

                # Actor损失
                actor_loss = -self.critics1[i](obs_cat, current_act_cat).mean()

                # 多目标多样性正则（可选）
                if hasattr(self, 'weight_vectors') and len(self.weight_vectors) > 0:
                    action_std = torch.std(current_act_cat, dim=1).mean()
                    actor_loss = actor_loss - 0.01 * action_std  # 鼓励多样性

                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                self.actor_optimizers[i].step()

                actor_losses.append(actor_loss.item())

        # 软更新目标网络
        if self.total_it % self.policy_freq == 0:
            for i in range(self.n_agents):
                for param, target in zip(self.critics1[i].parameters(), self.critic_targets1[i].parameters()):
                    target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
                for param, target in zip(self.critics2[i].parameters(), self.critic_targets2[i].parameters()):
                    target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
                for param, target in zip(self.actors[i].parameters(), self.actor_targets[i].parameters()):
                    target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)

        avg_critic = np.mean(critic_losses) if critic_losses else 0.0
        avg_actor = np.mean(actor_losses) if actor_losses else 0.0
        return avg_critic, avg_actor

    def _WSE_update(self):
        if len(self.multi_objective_values) < 10:
            return

        recent = np.array(self.multi_objective_values[-min(50, len(self.multi_objective_values)):])
        if recent.ndim == 1:
            recent = recent.reshape(1, -1)
        if recent.shape[1] != self.nsga2_config['n_objectives']:
            return

        # 检查是否包含非有限值
        if not np.all(np.isfinite(recent)):
            print("Warning: multi_objective_values contains NaN/Inf, skipping NSGA2 update")
            return

        # 如果数据点太少，直接返回
        if recent.shape[0] < 2:
            return

        # 计算每个权重向量的平均加权得分
        scores = []
        for w in self.weight_vectors:
            weighted = recent @ w  # (n,4) @ (4,) -> (n,)
            score = weighted.mean()
            scores.append(score)

        best_idx = np.argmax(scores) if scores else 0
        # 交叉变异生成新权重种群（原有代码）

        # 生成新种群
        new_weights = []
        pop_size = len(self.weight_vectors)
        for i in range(pop_size):
            if i == 0:
                new_weights.append(self.weight_vectors[best_idx].copy())
            else:
                p1 = self.weight_vectors[np.random.randint(pop_size)]
                p2 = self.weight_vectors[np.random.randint(pop_size)]
                child = self._crossover_weights(p1, p2)
                child = self._mutate_weights(child)
                new_weights.append(child)

        self.weight_vectors = new_weights

        # 提取Pareto前沿并记录
        try:
            front = self._extract_pareto_front(recent)
            if len(front) > 0:
                self.pareto_front_history.append(front)
        except:
            pass

    def _crossover_weights(self, w1, w2):
        alpha = np.random.random()
        child = alpha * w1 + (1 - alpha) * w2
        child = child / child.sum()
        return child

    def _mutate_weights(self, w):
        rate = self.nsga2_config['mutation_rate']
        std = self.nsga2_config['mutation_std']
        mutated = w.copy()
        for i in range(len(w)):
            if np.random.rand() < rate:
                mutated[i] += np.random.normal(0, std)
        mutated = np.clip(mutated, 0.01, 1.0)
        mutated = mutated / mutated.sum()
        return mutated

    def _extract_pareto_front(self, objectives):
        n = len(objectives)
        pareto = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                # 如果 j 的所有目标值都小于 i，则 j 支配 i（最小化问题）
                if all(objectives[j][k] < objectives[i][k] for k in range(objectives.shape[1])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(i)
        return objectives[pareto]

    def train(self, n_episodes, save_path):
        """训练算法"""
        os.makedirs(save_path, exist_ok=True)

        data = {
            'episode': [], 'reward': [], 'success': [],
            'avg_damping_ratio': [], 'avg_oscillation_freq': [],
            'freq_deviation': [], 'voltage_deviation': [],
            'control_cost': [], 'weight_idx': [],
            'critic_loss': [], 'actor_loss': [], 'buffer_size': []
        }

        print(f"开始训练WSE_MATD3算法，共{n_episodes}个episodes...")

        for episode in range(n_episodes):
            # 修改点4：噪声退火
            progress = episode / max(1, n_episodes - 1)
            self.noise_std = max(0.01, self.noise_std_initial * (1 - progress))

            obs, info = self.env.reset()
            episode_reward = {agent: 0.0 for agent in self.agent_names}
            episode_success = False

            freq_deviations = []
            voltage_deviations = []
            damping_ratios = []
            osc_freqs = []
            control_costs = []
            critic_losses = []
            actor_losses = []

            step = 0
            crashed = False

            while step < self.env.max_steps:
                try:
                    actions = self.select_action(obs, explore=True)
                    next_obs, rewards, dones, truncs, infos = self.env.step(actions)

                    # 检查崩溃
                    if infos and any(info.get('sim_crashed', False) for info in infos.values()):
                        crashed = True
                        print(f"Episode {episode}: 仿真在步数{step}崩溃")
                        break

                    # 计算多目标奖励
                    mo_reward, mo_metrics = self._compute_multi_objective_reward(rewards, obs, actions)
                    # 为每个智能体分配相同奖励
                    for agent in self.agent_names:
                        rewards[agent] = mo_reward / self.n_agents

                    # 存储经验
                    obs_list = [obs[agent] for agent in self.agent_names]
                    act_list = [actions[agent] for agent in self.agent_names]
                    rew_list = [rewards[agent] for agent in self.agent_names]
                    next_obs_list = [next_obs[agent] for agent in self.agent_names]
                    done_list = [dones[agent] for agent in self.agent_names]
                    self.buffer.store(obs_list, act_list, rew_list, next_obs_list, done_list)

                    # 累加奖励
                    for agent in self.agent_names:
                        episode_reward[agent] += rewards[agent]

                    # 收集系统数据
                    if hasattr(self.env, 'wcoi_idx'):
                        try:
                            wcoi = self.env.sim_case.dae.y[self.env.wcoi_idx].astype(np.float32)
                            freq_deviations.append(np.mean(np.abs(wcoi - 1.0)))
                        except:
                            pass
                    if hasattr(self.env, 'voltage_idx'):
                        try:
                            volt = self.env.sim_case.dae.y[self.env.voltage_idx].astype(np.float32)
                            voltage_deviations.append(np.mean(np.abs(volt - 1.0)))
                        except:
                            pass
                    if hasattr(self.env, 'prony_analyzers'):
                        for agent in self.agent_names:
                            ana = self.env.prony_analyzers[agent]
                            if ana and ana.last_analysis:
                                damping_ratios.append(ana.last_analysis['damping_ratio'])
                                osc_freqs.append(ana.last_analysis['oscillation_freq'])
                    control_costs.append(mo_metrics['control_cost'])

                    # 更新网络
                    if self.buffer.size >= self.batch_size:
                        cl, al = self.update()
                        if cl is not None:
                            critic_losses.append(cl)
                        if al is not None:
                            actor_losses.append(al)

                    obs = next_obs
                    step += 1

                    if all(dones.values()) or all(truncs.values()):
                        episode_success = not crashed
                        break

                except Exception as e:
                    print(f"Episode {episode}, Step {step} 出错: {e}")
                    crashed = True
                    break

            # 统计
            if crashed:
                total_reward = self.crash_penalty
                avg_freq = avg_volt = avg_damp = avg_osc = avg_cost = 0.0
                avg_cl = avg_al = 0.0
                success = 0
            else:
                total_reward = sum(episode_reward.values())
                avg_freq = np.nanmean(freq_deviations) if freq_deviations else 0.0
                avg_volt = np.nanmean(voltage_deviations) if voltage_deviations else 0.0
                avg_damp = np.nanmean(damping_ratios) if damping_ratios else 0.0
                avg_osc = np.nanmean(osc_freqs) if osc_freqs else 0.0
                avg_cost = np.nanmean(control_costs) if control_costs else 0.0
                avg_cl = np.nanmean(critic_losses) if critic_losses else 0.0
                avg_al = np.nanmean(actor_losses) if actor_losses else 0.0
                success = 1

            self.episode_rewards.append(total_reward)
            if not np.isnan(avg_freq) and not np.isnan(avg_volt) and not np.isnan(avg_damp) and not np.isnan(avg_cost):
                self.multi_objective_values.append([avg_freq, avg_volt, avg_damp, avg_cost])

            # 记录数据
            data['episode'].append(episode)
            data['reward'].append(total_reward)
            data['success'].append(success)
            data['avg_damping_ratio'].append(avg_damp)
            data['avg_oscillation_freq'].append(avg_osc)
            data['freq_deviation'].append(avg_freq)
            data['voltage_deviation'].append(avg_volt)
            data['control_cost'].append(avg_cost)
            data['weight_idx'].append(self.current_weight_idx)
            data['critic_loss'].append(avg_cl)
            data['actor_loss'].append(avg_al)
            data['buffer_size'].append(self.buffer.size)

            # NSGA2权重更新
            if (episode % self.weight_update_freq == 0 and episode > 0 and
                    len(self.multi_objective_values) > 10):
                self._WSE_update()
                self.current_weight_idx = (self.current_weight_idx + 1) % len(self.weight_vectors)

            # 定期保存
            if episode % 10 == 0 or episode == n_episodes - 1:
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(save_path, 'training_data.csv'), index=False)
                self.save_model(save_path, episode)

                w = self.weight_vectors[self.current_weight_idx] if self.weight_vectors else [0.25]*4
                print(f"WSE_MATD3 - Episode {episode}: "
                      f"奖励={total_reward:.2f}, 阻尼比={avg_damp:.4f}, 频率偏差={avg_freq:.4f}, "
                      f"权重={w}, 成功={success}, Critic损失={avg_cl:.4f}, Actor损失={avg_al:.4f}")

        # 最终保存
        self.save_model(save_path, 'final')
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_path, 'training_data.csv'), index=False)
        # 保存配置...
        env_config = {
            'tf': self.env.tf,
            'tstep': self.env.tstep,
            'max_steps': self.env.max_steps,
            'include_prony': self.env.include_prony,
            'shared_observations': self.env.shared_observations,
            'prony_coordination': self.env.prony_coordination,
            'algorithm_type': self.env.algorithm_type
        }
        algo_config = {
            'gamma': self.gamma,
            'tau': self.tau,
            'noise_std': self.noise_std_initial,
            'noise_clip': self.noise_clip,
            'policy_noise': self.policy_noise,
            'policy_freq': self.policy_freq,
            'batch_size': self.batch_size,
            'actor_lr': self.config.get('actor_lr', 1e-4),
            'critic_lr': self.config.get('critic_lr', 1e-3),
            'nsga2_population_size': self.nsga2_config['population_size'],
            'weight_update_freq': self.weight_update_freq
        }
        save_config_to_json(save_path, env_config, algo_config)

        # 保存Pareto历史
        if self.pareto_front_history:
            with open(os.path.join(save_path, 'pareto_front_history.pkl'), 'wb') as f:
                pickle.dump(self.pareto_front_history, f)

        print(f"WSE_MATD3_Mix训练完成！数据保存到: {save_path}")
        return data

    def save_model(self, path, episode):
        """保存模型"""
        model_dict = {
            'actors': [self.actors[i].state_dict() for i in range(self.n_agents)],
            'critics1': [self.critics1[i].state_dict() for i in range(self.n_agents)],
            'critics2': [self.critics2[i].state_dict() for i in range(self.n_agents)],
            'actor_targets': [self.actor_targets[i].state_dict() for i in range(self.n_agents)],
            'critic_targets1': [self.critic_targets1[i].state_dict() for i in range(self.n_agents)],
            'critic_targets2': [self.critic_targets2[i].state_dict() for i in range(self.n_agents)],
            'weight_vectors': self.weight_vectors,
            'current_weight_idx': self.current_weight_idx,
            'multi_objective_values': self.multi_objective_values,
            'pareto_front_history': self.pareto_front_history,
            'obs_dims': self.obs_dims,
            'act_dims': self.act_dims,
            'total_obs_dim': self.total_obs_dim,
            'total_act_dim': self.total_act_dim,
            'agent_names': self.agent_names,
            'config': self.config
        }
        torch.save(model_dict, os.path.join(path, f'model_{episode}.pth'))
        print(f"模型保存成功: {os.path.join(path, f'model_{episode}.pth')}")

class NSGA2_MATD3:

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.n_agents = len(env.possible_agents)
        self.agent_names = env.possible_agents

        # 获取每个智能体的观测和动作维度
        self.obs_dims = [env.observation_spaces[agent].shape[0] for agent in self.agent_names]
        self.act_dims = [env.action_spaces[agent].shape[0] for agent in self.agent_names]

        print(f"NSGA2_MATD3_Mix初始化: {self.n_agents}个智能体")
        print(f"观测维度: {self.obs_dims}")
        print(f"动作维度: {self.act_dims}")

        # 计算Critic的总输入维度
        self.total_obs_dim = sum(self.obs_dims)
        self.total_act_dim = sum(self.act_dims)

        # 创建Actor网络
        self.actors = [self._create_actor(self.obs_dims[i], self.act_dims[i]).to(config['device'])
                       for i in range(self.n_agents)]
        self.actor_targets = [self._create_actor(self.obs_dims[i], self.act_dims[i]).to(config['device'])
                              for i in range(self.n_agents)]

        # 使用 MultiAgentCriticNetwork 创建双Critic网络
        self.critics1 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                         for _ in range(self.n_agents)]
        self.critics2 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                         for _ in range(self.n_agents)]
        self.critic_targets1 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                                for _ in range(self.n_agents)]
        self.critic_targets2 = [MultiAgentCriticNetwork(self.total_obs_dim, self.total_act_dim).to(config['device'])
                                for _ in range(self.n_agents)]

        # 复制参数
        for i in range(self.n_agents):
            self.actor_targets[i].load_state_dict(self.actors[i].state_dict())
            self.critic_targets1[i].load_state_dict(self.critics1[i].state_dict())
            self.critic_targets2[i].load_state_dict(self.critics2[i].state_dict())

        # 优化器
        self.actor_optimizers = [torch.optim.Adam(self.actors[i].parameters(),
                                                  lr=config.get('actor_lr', 1e-4))
                                 for i in range(self.n_agents)]
        self.critic_optimizers1 = [torch.optim.Adam(self.critics1[i].parameters(),
                                                    lr=config.get('critic_lr', 1e-3))
                                   for i in range(self.n_agents)]
        self.critic_optimizers2 = [torch.optim.Adam(self.critics2[i].parameters(),
                                                    lr=config.get('critic_lr', 1e-3))
                                   for i in range(self.n_agents)]

        # 经验回放
        self.buffer_size = config.get('buffer_size', 50000)
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_dims, self.act_dims, self.n_agents)

        # MATD3训练参数
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.01)
        self.noise_std = config.get('noise_std', 0.05)
        self.noise_std_initial = self.noise_std
        self.noise_clip = config.get('noise_clip', 0.5)
        self.policy_noise = config.get('policy_noise', 0.2)
        self.policy_freq = config.get('policy_freq', 2)
        self.batch_size = config.get('batch_size', 256)
        self.device = config['device']

        # 崩溃惩罚
        self.crash_penalty = config.get('crash_penalty', -10.0)

        # NSGA2多目标参数
        self.nsga2_config = {
            'population_size': config.get('nsga2_population_size', 20),
            'crossover_rate': config.get('nsga2_crossover_rate', 0.8),
            'mutation_rate': config.get('nsga2_mutation_rate', 0.05),
            'mutation_std': config.get('nsga2_mutation_std', 0.05),
            'n_objectives': 4
        }

        # 多目标权重向量（Pareto前沿）—— 先初始化
        self.weight_vectors = self._initialize_weight_vectors()
        self.current_weight_idx = 0
        self.weight_update_freq = config.get('weight_update_freq', 20)

        # 记录每个权重向量的历史性能（初始化映射）
        self.weight_performance = {i: [] for i in range(len(self.weight_vectors))}

        # 训练记录
        self.episode_rewards = []
        self.pareto_front_history = []
        self.multi_objective_values = []  # 存储每个episode的多目标值
        self.training_data = []

        # 目标函数归一化参数
        self.objective_stats = {
            'freq_dev': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'volt_dev': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'damping': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'control_cost': {'mean': 0.0, 'std': 1.0, 'count': 0}
        }
        self.warmup_steps = config.get('warmup_steps', 200)

        self.total_it = 0

        print(f"算法配置: gamma={self.gamma}, tau={self.tau}, batch_size={self.batch_size}")
        print(
            f"多目标配置: 种群大小={self.nsga2_config['population_size']}, 目标数={self.nsga2_config['n_objectives']}")

    def _create_actor(self, obs_dim, act_dim):
        return nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )

    def _create_critic(self, obs_dim, act_dim):
        """创建Critic网络"""
        return nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _initialize_weight_vectors(self):
        """初始化多目标权重向量"""
        population_size = self.nsga2_config['population_size']
        n_objectives = self.nsga2_config['n_objectives']
        weight_vectors = []
        for _ in range(population_size):
            weights = np.random.dirichlet([1.0] * n_objectives)
            weight_vectors.append(weights)
        return weight_vectors

    def select_action(self, obs, explore=True):
        """选择动作"""
        actions = {}
        for i, agent in enumerate(self.agent_names):
            obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0).to(self.device)
            if torch.isnan(obs_tensor).any():
                raise ValueError(f"观测包含NaN: {obs[agent]}")

            with torch.no_grad():
                action = self.actors[i](obs_tensor).cpu().numpy()[0]

            if explore:
                noise = np.random.normal(0, self.noise_std, size=action.shape)
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
                action = action + noise

            action = np.clip(action, -1.0, 1.0)
            actions[agent] = action

        return actions

    def _compute_multi_objective_reward(self, rewards_dict, obs_dict, actions_dict):
        """计算多目标奖励，添加严格的数据有效性检查"""
        if not self.weight_vectors:
            return 0.0, {}

        current_weights = self.weight_vectors[self.current_weight_idx]

        # 提取多目标值
        freq_dev = 0.0
        volt_dev = 0.0
        damping = 0.0
        control_cost = 0.0

        try:
            # === 频率偏差 ===
            if hasattr(self.env, 'wcoi_idx'):
                wcoi = self.env.sim_case.dae.y[self.env.wcoi_idx].astype(np.float32)
                if len(wcoi) == 0:
                    raise ValueError("wcoi array is empty")
                freq_dev = np.mean(np.abs(wcoi - 1.0))
                if np.isnan(freq_dev) or np.isinf(freq_dev):
                    raise ValueError(f"freq_dev is NaN/inf: {freq_dev}")
            else:
                raise ValueError("env has no wcoi_idx")

            # === 电压偏差 ===
            if hasattr(self.env, 'voltage_idx'):
                voltage = self.env.sim_case.dae.y[self.env.voltage_idx].astype(np.float32)
                if len(voltage) == 0:
                    raise ValueError("voltage array is empty")
                volt_dev = np.mean(np.abs(voltage - 1.0))
                if np.isnan(volt_dev) or np.isinf(volt_dev):
                    raise ValueError(f"volt_dev is NaN/inf: {volt_dev}")
            else:
                raise ValueError("env has no voltage_idx")

            # === 阻尼比（Prony分析结果） ===
            if hasattr(self.env, 'prony_analyzers'):
                damping_vals = []
                for agent in self.agent_names:
                    analyzer = self.env.prony_analyzers[agent]
                    if analyzer and analyzer.last_analysis:
                        res = analyzer.last_analysis
                        if not res.get('valid', False):
                            raise ValueError(f"Prony analysis invalid for {agent}")
                        damping_vals.append(res['damping_ratio'])
                    else:
                        raise ValueError(f"No Prony analysis result for {agent}")
                if len(damping_vals) == 0:
                    raise ValueError("No damping ratios collected")
                damping = np.mean(damping_vals)
                if np.isnan(damping) or np.isinf(damping):
                    raise ValueError(f"damping is NaN/inf: {damping}")
            else:
                raise ValueError("env has no prony_analyzers")

            # === 控制代价（动作幅值） ===
            control_costs = []
            for agent in self.agent_names:
                if agent in actions_dict:
                    cost = np.mean(np.abs(actions_dict[agent]))
                    if np.isnan(cost) or np.isinf(cost):
                        raise ValueError(f"control cost for {agent} is NaN/inf: {cost}")
                    control_costs.append(cost)
                else:
                    raise ValueError(f"Action missing for {agent}")
            if len(control_costs) == 0:
                raise ValueError("No control costs collected")
            control_cost = np.mean(control_costs)

        except Exception as e:
            # 直接抛出原始异常，不允许静默失败
            raise RuntimeError(f"Error computing multi-objective reward: {e}") from e

        # 更新统计和归一化（保持不变）
        self._update_objective_stats(freq_dev, volt_dev, damping, control_cost)

        if self.objective_stats['freq_dev']['count'] > self.warmup_steps:
            freq_norm = (freq_dev - self.objective_stats['freq_dev']['mean']) / (
                        self.objective_stats['freq_dev']['std'] + 1e-6)
            volt_norm = (volt_dev - self.objective_stats['volt_dev']['mean']) / (
                        self.objective_stats['volt_dev']['std'] + 1e-6)
            damp_norm = (damping - self.objective_stats['damping']['mean']) / (
                        self.objective_stats['damping']['std'] + 1e-6)
            cost_norm = (control_cost - self.objective_stats['control_cost']['mean']) / (
                        self.objective_stats['control_cost']['std'] + 1e-6)
        else:
            freq_norm = freq_dev
            volt_norm = volt_dev
            damp_norm = damping
            cost_norm = control_cost

        # 加权奖励（负号表示最小化，阻尼比取正）
        weighted = (-current_weights[0] * freq_norm * 2.0 +
                    -current_weights[1] * volt_norm * 1.0 +
                    current_weights[2] * damp_norm * 10.0 +
                    -current_weights[3] * cost_norm * 0.5)

        scaled_reward = np.tanh(weighted) * 5.0

        return scaled_reward, {
            'freq_dev': freq_dev,
            'volt_dev': volt_dev,
            'damping': damping,
            'control_cost': control_cost,
            'weights': current_weights
        }

    def _update_objective_stats(self, freq_dev, volt_dev, damping, control_cost):
        """更新目标函数统计信息（带预热机制）"""
        alpha = 0.1  # 移动平均系数
        for key, val in zip(['freq_dev', 'volt_dev', 'damping', 'control_cost'],
                            [freq_dev, volt_dev, damping, control_cost]):
            stat = self.objective_stats[key]
            stat['count'] += 1
            if stat['count'] <= self.warmup_steps:
                # 预热阶段：直接累积平均
                if stat['count'] == 1:
                    stat['mean'] = val
                else:
                    stat['mean'] = (stat['mean'] * (stat['count'] - 1) + val) / stat['count']
                stat['std'] = 1.0  # 初始标准差设为1，避免除零
            else:
                # 正常运行阶段：使用指数移动平均
                old_mean = stat['mean']
                stat['mean'] = (1 - alpha) * old_mean + alpha * val
                stat['std'] = (1 - alpha) * stat['std'] + alpha * np.abs(val - old_mean)

    def update(self):
        """更新网络参数"""
        self.total_it += 1

        if self.buffer.size < self.batch_size:
            return None, None

        batch = self.buffer.sample(self.batch_size)

        # 转换为tensor
        obs = [torch.FloatTensor(batch['obs'][i]).to(self.device) for i in range(self.n_agents)]
        act = [torch.FloatTensor(batch['act'][i]).to(self.device) for i in range(self.n_agents)]
        rew = [torch.FloatTensor(batch['rew'][i]).to(self.device) for i in range(self.n_agents)]
        next_obs = [torch.FloatTensor(batch['next_obs'][i]).to(self.device) for i in range(self.n_agents)]
        done = [torch.FloatTensor(batch['done'][i]).to(self.device) for i in range(self.n_agents)]

        # 检查NaN
        for i in range(self.n_agents):
            if torch.isnan(obs[i]).any():
                raise ValueError(f"观测包含NaN，智能体索引 {i}, 观测: {obs[i]}")
            if torch.isnan(act[i]).any():
                raise ValueError(f"动作包含NaN，智能体索引 {i}, 动作: {act[i]}")
            if torch.isnan(rew[i]).any():
                raise ValueError(f"奖励包含NaN，智能体索引 {i}, 奖励: {rew[i]}")
            if torch.isnan(next_obs[i]).any():
                raise ValueError(f"下一观测包含NaN，智能体索引 {i}, 下一观测: {next_obs[i]}")

        obs_cat = torch.cat(obs, dim=1)
        act_cat = torch.cat(act, dim=1)
        next_obs_cat = torch.cat(next_obs, dim=1)

        critic_losses = []
        actor_losses = []

        for i in range(self.n_agents):
            # 目标Q值
            with torch.no_grad():
                next_actions = []
                for j in range(self.n_agents):
                    next_a = self.actor_targets[j](next_obs[j])
                    noise = torch.randn_like(next_a) * self.policy_noise
                    noise = noise.clamp(-self.noise_clip, self.noise_clip)
                    next_a = (next_a + noise).clamp(-1.0, 1.0)
                    next_actions.append(next_a)
                next_act_cat = torch.cat(next_actions, dim=1)

                target_q1 = self.critic_targets1[i](next_obs_cat, next_act_cat)
                target_q2 = self.critic_targets2[i](next_obs_cat, next_act_cat)
                target_q = torch.min(target_q1, target_q2)
                target_q = rew[i] + (1 - done[i]) * self.gamma * target_q

            # 当前Q值
            current_q1 = self.critics1[i](obs_cat, act_cat)
            current_q2 = self.critics2[i](obs_cat, act_cat)

            critic_loss1 = F.mse_loss(current_q1, target_q)
            critic_loss2 = F.mse_loss(current_q2, target_q)

            # 更新Critic1
            self.critic_optimizers1[i].zero_grad()
            critic_loss1.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critics1[i].parameters(), 0.5)#保持梯度裁剪阈值为 0.5 不变
            self.critic_optimizers1[i].step()

            # 更新Critic2
            self.critic_optimizers2[i].zero_grad()
            critic_loss2.backward()
            torch.nn.utils.clip_grad_norm_(self.critics2[i].parameters(), 0.5)
            self.critic_optimizers2[i].step()

            critic_losses.append((critic_loss1.item() + critic_loss2.item()) / 2)

            # 延迟策略更新
            if self.total_it % self.policy_freq == 0:
                # 当前动作
                current_actions = [self.actors[j](obs[j]) for j in range(self.n_agents)]
                current_act_cat = torch.cat(current_actions, dim=1)

                # Actor损失
                actor_loss = -self.critics1[i](obs_cat, current_act_cat).mean()

                # 多目标多样性正则（可选）
                if hasattr(self, 'weight_vectors') and len(self.weight_vectors) > 0:
                    action_std = torch.std(current_act_cat, dim=1).mean()
                    actor_loss = actor_loss - 0.01 * action_std  # 鼓励多样性

                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                self.actor_optimizers[i].step()

                actor_losses.append(actor_loss.item())

        # 软更新目标网络
        if self.total_it % self.policy_freq == 0:
            for i in range(self.n_agents):
                for param, target in zip(self.critics1[i].parameters(), self.critic_targets1[i].parameters()):
                    target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
                for param, target in zip(self.critics2[i].parameters(), self.critic_targets2[i].parameters()):
                    target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
                for param, target in zip(self.actors[i].parameters(), self.actor_targets[i].parameters()):
                    target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)

        avg_critic = np.mean(critic_losses) if critic_losses else 0.0
        avg_actor = np.mean(actor_losses) if actor_losses else 0.0
        return avg_critic, avg_actor

    def _nsga2_update(self):
        """使用 NSGA2 进化权重向量种群"""
        # 需要至少一定数量的性能数据
        total_samples = sum(len(v) for v in self.weight_performance.values())
        if total_samples < 10:
            return

        # 计算每个权重向量的平均性能（作为该个体的目标值）
        objectives = []
        valid_indices = []
        for idx, perfs in self.weight_performance.items():
            if len(perfs) == 0:
                continue
            # 取最近若干次性能的平均值（例如最近 5 次或全部）
            recent = perfs[-min(5, len(perfs)):]
            avg_obj = np.mean(recent, axis=0)
            objectives.append(avg_obj)
            valid_indices.append(idx)

        if len(objectives) < 2:
            return

        objectives = np.array(objectives)
        n_obj = objectives.shape[1]

        # ---------- NSGA2 核心步骤 ----------
        # 1. 非支配排序（所有目标均视为最小化，阻尼比取负）
        def dominates(a, b):
            # a 支配 b 当且仅当 a 在所有目标上不差于 b，且至少一个目标优于 b
            # 将阻尼比取负，使得所有目标均为最小化
            a_neg = np.copy(a)
            b_neg = np.copy(b)
            a_neg[2] = -a_neg[2]
            b_neg[2] = -b_neg[2]
            return np.all(a_neg <= b_neg) and np.any(a_neg < b_neg)

        n = len(objectives)
        domination_count = np.zeros(n)
        dominated_set = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if dominates(objectives[i], objectives[j]):
                    dominated_set[i].append(j)
                elif dominates(objectives[j], objectives[i]):
                    domination_count[i] += 1

        # 构建前沿面
        fronts = []
        front = [i for i in range(n) if domination_count[i] == 0]
        while front:
            fronts.append(front)
            next_front = []
            for i in front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            front = next_front

        # 2. 计算拥挤距离
        def crowding_distance(front_indices):
            if len(front_indices) <= 2:
                return [float('inf')] * len(front_indices)
            dist = np.zeros(len(front_indices))
            for obj_idx in range(n_obj):
                # 按目标值排序
                sorted_idx = sorted(front_indices, key=lambda i: objectives[i][obj_idx])
                # 边界点距离设为无穷大
                dist[0] = float('inf')
                dist[-1] = float('inf')
                min_val = objectives[sorted_idx[0]][obj_idx]
                max_val = objectives[sorted_idx[-1]][obj_idx]
                if max_val - min_val == 0:
                    continue
                # 中间点
                for k in range(1, len(sorted_idx) - 1):
                    prev = objectives[sorted_idx[k - 1]][obj_idx]
                    nxt = objectives[sorted_idx[k + 1]][obj_idx]
                    dist[k] += (nxt - prev) / (max_val - min_val)
            return dist

        # 为每个前沿计算拥挤距离
        crowding_distances = []
        for front in fronts:
            dist = crowding_distance(front)
            crowding_distances.extend(dist)

        # 3. 选择下一代（精英保留 + 交叉变异）
        # 当前种群大小
        pop_size = len(self.weight_vectors)

        # 将所有个体按 (前沿层级, 拥挤距离) 排序，以便选择精英
        individuals = []
        for idx_in_front, front in enumerate(fronts):
            for idx_in_individual, i in enumerate(front):
                rank = idx_in_front
                crowding = crowding_distances[sum(len(f) for f in fronts[:idx_in_front]) + idx_in_individual]
                individuals.append((valid_indices[i], rank, crowding, objectives[i]))

        # 排序：先按 rank 升序，再按 crowding 降序
        individuals.sort(key=lambda x: (x[1], -x[2]))

        # 精英选择：直接取前 pop_size 个
        selected_individuals = individuals[:pop_size]

        # 如果选择的数量不足 pop_size，则用交叉变异补充
        if len(selected_individuals) < pop_size:
            # 锦标赛选择函数（基于 rank 和 crowding）
            def tournament_select():
                # 随机选 3 个，取排名最好、拥挤距离最大的
                candidates = np.random.choice(len(selected_individuals), 3, replace=False)
                best = min(candidates, key=lambda c: (selected_individuals[c][1], -selected_individuals[c][2]))
                return selected_individuals[best][0]  # 返回权重向量的原始索引

            # 生成子代
            children_weights = []
            while len(selected_individuals) + len(children_weights) < pop_size:
                idx1 = tournament_select()
                idx2 = tournament_select()
                w1 = self.weight_vectors[idx1]
                w2 = self.weight_vectors[idx2]
                child1, child2 = self._crossover_weights(w1, w2)
                child1 = self._mutate_weights(child1)
                child2 = self._mutate_weights(child2)
                children_weights.extend([child1, child2])
            # 将子代加入选择列表
            for cw in children_weights:
                selected_individuals.append((None, 0, 0, None))  # 占位，实际权重在后续处理
        else:
            children_weights = []

        # 构建新的权重向量列表
        new_weight_vectors = []
        new_idx_mapping = {}  # 旧索引 -> 新索引（用于保留性能历史）

        # 先添加精英个体
        for i, (old_idx, rank, crowd, obj) in enumerate(selected_individuals[:pop_size - len(children_weights)]):
            if old_idx is not None:
                new_weight_vectors.append(self.weight_vectors[old_idx])
                new_idx_mapping[old_idx] = i
            else:
                # 不应该出现，因为精英部分都是原有索引
                pass

        # 再添加子代
        for cw in children_weights[:pop_size - len(new_weight_vectors)]:
            new_weight_vectors.append(cw)

        # 更新权重向量
        self.weight_vectors = new_weight_vectors

        # 重建性能记录映射
        new_weight_performance = {}
        for new_idx, w in enumerate(self.weight_vectors):
            # 尝试查找是否由精英保留而来
            found = False
            for old_idx, mapped_new_idx in new_idx_mapping.items():
                if mapped_new_idx == new_idx:
                    new_weight_performance[new_idx] = self.weight_performance[old_idx]
                    found = True
                    break
            if not found:
                # 新个体，清空历史
                new_weight_performance[new_idx] = []

        self.weight_performance = new_weight_performance
        self.current_weight_idx = 0  # 重置当前权重索引

    def _crossover_weights(self, w1, w2):
        alpha = np.random.random()
        child = alpha * w1 + (1 - alpha) * w2
        child = child / child.sum()
        return child

    def _mutate_weights(self, w):
        rate = self.nsga2_config['mutation_rate']
        std = self.nsga2_config['mutation_std']
        mutated = w.copy()
        for i in range(len(w)):
            if np.random.rand() < rate:
                mutated[i] += np.random.normal(0, std)
        mutated = np.clip(mutated, 0.01, 1.0)
        mutated = mutated / mutated.sum()
        return mutated

    def _extract_pareto_front(self, objectives):
        """从多目标值中提取帕累托前沿（最小化问题，阻尼比已取负）"""
        n = len(objectives)
        pareto = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                # 将阻尼比取负，统一为最小化
                obj_i_neg = objectives[i].copy()
                obj_j_neg = objectives[j].copy()
                obj_i_neg[2] = -obj_i_neg[2]
                obj_j_neg[2] = -obj_j_neg[2]
                if np.all(obj_j_neg <= obj_i_neg):
                    dominated = True
                    break
            if not dominated:
                pareto.append(i)
        return objectives[pareto]

    def train(self, n_episodes, save_path):
        """训练算法"""
        os.makedirs(save_path, exist_ok=True)

        data = {
            'episode': [], 'reward': [], 'success': [],
            'avg_damping_ratio': [], 'avg_oscillation_freq': [],
            'freq_deviation': [], 'voltage_deviation': [],
            'control_cost': [], 'weight_idx': [],
            'critic_loss': [], 'actor_loss': [], 'buffer_size': []
        }

        print(f"开始训练NSGA2_MATD3_Mix算法，共{n_episodes}个episodes...")

        for episode in range(n_episodes):
            # 修改点4：噪声退火
            progress = episode / max(1, n_episodes - 1)
            self.noise_std = max(0.01, self.noise_std_initial * (1 - progress))

            obs, info = self.env.reset()
            episode_reward = {agent: 0.0 for agent in self.agent_names}
            episode_success = False

            freq_deviations = []
            voltage_deviations = []
            damping_ratios = []
            osc_freqs = []
            control_costs = []
            critic_losses = []
            actor_losses = []

            step = 0
            crashed = False

            while step < self.env.max_steps:
                try:
                    actions = self.select_action(obs, explore=True)
                    next_obs, rewards, dones, truncs, infos = self.env.step(actions)

                    # 检查崩溃
                    if infos and any(info.get('sim_crashed', False) for info in infos.values()):
                        crashed = True
                        print(f"Episode {episode}: 仿真在步数{step}崩溃")
                        break

                    # 计算多目标奖励
                    mo_reward, mo_metrics = self._compute_multi_objective_reward(rewards, obs, actions)
                    # 为每个智能体分配相同奖励
                    for agent in self.agent_names:
                        rewards[agent] = mo_reward / self.n_agents

                    # 存储经验
                    obs_list = [obs[agent] for agent in self.agent_names]
                    act_list = [actions[agent] for agent in self.agent_names]
                    rew_list = [rewards[agent] for agent in self.agent_names]
                    next_obs_list = [next_obs[agent] for agent in self.agent_names]
                    done_list = [dones[agent] for agent in self.agent_names]
                    self.buffer.store(obs_list, act_list, rew_list, next_obs_list, done_list)

                    # 累加奖励
                    for agent in self.agent_names:
                        episode_reward[agent] += rewards[agent]

                    # 收集系统数据
                    if hasattr(self.env, 'wcoi_idx'):
                        try:
                            wcoi = self.env.sim_case.dae.y[self.env.wcoi_idx].astype(np.float32)
                            freq_deviations.append(np.mean(np.abs(wcoi - 1.0)))
                        except:
                            pass
                    if hasattr(self.env, 'voltage_idx'):
                        try:
                            volt = self.env.sim_case.dae.y[self.env.voltage_idx].astype(np.float32)
                            voltage_deviations.append(np.mean(np.abs(volt - 1.0)))
                        except:
                            pass
                    if hasattr(self.env, 'prony_analyzers'):
                        for agent in self.agent_names:
                            ana = self.env.prony_analyzers[agent]
                            if ana and ana.last_analysis:
                                damping_ratios.append(ana.last_analysis['damping_ratio'])
                                osc_freqs.append(ana.last_analysis['oscillation_freq'])
                    control_costs.append(mo_metrics['control_cost'])

                    # 更新网络
                    if self.buffer.size >= self.batch_size:
                        cl, al = self.update()
                        if cl is not None:
                            critic_losses.append(cl)
                        if al is not None:
                            actor_losses.append(al)

                    obs = next_obs
                    step += 1

                    if all(dones.values()) or all(truncs.values()):
                        episode_success = not crashed
                        break

                except Exception as e:
                    print(f"Episode {episode}, Step {step} 出错: {e}")
                    crashed = True
                    break

            # 统计
            if crashed:
                total_reward = self.crash_penalty
                avg_freq = avg_volt = avg_damp = avg_osc = avg_cost = 0.0
                avg_cl = avg_al = 0.0
                success = 0
            else:
                total_reward = sum(episode_reward.values())
                avg_freq = np.nanmean(freq_deviations) if freq_deviations else 0.0
                avg_volt = np.nanmean(voltage_deviations) if voltage_deviations else 0.0
                avg_damp = np.nanmean(damping_ratios) if damping_ratios else 0.0
                avg_osc = np.nanmean(osc_freqs) if osc_freqs else 0.0
                avg_cost = np.nanmean(control_costs) if control_costs else 0.0
                avg_cl = np.nanmean(critic_losses) if critic_losses else 0.0
                avg_al = np.nanmean(actor_losses) if actor_losses else 0.0
                success = 1

            self.episode_rewards.append(total_reward)
            if not np.isnan(avg_freq) and not np.isnan(avg_volt) and not np.isnan(avg_damp) and not np.isnan(avg_cost):
                self.multi_objective_values.append([avg_freq, avg_volt, avg_damp, avg_cost])

            # 记录数据
            data['episode'].append(episode)
            data['reward'].append(total_reward)
            data['success'].append(success)
            data['avg_damping_ratio'].append(avg_damp)
            data['avg_oscillation_freq'].append(avg_osc)
            data['freq_deviation'].append(avg_freq)
            data['voltage_deviation'].append(avg_volt)
            data['control_cost'].append(avg_cost)
            data['weight_idx'].append(self.current_weight_idx)
            data['critic_loss'].append(avg_cl)
            data['actor_loss'].append(avg_al)
            data['buffer_size'].append(self.buffer.size)

            # NSGA2权重更新
            if (episode % self.weight_update_freq == 0 and episode > 0 and
                    len(self.multi_objective_values) > 10):
                self._nsga2_update()
                self.current_weight_idx = (self.current_weight_idx + 1) % len(self.weight_vectors)

            # 定期保存
            if episode % 10 == 0 or episode == n_episodes - 1:
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(save_path, 'training_data.csv'), index=False)
                self.save_model(save_path, episode)

                w = self.weight_vectors[self.current_weight_idx] if self.weight_vectors else [0.25]*4
                print(f"NSGA2_MATD3_Mix - Episode {episode}: "
                      f"奖励={total_reward:.2f}, 阻尼比={avg_damp:.4f}, 频率偏差={avg_freq:.4f}, "
                      f"权重={w}, 成功={success}, Critic损失={avg_cl:.4f}, Actor损失={avg_al:.4f}")

        # 最终保存
        self.save_model(save_path, 'final')
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_path, 'training_data.csv'), index=False)
        # 保存配置...
        env_config = {
            'tf': self.env.tf,
            'tstep': self.env.tstep,
            'max_steps': self.env.max_steps,
            'include_prony': self.env.include_prony,
            'shared_observations': self.env.shared_observations,
            'prony_coordination': self.env.prony_coordination,
            'algorithm_type': self.env.algorithm_type
        }
        algo_config = {
            'gamma': self.gamma,
            'tau': self.tau,
            'noise_std': self.noise_std_initial,
            'noise_clip': self.noise_clip,
            'policy_noise': self.policy_noise,
            'policy_freq': self.policy_freq,
            'batch_size': self.batch_size,
            'actor_lr': self.config.get('actor_lr', 1e-4),
            'critic_lr': self.config.get('critic_lr', 1e-3),
            'nsga2_population_size': self.nsga2_config['population_size'],
            'weight_update_freq': self.weight_update_freq
        }
        save_config_to_json(save_path, env_config, algo_config)

        # 保存Pareto历史
        if self.pareto_front_history:
            with open(os.path.join(save_path, 'pareto_front_history.pkl'), 'wb') as f:
                pickle.dump(self.pareto_front_history, f)

        print(f"NSGA2_MATD3_Mix训练完成！数据保存到: {save_path}")
        return data

    def save_model(self, path, episode):
        """保存模型"""
        model_dict = {
            'actors': [self.actors[i].state_dict() for i in range(self.n_agents)],
            'critics1': [self.critics1[i].state_dict() for i in range(self.n_agents)],
            'critics2': [self.critics2[i].state_dict() for i in range(self.n_agents)],
            'actor_targets': [self.actor_targets[i].state_dict() for i in range(self.n_agents)],
            'critic_targets1': [self.critic_targets1[i].state_dict() for i in range(self.n_agents)],
            'critic_targets2': [self.critic_targets2[i].state_dict() for i in range(self.n_agents)],
            'weight_vectors': self.weight_vectors,
            'current_weight_idx': self.current_weight_idx,
            'multi_objective_values': self.multi_objective_values,
            'pareto_front_history': self.pareto_front_history,
            'obs_dims': self.obs_dims,
            'act_dims': self.act_dims,
            'total_obs_dim': self.total_obs_dim,
            'total_act_dim': self.total_act_dim,
            'agent_names': self.agent_names,
            'config': self.config
        }
        torch.save(model_dict, os.path.join(path, f'model_{episode}.pth'))
        print(f"模型保存成功: {os.path.join(path, f'model_{episode}.pth')}")

def run_all_algorithms(main_folder):
    """运行所有10种算法的训练"""
    # 创建主文件夹
    main_folder = main_folder

    # 环境配置（所有算法共享）
    env_config = {
        'tf': 15.0,
        'tstep': 1 / 30,
        'max_steps': 28,
        'include_prony': True,
        'prony_coordination': True
    }

    # 算法特定配置 - 减小学习率避免NaN
    algorithm_configs = {

        'MATD3': {
            'buffer_size': 50000,
            'batch_size': 256,
            'actor_lr': 5e-5,#1e-4
            'critic_lr': 3e-4,#1e-3
            'gamma': 0.99,
            'tau': 0.01,
            'noise_std': 0.05,#0.02
            'noise_clip': 0.5,#0.3
            'policy_noise': 0.1,#0.1
            'policy_freq': 2,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        },
        'MAPPO': {
            'actor_lr': 3e-4,
            'critic_lr': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.1,
            'ppo_epochs': 5,
            'batch_size': 32,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        },
        'MASAC': {
            'buffer_size': 50000,
            'batch_size': 64,
            'actor_lr': 3e-4,
            'critic_lr': 3e-3,
            'alpha_lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.01,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        },

        'WSE_MATD3': {
            'buffer_size': 50000,
            'batch_size': 256,#64
            'actor_lr': 5e-5,#1e-4
            'critic_lr': 3e-4,#1e-3
            'gamma': 0.99,
            'tau': 0.01,
            'noise_std': 0.05,  # 减小探索噪声0.02
            'noise_clip': 0.5,#0.3
            'policy_noise': 0.1,#0.1
            'policy_freq': 2,
            'nsga2_population_size': 10,
            'nsga2_crossover_rate': 0.8,
            'nsga2_mutation_rate': 0.05,
            'nsga2_mutation_std': 0.05,
            'weight_update_freq': 10,  # 延长更新周期
            'crash_penalty': -10.0,  # 降低崩溃惩罚
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        },
        'NSGA2_MATD3': {
            'buffer_size': 50000,
            'batch_size': 256,#64
            'actor_lr': 5e-5,#1e-4
            'critic_lr': 3e-4,#1e-3
            'gamma': 0.99,
            'tau': 0.01,
            'noise_std': 0.05,  # 减小探索噪声0.02
            'noise_clip': 0.5,#0.3
            'policy_noise': 0.1,#0.1
            'policy_freq': 2,
            'nsga2_population_size': 10,
            'nsga2_crossover_rate': 0.8,
            'nsga2_mutation_rate': 0.05,
            'nsga2_mutation_std': 0.05,
            'weight_update_freq': 10,  # 延长更新周期
            'crash_penalty': -10.0,  # 降低崩溃惩罚
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        }
    }

    # 算法列表（10种算法）
    algorithms = [
        ('MAPPO', make_mappo_env, MAPPO),
        ('MASAC', make_masac_env, MASAC),
        ('MATD3', make_matd3_env, MATD3),
        ('WSE_MATD3', make_wse_matd3_env, WSE_MATD3),
        ('NSGA2_MATD3', make_nsga2_matd3_mix_env, NSGA2_MATD3),

    ]

    # 训练参数 - 减少episodes数
    n_episodes = 100

    # 存储所有算法的结果
    all_results = {}

    for algo_name, env_maker, algo_class in algorithms:
        print(f"\n{'=' * 80}")
        print(f"开始训练 {algo_name} 算法")
        print(f"{'=' * 80}")

        try:
            # 创建算法文件夹
            algo_folder = os.path.join(main_folder, algo_name)
            os.makedirs(algo_folder, exist_ok=True)

            # 根据算法类型调整环境配置
            if algo_name == 'MASAC':
                current_env_config = env_config.copy()
                current_env_config['shared_observations'] = False
                current_env_config['algorithm_type'] = algo_name
                print(f"配置: MASAC算法，禁用共享观测")

            else:
                current_env_config = env_config.copy()
                current_env_config['shared_observations'] = True
                current_env_config['algorithm_type'] = algo_name
                print(f"配置: {algo_name}算法，启用共享观测")

            # 创建环境
            print(f"创建 {algo_name} 环境...")
            env = env_maker(current_env_config)

            # 获取算法配置
            algo_config = algorithm_configs[algo_name]

            # 创建算法实例
            print(f"初始化 {algo_name} 算法...")
            algo = algo_class(env, algo_config)

            # 训练算法
            print(f"\n开始 {algo_name} 训练，共 {n_episodes} 个episodes...")
            start_time = time.time()

            results = algo.train(n_episodes, algo_folder)

            training_time = time.time() - start_time
            print(f"{algo_name} 训练完成！耗时: {training_time:.2f} 秒")

            # 保存结果
            all_results[algo_name] = results

            # 保存配置
            save_config_to_json(algo_folder, current_env_config, algo_config)

            # 清理内存
            del algo
            del env
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"{algo_name} 训练过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()

            # 记录错误信息
            error_file = os.path.join(main_folder, f"{algo_name}_error.txt")
            with open(error_file, 'w') as f:
                f.write(f"Algorithm: {algo_name}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(traceback.format_exc())

    print(f"\n{'=' * 80}")
    print("所有算法训练完成！")
    print(f"{'=' * 80}")

    # 生成训练摘要
    summary_file = os.path.join(main_folder, "training_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("多智能体算法训练摘要\n")
        f.write("=" * 50 + "\n\n")

        for algo_name in all_results.keys():
            f.write(f"算法: {algo_name}\n")
            f.write(f"状态: 成功\n")
            if algo_name in all_results and all_results[algo_name] is not None:
                df = pd.DataFrame(all_results[algo_name])
                if 'reward' in df.columns:
                    avg_reward = df['reward'].mean()
                    f.write(f"平均奖励: {avg_reward:.2f}\n")
                if 'success' in df.columns:
                    success_rate = df['success'].mean() * 100
                    f.write(f"成功率: {success_rate:.1f}%\n")
            f.write("\n")

    print(f"训练摘要已保存到: {summary_file}")

    return all_results
# ============================================================================
# 可视化函数
# ============================================================================

def visualize_comparison(main_folder):
    """可视化比较所有算法的性能（生成9张独立图片）"""
    main_folder = main_folder

    # 算法名称列表（按需调整顺序）
    algorithms = ['MAPPO', 'MASAC', 'MATD3', 'NSGA2_MATD3', 'WSE_MATD3']

    # 算法颜色、线型、标记定义
    algorithm_colors = {
        'MAPPO': '#bcbd22',
        'MATD3': '#ff7f0e',
        'MASAC': '#d62728',
        'NSGA2_MATD3': '#9467bd',
        'WSE_MATD3': '#4d4d4d',
    }
    algorithm_linestyles = {
        'MATD3': ':',
        'MAPPO': (0, (5, 1)),
        'MASAC': (0, (3, 1, 1, 1, 1, 1)),
        'NSGA2_MATD3': (0, (3, 1, 1, 1)),
        'WSE_MATD3': '-',
    }
    algorithm_markers = {
        'MATD3': 's',
        'MAPPO': '^',
        'MASAC': 'D',
        'NSGA2_MATD3': 'v',
        'WSE_MATD3': 'p',
    }

    # 加载并处理每个算法的数据
    all_data = {}
    for algo in algorithms:
        data_path = os.path.join(main_folder, algo, 'training_data.csv')
        if os.path.exists(data_path):
            try:
                df = pd.read_csv(data_path)
                all_data[algo] = df
                print(f"成功加载{algo}数据，形状: {df.shape}")
            except Exception as e:
                print(f"加载{algo}数据失败: {e}")
                all_data[algo] = None
        else:
            all_data[algo] = None

    # 过滤出有数据的算法
    valid_algorithms = [algo for algo in algorithms if all_data[algo] is not None]

    def save_figure(filename, figsize, draw_func):
        """保存图片为 PNG 和 PDF，dpi=1500，字体统一"""
        # 创建新图形并应用全局字体设置
        fig = plt.figure(figsize=figsize, dpi=1500)
        ax = draw_func(fig) if draw_func.__code__.co_argcount else draw_func()
        # 若 draw_func 返回了 axes，可进一步设置，但通常直接绘制
        plt.tight_layout()
        # 保存 PNG
        png_path = os.path.join(main_folder, filename.replace('.png', '.png'))
        plt.savefig(png_path, dpi=1500, bbox_inches='tight')
        # 保存 PDF
        pdf_path = os.path.join(main_folder, filename.replace('.png', '.pdf'))
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ 图片已保存: {png_path} 和 {pdf_path}")

    def plot_freq_volt_dev():

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=1500)

        # 获取有效算法列表
        valid_algos = [algo for algo in algorithms if all_data.get(algo) is not None]

        handles = []
        labels = []

        # ----- 子图1：频率偏差 -----
        for algo in valid_algos:
            df = all_data[algo]
            x_col = next((col for col in ['episode', 'generation', 'iteration'] if col in df.columns), None)
            freq_col = next((col for col in ['best_freq_dev', 'freq_deviation', 'freq_dev'] if col in df.columns), None)
            if x_col and freq_col:
                line, = ax1.plot(df[x_col], df[freq_col],
                                 label=algo,
                                 color=algorithm_colors.get(algo, '#000000'),
                                 linestyle= '-',  #algorithm_linestyles.get(algo, '-'),
                                 linewidth=2, alpha=0.8)
                handles.append(line)
                labels.append(algo)
        ax1.set_ylabel('Frequency Deviation/p.u.', fontsize=14)
        ax1.yaxis.set_label_coords(-0.08, 0.5)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 100)
        # ----- 子图2：电压偏差 -----
        for algo in valid_algos:
            df = all_data[algo]
            x_col = next((col for col in ['episode', 'generation', 'iteration'] if col in df.columns), None)
            volt_col = next((col for col in ['best_volt_dev', 'voltage_deviation', 'volt_dev'] if col in df.columns),
                            None)
            if x_col and volt_col:
                ax2.plot(df[x_col], df[volt_col],
                         color=algorithm_colors.get(algo, '#000000'),
                         linestyle='-',  #algorithm_linestyles.get(algo, '-'),
                         linewidth=2, alpha=0.8)
        ax2.set_xlabel('Episode', fontsize=14)
        ax2.set_ylabel('Voltage Deviation/p.u.', fontsize=14)
        ax2.yaxis.set_label_coords(-0.08, 0.5)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 100)
        # 去重（确保每个算法只出现一次）
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels):
            if l not in unique_labels:
                unique_handles.append(h)
                unique_labels.append(l)

        # 将图例中的下划线替换为连字符，用于显示
        display_labels = [l.replace('_', '-') for l in unique_labels]

        # 将图例放在整个图形的顶部右侧（外部），一行5列，紧凑设置
        fig.legend(unique_handles, display_labels,
                   loc='upper right', bbox_to_anchor=(0.98, 0.98),
                   ncol=5, fontsize=12, frameon=False,
                   handletextpad=0.5, columnspacing=1.2)

        # 调整子图布局，为顶部图例留出空间
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存图片
        png_path = os.path.join(main_folder, 'freq_volt_deviation_combined.png')
        pdf_path = os.path.join(main_folder, 'freq_volt_deviation_combined.pdf')
        fig.savefig(png_path, dpi=1500, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ 频率-电压偏差合并图已保存: {png_path} 和 {pdf_path}")

    plot_freq_volt_dev()


    def draw_final_reward_boxplot():
        """
        绘制最终奖励箱线图，横坐标位置非等间距：长名称（NSGA2_MATD3, WSE_MATD3）之间间距更大，
        短名称之间间距较小，以改善标签显示。
        """
        final_rewards = []
        labels = []
        for algo in valid_algorithms:
            df = all_data[algo]
            reward_col = next((col for col in ['reward', 'Reward', 'total_reward'] if col in df.columns), None)
            if reward_col:
                n = len(df)
                if n > 0:
                    final_data = df[reward_col][int(0.8 * n):]
                    if len(final_data) > 0:
                        final_rewards.append(final_data.values)
                        labels.append(algo)

        if not final_rewards:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=1500)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            return fig

        # 动态计算非等间距的 x 坐标位置
        positions = [1.0]  # 第一个箱线图的位置
        for i in range(1, len(labels)):
            prev_len = len(labels[i - 1])
            curr_len = len(labels[i])
            extra = 0.8 if (prev_len > 9 or curr_len > 9) else 0.2
            positions.append(positions[-1] + 1.0 + extra)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=1500)  # 高度从6减小到5
        bp = ax.boxplot(final_rewards, positions=positions, labels=labels, patch_artist=True)

        for i, patch in enumerate(bp['boxes']):
            base_color = algorithm_colors.get(labels[i], '#000000')
            rgba_color = to_rgba(base_color, alpha=0.6)
            patch.set_facecolor(rgba_color)

        display_labels = [label.replace('_', '-') for label in labels]
        ax.set_ylabel('Final Reward Distribution', fontsize=14)
        ax.set_xlabel('Algorithm', fontsize=14)  # 新增横轴标签
        ax.set_xticks(positions)
        ax.set_xticklabels(display_labels, rotation=0, fontsize=14)
        ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
        ax.grid(True, alpha=0.3, axis='y')

        return fig
    save_figure('final_reward_boxplot.png', (8, 6), draw_final_reward_boxplot)

    summary_data = []
    for algo, df in all_data.items():
        if df is not None:
            row = {'Algorithm': algo}
            reward_col = next((col for col in ['reward', 'Reward', 'total_reward'] if col in df.columns), None)
            if reward_col:
                row['Avg Reward'] = df[reward_col].mean()
                row['Std Reward'] = df[reward_col].std()
                row['Max Reward'] = df[reward_col].max()
            if 'success' in df.columns:
                row['Success Rate'] = df['success'].mean() * 100
            damping_col = next((col for col in ['avg_damping_ratio', 'best_damping', 'damping'] if col in df.columns), None)
            if damping_col:
                row['Avg Damping'] = df[damping_col].mean()
                row['Max Damping'] = df[damping_col].max()
            osc_col = next((col for col in ['avg_oscillation_freq', 'osc_freqs'] if col in df.columns), None)
            if osc_col:
                row['Avg Osc Freq'] = df[osc_col].mean()
            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(main_folder, 'algorithm_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"汇总表格已保存到: {summary_path}")
    print("\n算法性能汇总:")
    print(summary_df.to_string())

    return summary_df
if __name__ == "__main__":
    print("IEEE 39-bus多智能体控制算法对比训练")
    print("=" * 60)

    # 创建主文件夹
    main_folder = "MARL_Train"
    os.makedirs(main_folder, exist_ok=True)

    print("\n开始训练所有8种算法...")
    # all_results = run_all_algorithms(main_folder)
    # 可视化比较
    try:
        print("\n生成算法比较图...")
        summary_df = visualize_comparison(main_folder)

        # 保存比较数据
        comparison_file = os.path.join(main_folder, "algorithm_comparison.csv")
        summary_df.to_csv(comparison_file, index=False)
        print(f"比较数据已保存到: {comparison_file}")

    except Exception as e:
        print(f"可视化比较失败: {e}")


