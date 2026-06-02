import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
# 环境设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# 设置中文字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# 导入环境和算法类
from andes_marl import (
    make_matd3_env, make_mappo_env, make_masac_env,
    make_nsga2_matd3_mix_env,make_wse_matd3_env

)

from MARL_train import (MAPPO, MASAC,MATD3,NSGA2_MATD3,WSE_MATD3)


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class ScaleLayer(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class EnhancedModelLoader:
    """增强的模型加载器 - 修复版本"""

    def __init__(self, model_base_path: str = "MARL_Train0314v1"):
        self.model_base_path = model_base_path
        self.loaded_models = {}

    def load_all_models(self, algorithms: List[str]):
        """加载所有算法的最新模型"""
        for algorithm_name in algorithms:
            try:
                model_path = os.path.join(self.model_base_path, algorithm_name)

                if not os.path.exists(model_path):
                    print(f"警告: {algorithm_name} 模型路径不存在: {model_path}")
                    continue

                # 查找最新的模型文件
                model_files = []
                for file in os.listdir(model_path):
                    if file.startswith('model_') and (file.endswith('.pth') or file.endswith('.pkl')):
                        model_files.append(os.path.join(model_path, file))

                if not model_files:
                    print(f"警告: {algorithm_name} 没有找到模型文件")
                    continue

                # 优先使用final模型，否则使用数字最大的模型
                final_models = [f for f in model_files if 'final' in f]
                if final_models:
                    latest_model = final_models[0]
                else:
                    # 按文件名排序，选择数字最大的
                    def extract_number(f):
                        import re
                        match = re.search(r'model_(\d+)', f)
                        return int(match.group(1)) if match else 0

                    model_files.sort(key=extract_number, reverse=True)
                    latest_model = model_files[0]

                print(f"加载 {algorithm_name} 模型: {os.path.basename(latest_model)}")

                # 加载模型
                if latest_model.endswith('.pth'):
                    model_dict = torch.load(latest_model, map_location=device)
                else:
                    with open(latest_model, 'rb') as f:
                        model_dict = pickle.load(f)

                self.loaded_models[algorithm_name] = model_dict
                print(f"{algorithm_name} 模型加载成功")

            except Exception as e:
                print(f"加载 {algorithm_name} 模型失败: {e}")
                import traceback
                traceback.print_exc()

        return self.loaded_models

    def get_algorithm_config(self, algorithm_name: str) -> Dict:
        """获取算法配置"""
        base_configs = {
            'MAPPO': {
                'actor_lr': 3e-4,
                'critic_lr': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'ppo_epochs': 10,
                'batch_size': 64,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'device': device
            },
            'MASAC': {
                'buffer_size': 50000,
                'batch_size': 256,
                'actor_lr': 3e-4,
                'critic_lr': 3e-3,
                'alpha_lr': 3e-4,
                'gamma': 0.99,
                'tau': 0.01,
                'device': device
            },
            'MATD3': {
                'buffer_size': 50000,
                'batch_size': 256,
                'actor_lr': 1e-4,
                'critic_lr': 1e-3,
                'gamma': 0.99,
                'tau': 0.01,
                'noise_std': 0.1,
                'noise_clip': 0.5,
                'policy_noise': 0.2,
                'policy_freq': 2,
                'device': device
            },

            'NSGA2_MATD3': {
                'buffer_size': 50000,
                'batch_size': 256,
                'actor_lr': 1e-4,
                'critic_lr': 1e-3,
                'gamma': 0.99,
                'tau': 0.01,
                'noise_std': 0.1,
                'noise_clip': 0.5,
                'policy_noise': 0.2,
                'policy_freq': 2,
                'nsga2_population_size': 10,
                'nsga2_crossover_rate': 0.8,
                'nsga2_mutation_rate': 0.05,
                'nsga2_mutation_std': 0.05,
                'weight_update_freq': 10,
                'crash_penalty': -10.0,  # 降低崩溃惩罚
                'device': device
            },

            'WSE_MATD3': {
                'buffer_size': 50000,
                'batch_size': 256,  # 64
                'actor_lr': 5e-5,  # 1e-4
                'critic_lr': 3e-4,  # 1e-3
                'gamma': 0.99,
                'tau': 0.01,
                'noise_std': 0.05,  # 减小探索噪声0.02
                'noise_clip': 0.5,  # 0.3
                'policy_noise': 0.1,  # 0.1
                'policy_freq': 2,
                'nsga2_population_size': 10,
                'nsga2_crossover_rate': 0.8,
                'nsga2_mutation_rate': 0.05,
                'nsga2_mutation_std': 0.05,
                'weight_update_freq': 10,  # 延长更新周期
                'crash_penalty': -10.0,  # 降低崩溃惩罚
                'device': device
            },
        }

        return base_configs.get(algorithm_name, {})

class EnhancedMARL_Tester:
    def __init__(self, algorithms: List[str], model_base_path: str, result_dir: str):
        self.algorithms = algorithms
        self.model_base_path = model_base_path
        self.result_dir = result_dir
        self.model_loader = EnhancedModelLoader(model_base_path)
        self.loaded_models = self.model_loader.load_all_models(algorithms)
        self.test_results = {}
        self.simulation_data = {}
        os.makedirs(self.result_dir, exist_ok=True)   # 现在 result_dir 是传入的参数

        # 环境配置
        self.env_config = {
            'tf': 15.0,
            'tstep': 1 / 30,
            'max_steps': 28,
            'include_prony': True,
            'shared_observations': True,
            'prony_coordination': True
        }

        print(f"初始化增强MARL测试器，测试算法: {self.algorithms}")

    def create_environment(self, algorithm_name: str):
        """创建环境（与原代码相同）"""
        try:
            env_config = self.env_config.copy()

            # 根据算法类型调整环境配置
            if algorithm_name == "MASAC":
                env_config['shared_observations'] = False
                env_config['algorithm_type'] = algorithm_name
            else:
                env_config['algorithm_type'] = algorithm_name

            # 创建环境
            if algorithm_name == "MATD3":
                env = make_matd3_env(env_config)
            elif algorithm_name == "MAPPO":
                env = make_mappo_env(env_config)
            elif algorithm_name == "MASAC":
                env = make_masac_env(env_config)
            elif algorithm_name == "WSE_MATD3":
                env = make_wse_matd3_env(env_config)
            elif algorithm_name == "NSGA2_MATD3":
                env = make_nsga2_matd3_mix_env(env_config)
            else:
                raise ValueError(f"未知算法: {algorithm_name}")

            print(f"为{algorithm_name}创建环境成功")
            return env

        except Exception as e:
            print(f"创建环境失败 ({algorithm_name}): {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_algorithm_instance(self, algorithm_name: str, env):
        """根据算法名称创建算法实例并加载训练好的模型"""
        if algorithm_name not in self.loaded_models:
            print(f"错误：未找到算法 {algorithm_name} 的模型数据")
            return None

        model_dict = self.loaded_models[algorithm_name]
        config = self.model_loader.get_algorithm_config(algorithm_name)

        # 根据算法名称实例化对应的类
        try:

            if algorithm_name == "MATD3":

                algo = MATD3(env, config)
                for i in range(algo.n_agents):
                    algo.actors[i].load_state_dict(model_dict['actors'][i])
                    algo.actor_targets[i].load_state_dict(model_dict['actors'][i])  # 注意：目标网络可用相同权重
                    algo.critics1[i].load_state_dict(model_dict['critics1'][i])
                    algo.critics2[i].load_state_dict(model_dict['critics2'][i])
                print(f"MATD3模型加载成功")

            elif algorithm_name == "MAPPO":

                algo = MAPPO(env, config)
                for i in range(algo.n_agents):
                    algo.actors[i].load_state_dict(model_dict['actors'][i])
                algo.critic.load_state_dict(model_dict['critic'])
                print(f"MAPPO模型加载成功")

            elif algorithm_name == "MASAC":

                algo = MASAC(env, config)
                for i in range(algo.n_agents):
                    algo.actors[i].load_state_dict(model_dict['actors'][i])
                    algo.critics1[i].load_state_dict(model_dict['critics1'][i])
                    algo.critics2[i].load_state_dict(model_dict['critics2'][i])
                    algo.critic_targets1[i].load_state_dict(model_dict['critic_targets1'][i])
                    algo.critic_targets2[i].load_state_dict(model_dict['critic_targets2'][i])
                algo.log_alpha = model_dict['log_alpha']
                print(f"MASAC模型加载成功")

            elif algorithm_name == "WSE_MATD3":

                algo = WSE_MATD3(env, config)
                for i in range(algo.n_agents):
                    algo.actors[i].load_state_dict(model_dict['actors'][i])
                # 其他参数如 weight_vectors 等
                if 'weight_vectors' in model_dict:
                    algo.weight_vectors = model_dict['weight_vectors']
                print(f"WSE_MATD3模型加载成功")

            elif algorithm_name == "NSGA2_MATD3":

                algo = NSGA2_MATD3(env, config)
                for i in range(algo.n_agents):
                    algo.actors[i].load_state_dict(model_dict['actors'][i])
                # 其他参数如 weight_vectors 等
                if 'weight_vectors' in model_dict:
                    algo.weight_vectors = model_dict['weight_vectors']
                print(f"NSGA2_MATD3_MIX模型加载成功")

            else:
                print(f"未知算法类型: {algorithm_name}")
                return None

            return algo

        except Exception as e:
            print(f"加载算法实例失败 ({algorithm_name}): {e}")
            import traceback
            traceback.print_exc()
            return None

    def test_single_algorithm(self, algorithm_name: str, seed: int, episode_idx: int):
        try:
            print(f"\n{'=' * 60}")
            print(f"测试算法: {algorithm_name}, Seed: {seed}, Episode: {episode_idx}")
            print(f"{'=' * 60}")

            env = self.create_environment(algorithm_name)
            if env is None:
                return None, None, None

            env.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            algo = self._load_algorithm_instance(algorithm_name, env)
            if algo is None:
                env.close()
                return None, None, None

            test_data = {
                'algorithm': algorithm_name,
                'episode': episode_idx,
                'seed': seed,
                'disturbance_type': None,
                'disturbance_location': None,
                'disturbance_severity': None,
                'disturbance_duration': None,
                'step_data': [],
                'actions': [],
                'rewards': [],
                'multi_objective': [],
                'system_metrics': {}
            }

            obs, info = env.reset()

            if info and 'area_1' in info:
                initial_info = info['area_1']
                test_data['disturbance_type'] = initial_info.get('disturbance_type', 'unknown')
                test_data['disturbance_location'] = initial_info.get('disturbance_location', 'unknown')
                test_data['disturbance_severity'] = float(initial_info.get('disturbance_severity', 0.0))
                test_data['disturbance_duration'] = getattr(env, 'dist_duration', 1.0)

            print(f"扰动信息: {test_data['disturbance_type']} at {test_data['disturbance_location']}, "
                  f"严重程度: {test_data['disturbance_severity']:.3f}")

            episode_reward = {agent: 0.0 for agent in env.possible_agents}
            step = 0

            while step < env.max_steps:
                try:
                    if algorithm_name in ['MATD3', 'MASAC']:
                        actions = algo.select_action(obs, explore=False)
                    elif algorithm_name == 'WSE_MATD3':
                        actions = algo.select_action(obs, explore=False)
                    elif algorithm_name == 'NSGA2_MATD3':
                        actions = algo.select_action(obs, explore=False)
                    elif algorithm_name == 'MAPPO':
                        actions, _, _ = algo.select_action(obs, explore=False)
                    else:
                        actions = {agent: np.zeros(env.action_spaces[agent].shape[0])
                                   for agent in env.possible_agents}
                    # --- 动作选择结束 ---

                    next_obs, rewards, dones, truncs, infos = env.step(actions)

                    # 使用环境预设的动作时间点，确保与实际仿真时间一致
                    if hasattr(env, 'action_instants') and step < len(env.action_instants):
                        current_time = float(env.action_instants[step])
                    else:
                        # 后备方案：根据步长估算（通常不会触发）
                        current_time = float(step * env.tstep)

                    # 记录发电机级别的动作（每个发电单元独立列）
                    area_to_gens = env.area_to_gens
                    gen_actions = {}
                    for agent, action_vec in actions.items():
                        gen_indices = area_to_gens[agent]
                        for i, gen_idx in enumerate(gen_indices):
                            if i < len(action_vec):
                                gen_actions[f'gen_{gen_idx}_action'] = float(action_vec[i])

                    step_data = {
                        'step': step,
                        'time': current_time,
                        'actions': {agent: actions[agent].tolist() for agent in env.possible_agents},
                        'gen_actions': gen_actions,
                        'rewards': {agent: float(rewards[agent]) for agent in env.possible_agents},
                        'observations': {agent: next_obs[agent].tolist() for agent in env.possible_agents}
                    }
                    # 在 test_single_algorithm 中，记录 step_data 后添加
                    print(f"Step {step}: actions = {actions}")
                    print(f"gen_actions = {gen_actions}")
                    # 记录阻尼比和振荡频率（每个区域）- 严格检查有效性
                    if hasattr(env, 'prony_analyzers') and env.prony_analyzers is not None:
                        for agent in env.possible_agents:
                            analyzer = env.prony_analyzers.get(agent)
                            damping_val = float('nan')
                            freq_val = float('nan')
                            if analyzer and analyzer.last_analysis:
                                res = analyzer.last_analysis
                                if res.get('valid', False):
                                    damping_val = res.get('damping_ratio', float('nan'))
                                    freq_val = res.get('oscillation_freq', float('nan'))
                            step_data[f'damping_ratio_{agent}'] = damping_val
                            step_data[f'oscillation_freq_{agent}'] = freq_val
                    else:
                        for agent in env.possible_agents:
                            step_data[f'damping_ratio_{agent}'] = float('nan')
                            step_data[f'oscillation_freq_{agent}'] = float('nan')

                    test_data['step_data'].append(step_data)
                    test_data['actions'].append(step_data['actions'])
                    test_data['rewards'].append(step_data['rewards'])

                    for agent in env.possible_agents:
                        episode_reward[agent] += rewards[agent]

                    obs = next_obs
                    step += 1

                    if all(dones.values()) or all(truncs.values()):
                        break

                except Exception as e:
                    print(f"Step {step} 执行失败: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            # 提取仿真数据
            if hasattr(env, '_extract_simulation_data'):
                try:
                    env._extract_simulation_data()
                    if hasattr(env, 'system_data'):
                        test_data['system_data'] = env.system_data
                        test_data['system_metrics'] = self._calculate_system_metrics(
                            env.system_data, algorithm_name, episode_idx
                        )
                except Exception as e:
                    print(f"提取仿真数据失败: {e}")

            total_reward = sum(episode_reward.values())
            episode_result = {
                'total_reward': total_reward,
                'total_steps': step,
                'success': step == env.max_steps
            }

            print(f"算法 {algorithm_name} 测试完成, 总奖励: {total_reward:.2f}, 步数: {step}")

            env.close()
            del algo
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return test_data, episode_result, test_data.get('system_data')

        except Exception as e:
            print(f"测试算法 {algorithm_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def _calculate_system_metrics(self, system_data, algorithm_name, episode_idx):
        """计算系统指标 - 修复版本"""
        metrics = {
            'algorithm': algorithm_name,
            'episode': episode_idx,
            'generator_metrics': {},
            'vsg_metrics': {},
            'performance_metrics': {}
        }

        if not system_data:
            return metrics

        try:
            # 提取时间数据
            time = system_data.get('time', np.array([]))
            if len(time) == 0:
                return metrics

            # 计算频率偏差
            freq_deviation_data = {}

            # 发电机频率数据
            if 'generator_frequencies' in system_data:
                freq_data = system_data['generator_frequencies']
                n_generators = freq_data.shape[1] if freq_data.ndim > 1 else 1

                for i in range(min(9, n_generators)):
                    if freq_data.ndim > 1:
                        freq = freq_data[:, i]
                    else:
                        freq = freq_data

                    freq_dev = np.abs(freq - 1.0)
                    max_freq_dev = float(np.max(freq_dev))

                    # 计算恢复时间（偏差小于0.01pu）
                    threshold = 0.01
                    recovery_indices = np.where(freq_dev < threshold)[0]

                    if len(recovery_indices) > 0:
                        first_recovery_idx = recovery_indices[0]
                        recovery_time = float(time[first_recovery_idx])
                    else:
                        recovery_time = float(time[-1])

                    freq_deviation_data[f'gen_{i}'] = {
                        'max_freq_dev': max_freq_dev,
                        'recovery_time': recovery_time,
                        'mean_freq_dev': float(np.mean(freq_dev))
                    }

            # VSG频率数据
            if 'vsg_frequencies' in system_data:
                vsg_freq = system_data['vsg_frequencies']
                if vsg_freq.size > 0:
                    if vsg_freq.ndim > 1:
                        freq = vsg_freq[:, 0] if vsg_freq.shape[1] > 0 else vsg_freq.flatten()
                    else:
                        freq = vsg_freq

                    freq_dev = np.abs(freq - 1.0)
                    max_freq_dev = float(np.max(freq_dev))

                    threshold = 0.01
                    recovery_indices = np.where(freq_dev < threshold)[0]

                    if len(recovery_indices) > 0:
                        first_recovery_idx = recovery_indices[0]
                        recovery_time = float(time[first_recovery_idx])
                    else:
                        recovery_time = float(time[-1])

                    freq_deviation_data['vsg'] = {
                        'max_freq_dev': max_freq_dev,
                        'recovery_time': recovery_time,
                        'mean_freq_dev': float(np.mean(freq_dev))
                    }

            # 计算电压偏差
            volt_deviation_data = {}
            if 'bus_voltages' in system_data:
                voltage_data = system_data['bus_voltages']
                key_buses = [38, 31, 37, 32]  # 关键母线

                for i, bus in enumerate(key_buses):
                    if bus < voltage_data.shape[1]:
                        voltage = voltage_data[:, bus]
                        volt_dev = np.abs(voltage - 1.0)
                        max_volt_dev = float(np.max(volt_dev))

                        # 电压恢复阈值0.1pu
                        threshold = 0.1
                        recovery_indices = np.where(volt_dev < threshold)[0]

                        if len(recovery_indices) > 0:
                            first_recovery_idx = recovery_indices[0]
                            recovery_time = float(time[first_recovery_idx])
                        else:
                            recovery_time = float(time[-1])

                        volt_deviation_data[f'bus_{bus}'] = {
                            'max_volt_dev': max_volt_dev,
                            'recovery_time': recovery_time,
                            'mean_volt_dev': float(np.mean(volt_dev))
                        }

            # 计算总体性能指标
            overall_metrics = {}

            # 总体频率偏差
            if freq_deviation_data:
                all_max_freq_devs = [v['max_freq_dev'] for v in freq_deviation_data.values()]
                all_recovery_times = [v['recovery_time'] for v in freq_deviation_data.values()]
                all_mean_freq_devs = [v['mean_freq_dev'] for v in freq_deviation_data.values()]

                overall_metrics['overall_max_freq_dev'] = float(np.max(all_max_freq_devs))
                overall_metrics['overall_avg_freq_dev'] = float(np.mean(all_mean_freq_devs))
                overall_metrics['worst_recovery_time'] = float(np.max(all_recovery_times))
                overall_metrics['avg_recovery_time'] = float(np.mean(all_recovery_times))

            # 总体电压偏差
            if volt_deviation_data:
                all_max_volt_devs = [v['max_volt_dev'] for v in volt_deviation_data.values()]
                all_volt_recovery_times = [v['recovery_time'] for v in volt_deviation_data.values()]
                all_mean_volt_devs = [v['mean_volt_dev'] for v in volt_deviation_data.values()]

                overall_metrics['overall_max_volt_dev'] = float(np.max(all_max_volt_devs))
                overall_metrics['overall_avg_volt_dev'] = float(np.mean(all_mean_volt_devs))
                overall_metrics['worst_volt_recovery_time'] = float(np.max(all_volt_recovery_times))
                overall_metrics['avg_volt_recovery_time'] = float(np.mean(all_volt_recovery_times))

            # 保存指标
            metrics['freq_deviation_data'] = freq_deviation_data
            metrics['volt_deviation_data'] = volt_deviation_data
            metrics['performance_metrics'] = overall_metrics

        except Exception as e:
            print(f"计算系统指标失败: {e}")

        return metrics

    def run_all_tests(self, n_episodes: int = 5, seeds: List[int] = None):
        """运行所有测试 - 修复版本"""
        if seeds is None:
            seeds = [42 + i for i in range(n_episodes)]

        print(f"\n开始测试 {len(self.algorithms)} 种算法，共 {n_episodes} 个episode")
        print(f"测试种子: {seeds}")

        all_results = {}
        all_simulation_data = {}

        for episode_idx in range(n_episodes):
            seed = seeds[episode_idx]
            print(f"\n{'=' * 80}")
            print(f"Episode {episode_idx + 1}/{n_episodes}, Seed: {seed}")
            print(f"{'=' * 80}")

            episode_dir = os.path.join(self.result_dir, f"episode_{episode_idx}")
            os.makedirs(episode_dir, exist_ok=True)

            episode_results = {}
            episode_sim_data = {}

            for algorithm_name in self.algorithms:
                # 测试算法
                test_data, episode_result, sim_data = self.test_single_algorithm(
                    algorithm_name, seed, episode_idx
                )

                if test_data is not None:
                    # 保存测试数据
                    algo_episode_dir = os.path.join(episode_dir, algorithm_name)
                    os.makedirs(algo_episode_dir, exist_ok=True)

                    # 保存测试数据
                    self._save_test_data(test_data, algo_episode_dir)

                    # 保存仿真数据
                    if sim_data is not None:
                        self._save_simulation_data(sim_data, algo_episode_dir)
                        episode_sim_data[algorithm_name] = sim_data

                    episode_results[algorithm_name] = {
                        'test_data': test_data,
                        'episode_result': episode_result
                    }

                    print(f"✓ 算法 {algorithm_name} 测试完成")
                else:
                    print(f"✗ 算法 {algorithm_name} 测试失败")

            all_results[episode_idx] = episode_results
            all_simulation_data[episode_idx] = episode_sim_data

            # 保存episode汇总
            self._save_episode_summary(episode_results, episode_dir, episode_idx)

        # 保存全局汇总
        self._save_global_summary(all_results)

        self.test_results = all_results
        self.simulation_data = all_simulation_data

        print(f"\n所有测试完成！结果保存到: {self.result_dir}")
        return all_results, all_simulation_data

    def _save_test_data(self, test_data, save_dir):
        """保存测试数据"""
        try:
            # 保存基础信息
            base_info = {
                'algorithm': test_data['algorithm'],
                'episode': test_data['episode'],
                'seed': test_data['seed'],
                'disturbance_type': test_data['disturbance_type'],
                'disturbance_location': test_data['disturbance_location'],
                'disturbance_severity': float(test_data['disturbance_severity']),
                'disturbance_duration': float(test_data['disturbance_duration']) if test_data[
                    'disturbance_duration'] else 0.0
            }

            with open(os.path.join(save_dir, 'test_info.json'), 'w') as f:
                json.dump(base_info, f, indent=2, default=str)

            # 保存step数据
            if test_data.get('step_data'):
                step_records = []
                for step_data in test_data['step_data']:
                    record = {
                        'step': int(step_data['step']),
                        'time': float(step_data['time'])
                    }

                    # 保存发电机级别的动作数据
                    if 'gen_actions' in step_data:
                        for key, val in step_data['gen_actions'].items():
                            record[key] = float(val)

                    # 保存阻尼比和振荡频率（所有可能的区域键）
                    for key in step_data:
                        if key.startswith('damping_ratio_') or key.startswith('oscillation_freq_'):
                            record[key] = float(step_data[key])

                    # 保存奖励数据
                    if 'rewards' in step_data:
                        for agent, reward in step_data['rewards'].items():
                            record[f'{agent}_reward'] = float(reward)

                    step_records.append(record)

                if step_records:
                    step_df = pd.DataFrame(step_records)
                    step_df.to_csv(os.path.join(save_dir, 'step_data.csv'), index=False)

            # 保存系统指标
            if test_data.get('system_metrics'):
                # 将指标保存为JSON
                with open(os.path.join(save_dir, 'system_metrics.json'), 'w') as f:
                    json.dump(test_data['system_metrics'], f, indent=2, default=str)

                # 同时保存为CSV（扁平化）
                flattened_metrics = {}

                # 基础信息
                flattened_metrics['algorithm'] = test_data['system_metrics'].get('algorithm', '')
                flattened_metrics['episode'] = test_data['system_metrics'].get('episode', 0)

                # 频率偏差数据
                freq_data = test_data['system_metrics'].get('freq_deviation_data', {})
                for key, value in freq_data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            flattened_metrics[f'{key}_{sub_key}'] = float(sub_value)

                # 电压偏差数据
                volt_data = test_data['system_metrics'].get('volt_deviation_data', {})
                for key, value in volt_data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            flattened_metrics[f'{key}_{sub_key}'] = float(sub_value)

                # 性能指标
                perf_metrics = test_data['system_metrics'].get('performance_metrics', {})
                for key, value in perf_metrics.items():
                    flattened_metrics[key] = float(value)

                metrics_df = pd.DataFrame([flattened_metrics])
                metrics_df.to_csv(os.path.join(save_dir, 'system_metrics.csv'), index=False)

            print(f"测试数据保存到: {save_dir}")

        except Exception as e:
            print(f"保存测试数据失败: {e}")

    def _save_simulation_data(self, sim_data, save_dir):
        """保存仿真数据"""
        try:
            if not sim_data:
                return

            # 保存时间数据
            if 'time' in sim_data:
                time_df = pd.DataFrame({'time': sim_data['time']})
                time_df.to_csv(os.path.join(save_dir, 'simulation_time.csv'), index=False)

            # 保存发电机频率数据
            if 'generator_frequencies' in sim_data:
                freq_data = sim_data['generator_frequencies']
                n_generators = freq_data.shape[1] if freq_data.ndim > 1 else 1
                freq_columns = [f'gen_{i}_freq' for i in range(n_generators)]

                if freq_data.ndim > 1:
                    freq_df = pd.DataFrame(freq_data, columns=freq_columns[:n_generators])
                else:
                    freq_df = pd.DataFrame({f'gen_0_freq': freq_data})

                freq_df.to_csv(os.path.join(save_dir, 'generator_frequencies.csv'), index=False)

            # 保存发电机功角数据
            if 'generator_angles' in sim_data:
                angle_data = sim_data['generator_angles']
                n_generators = angle_data.shape[1] if angle_data.ndim > 1 else 1
                angle_columns = [f'gen_{i}_angle' for i in range(n_generators)]

                if angle_data.ndim > 1:
                    angle_df = pd.DataFrame(angle_data, columns=angle_columns[:n_generators])
                else:
                    angle_df = pd.DataFrame({f'gen_0_angle': angle_data})

                angle_df.to_csv(os.path.join(save_dir, 'generator_angles.csv'), index=False)

            # 保存发电机功率数据
            if 'generator_power' in sim_data:
                power_data = sim_data['generator_power']
                n_generators = power_data.shape[1] if power_data.ndim > 1 else 1
                power_columns = [f'gen_{i}_power' for i in range(n_generators)]

                if power_data.ndim > 1:
                    power_df = pd.DataFrame(power_data, columns=power_columns[:n_generators])
                else:
                    power_df = pd.DataFrame({f'gen_0_power': power_data})

                power_df.to_csv(os.path.join(save_dir, 'generator_power.csv'), index=False)

            # 保存VSG数据
            if 'vsg_frequencies' in sim_data:
                vsg_freq = sim_data['vsg_frequencies']
                vsg_freq_df = pd.DataFrame({'vsg_freq': vsg_freq.flatten()})
                vsg_freq_df.to_csv(os.path.join(save_dir, 'vsg_frequencies.csv'), index=False)

            if 'vsg_power' in sim_data:
                vsg_power = sim_data['vsg_power']
                vsg_power_df = pd.DataFrame({'vsg_power': vsg_power.flatten()})
                vsg_power_df.to_csv(os.path.join(save_dir, 'vsg_power.csv'), index=False)

            # 保存母线电压数据
            if 'bus_voltages' in sim_data:
                voltage_data = sim_data['bus_voltages']
                n_buses = voltage_data.shape[1] if voltage_data.ndim > 1 else 1
                voltage_columns = [f'bus_{i}_voltage' for i in range(n_buses)]

                if voltage_data.ndim > 1:
                    voltage_df = pd.DataFrame(voltage_data, columns=voltage_columns[:n_buses])
                else:
                    voltage_df = pd.DataFrame({'bus_0_voltage': voltage_data})

                voltage_df.to_csv(os.path.join(save_dir, 'bus_voltages.csv'), index=False)

            # 保存COI数据
            if 'coi_frequencies' in sim_data:
                coi_freq = sim_data['coi_frequencies']
                n_coi = coi_freq.shape[1] if coi_freq.ndim > 1 else 1
                coi_columns = [f'coi_{i}_freq' for i in range(n_coi)]

                if coi_freq.ndim > 1:
                    coi_freq_df = pd.DataFrame(coi_freq, columns=coi_columns[:n_coi])
                else:
                    coi_freq_df = pd.DataFrame({'coi_0_freq': coi_freq})

                coi_freq_df.to_csv(os.path.join(save_dir, 'coi_frequencies.csv'), index=False)

            if 'coi_angles' in sim_data:
                coi_angle = sim_data['coi_angles']
                n_coi = coi_angle.shape[1] if coi_angle.ndim > 1 else 1
                coi_columns = [f'coi_{i}_angle' for i in range(n_coi)]

                if coi_angle.ndim > 1:
                    coi_angle_df = pd.DataFrame(coi_angle, columns=coi_columns[:n_coi])
                else:
                    coi_angle_df = pd.DataFrame({'coi_0_angle': coi_angle})

                coi_angle_df.to_csv(os.path.join(save_dir, 'coi_angles.csv'), index=False)

            # 保存ACE信号
            if 'ace_signals' in sim_data:
                ace_data = sim_data['ace_signals']
                n_ace = ace_data.shape[1] if ace_data.ndim > 1 else 1
                ace_columns = [f'ace_{i}' for i in range(n_ace)]

                if ace_data.ndim > 1:
                    ace_df = pd.DataFrame(ace_data, columns=ace_columns[:n_ace])
                else:
                    ace_df = pd.DataFrame({'ace_0': ace_data})

                ace_df.to_csv(os.path.join(save_dir, 'ace_signals.csv'), index=False)

            # 保存完整的仿真数据
            with open(os.path.join(save_dir, 'simulation_data.pkl'), 'wb') as f:
                pickle.dump(sim_data, f)

            print(f"仿真数据保存到: {save_dir}")

        except Exception as e:
            print(f"保存仿真数据失败: {e}")

    def _save_episode_summary(self, episode_results, save_dir, episode_idx):
        """保存episode汇总"""
        try:
            summary = {
                'episode': episode_idx,
                'algorithms_tested': list(episode_results.keys()),
                'results': {}
            }

            for algo_name, result in episode_results.items():
                if 'episode_result' in result:
                    summary['results'][algo_name] = {
                        'total_reward': result['episode_result'].get('total_reward', 0.0),
                        'total_steps': result['episode_result'].get('total_steps', 0),
                        'success': result['episode_result'].get('success', False)
                    }

            with open(os.path.join(save_dir, 'episode_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)

            # 保存为CSV
            summary_list = []
            for algo_name, result in summary['results'].items():
                summary_list.append({
                    'algorithm': algo_name,
                    'episode': episode_idx,
                    'total_reward': result['total_reward'],
                    'total_steps': result['total_steps'],
                    'success': result['success']
                })

            if summary_list:
                summary_df = pd.DataFrame(summary_list)
                summary_df.to_csv(os.path.join(save_dir, 'episode_summary.csv'), index=False)

        except Exception as e:
            print(f"保存episode汇总失败: {e}")

    def _save_global_summary(self, all_results):
        """保存全局汇总"""
        try:
            summary_path = os.path.join(self.result_dir, "global_summary")
            os.makedirs(summary_path, exist_ok=True)

            # 收集所有性能数据
            all_performance_data = []

            for episode_idx, episode_results in all_results.items():
                for algo_name, result in episode_results.items():
                    if 'test_data' in result and 'system_metrics' in result['test_data']:
                        sys_metrics = result['test_data']['system_metrics']

                        # 创建汇总记录
                        summary_record = {
                            'algorithm': algo_name,
                            'episode': episode_idx,
                            'total_reward': result.get('episode_result', {}).get('total_reward', 0.0),
                            'total_steps': result.get('episode_result', {}).get('total_steps', 0),
                            'success': result.get('episode_result', {}).get('success', False)
                        }

                        # 添加性能指标
                        perf_metrics = sys_metrics.get('performance_metrics', {})
                        for key, value in perf_metrics.items():
                            summary_record[key] = float(value)

                        all_performance_data.append(summary_record)

            # 保存汇总数据
            if all_performance_data:
                summary_df = pd.DataFrame(all_performance_data)
                summary_df.to_csv(os.path.join(summary_path, 'performance_summary.csv'), index=False)

                # 计算统计摘要
                algorithms = summary_df['algorithm'].unique()
                stats_summary = {}

                for algo in algorithms:
                    algo_data = summary_df[summary_df['algorithm'] == algo]

                    stats_summary[algo] = {
                        'avg_total_reward': float(algo_data['total_reward'].mean()),
                        'std_total_reward': float(algo_data['total_reward'].std()),
                        'success_rate': float(algo_data['success'].mean()) * 100
                    }

                # 保存统计摘要
                with open(os.path.join(summary_path, 'performance_stats.json'), 'w') as f:
                    json.dump(stats_summary, f, indent=2)

                # 保存为CSV
                stats_list = []
                for algo, stats in stats_summary.items():
                    stats['algorithm'] = algo
                    stats_list.append(stats)

                stats_df = pd.DataFrame(stats_list)
                stats_df.to_csv(os.path.join(summary_path, 'performance_stats.csv'), index=False)

            # 保存全局信息
            global_info = {
                'total_episodes': len(all_results),
                'algorithms': self.algorithms,
                'test_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'env_config': self.env_config
            }

            with open(os.path.join(summary_path, 'global_info.json'), 'w') as f:
                json.dump(global_info, f, indent=2, default=str)

            print(f"全局汇总保存到: {summary_path}")

        except Exception as e:
            print(f"保存全局汇总失败: {e}")

class EnhancedMARL_Visualizer:
    """增强的MARL可视化器 - 修复版本，绘图统一保存在 result_dir/plot 下"""

    def __init__(self, result_dir: str):
        """
        参数:
            result_dir: 测试结果主目录（与 Tester 的 result_dir 相同）
        """
        self.result_dir = result_dir
        self.plots_dir = os.path.join(result_dir, "plot")
        os.makedirs(self.plots_dir, exist_ok=True)
        # 设置全局字体为 Times New Roman，字号 14
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 16
        # 算法颜色和标记
        self.algorithm_colors = {
            'MAPPO': '#bcbd22',
            'MATD3': '#ff7f0e',
            'MASAC': '#d62728',
            'NSGA2_MATD3': '#9467bd',
            'WSE_MATD3': '#4d4d4d',
        }

        self.algorithm_linestyles = {
            'MATD3': ':',
            'MAPPO': (0, (5, 1)),
            'MASAC': (0, (3, 1, 1, 1, 1, 1)),
            'NSGA2_MATD3': (0, (3, 1, 1, 1)),
            'WSE_MATD3': '-',
        }

        self.algorithm_markers = {
            'MATD3': 's',
            'MAPPO': '^',
            'MASAC': 'D',
            'NSGA2_MATD3': 'v',
            'WSE_MATD3': 'p',
        }

        # 区域颜色
        self.area_colors = {
            'area_1': '#1f77b4',  # 蓝色
            'area_2': '#ff7f0e',  # 橙色
            'area_3': '#2ca02c',  # 绿色
            'area_4': '#d62728'  # 红色
        }

    def _save_figure(self, fig, filename):
        """保存图为 PNG 和 PDF，dpi=1500"""
        png_path = os.path.join(self.plots_dir, filename + '.png')
        pdf_path = os.path.join(self.plots_dir, filename + '.pdf')
        fig.savefig(png_path, dpi=1500, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ 图表已保存: {png_path} 和 {pdf_path}")

    def load_all_episode_data(self):
        """加载所有episode的数据"""
        print("加载所有episode数据...")

        all_episode_data = {}

        # 查找所有episode目录
        episode_dirs = [d for d in os.listdir(self.result_dir)
                        if d.startswith('episode_') and os.path.isdir(os.path.join(self.result_dir, d))]

        for episode_dir in episode_dirs:
            try:
                episode_idx = int(episode_dir.split('_')[1])
                episode_path = os.path.join(self.result_dir, episode_dir)

                # 加载episode汇总
                summary_file = os.path.join(episode_path, 'episode_summary.json')
                if os.path.exists(summary_file):
                    with open(summary_file, 'r') as f:
                        episode_data = json.load(f)
                else:
                    episode_data = {'episode': episode_idx, 'algorithms': {}}

                # 加载每个算法的详细数据
                algorithm_dirs = [d for d in os.listdir(episode_path)
                                  if os.path.isdir(os.path.join(episode_path, d))]

                for algorithm_dir in algorithm_dirs:
                    algorithm_path = os.path.join(episode_path, algorithm_dir)

                    # 检查是否存在仿真数据
                    sim_data_file = os.path.join(algorithm_path, 'simulation_data.pkl')
                    if os.path.exists(sim_data_file):
                        try:
                            with open(sim_data_file, 'rb') as f:
                                sim_data = pickle.load(f)

                            # 加载测试信息
                            test_info_file = os.path.join(algorithm_path, 'test_info.json')
                            test_info = {}
                            if os.path.exists(test_info_file):
                                with open(test_info_file, 'r') as f:
                                    test_info = json.load(f)

                            # 加载step数据
                            step_data_file = os.path.join(algorithm_path, 'step_data.csv')
                            step_data = None
                            if os.path.exists(step_data_file):
                                step_data = pd.read_csv(step_data_file)

                            # 加载系统指标
                            metrics_file = os.path.join(algorithm_path, 'system_metrics.json')
                            metrics = {}
                            if os.path.exists(metrics_file):
                                with open(metrics_file, 'r') as f:
                                    metrics = json.load(f)

                            # 存储数据
                            if 'algorithms' not in episode_data:
                                episode_data['algorithms'] = {}

                            episode_data['algorithms'][algorithm_dir] = {
                                'simulation_data': sim_data,
                                'test_info': test_info,
                                'step_data': step_data,
                                'metrics': metrics
                            }

                        except Exception as e:
                            print(f"加载{algorithm_dir}数据失败: {e}")

                all_episode_data[episode_idx] = episode_data
                print(f"✓ Episode {episode_idx} 数据加载完成")

            except Exception as e:
                print(f"加载{episode_dir}失败: {e}")

        print(f"共加载 {len(all_episode_data)} 个episode的数据")
        return all_episode_data

    def plot_type1_statistics(self, all_episode_data):
        """绘制类型1：频率最大偏差（同步机）与电压最大偏差（所有母线）的小提琴图"""
        print("\n正在生成类型1统计图（小提琴图）...")

        # 收集数据
        all_metrics = []

        for episode_idx, episode_data in all_episode_data.items():
            for algo_name, algo_data in episode_data.get('algorithms', {}).items():
                metrics = algo_data.get('metrics', {})
                if not metrics:
                    continue

                perf_metrics = metrics.get('performance_metrics', {}).copy()

                # 剔除VSG频率偏差，只保留同步机
                freq_data = metrics.get('freq_deviation_data', {})
                if freq_data:
                    sync_max_freq_devs = []
                    for key, val in freq_data.items():
                        if key.startswith('gen_') and key != 'vsg':
                            sync_max_freq_devs.append(val.get('max_freq_dev', 0.0))
                    if sync_max_freq_devs:
                        perf_metrics['overall_max_freq_dev'] = max(sync_max_freq_devs)

                # 电压偏差
                record = {
                    'algorithm': algo_name,
                    'episode': episode_idx,
                    'max_freq_dev': perf_metrics.get('overall_max_freq_dev', 0.0),
                    'max_volt_dev': perf_metrics.get('overall_max_volt_dev', 0.0)
                }
                all_metrics.append(record)

        if not all_metrics:
            print("警告: 没有找到足够的数据生成类型1图表")
            return

        df = pd.DataFrame(all_metrics)

        # 定义算法顺序（原始名称）
        algorithm_order = ['MAPPO', 'MASAC', 'MATD3', 'NSGA2_MATD3', 'WSE_MATD3']
        existing_algorithms = [algo for algo in algorithm_order if algo in df['algorithm'].unique()]
        df = df[df['algorithm'].isin(existing_algorithms)]
        df['algorithm'] = pd.Categorical(df['algorithm'], categories=existing_algorithms, ordered=True)

        # 创建显示名称映射（下划线 -> 连字符）
        display_name_map = {
            'NSGA2_MATD3': 'NSGA2-MATD3',
            'WSE_MATD3': 'WSE-MATD3'
        }
        # 其他算法名称保持不变
        for algo in existing_algorithms:
            if algo not in display_name_map:
                display_name_map[algo] = algo

        # 添加显示名称列
        df['algorithm_display'] = df['algorithm'].map(display_name_map)

        # 将宽数据转为长格式，使用显示名称列
        df_melt = df.melt(
            id_vars=['algorithm_display', 'episode'],
            value_vars=['max_freq_dev', 'max_volt_dev'],
            var_name='metric',
            value_name='deviation'
        )
        metric_labels = {'max_freq_dev': 'Max_Freq_Deviation',
                         'max_volt_dev': 'Max_Volt_Deviation'}
        df_melt['metric'] = df_melt['metric'].map(metric_labels)

        # 创建图形，x轴使用显示名称列
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(
            data=df_melt,
            x='algorithm_display',
            y='deviation',
            hue='metric',
            palette={'Max_Freq_Deviation': 'lightblue',
                     'Max_Volt_Deviation': 'lightgreen'},
            split=False,
            dodge=True,
            width=0.5,
            ax=ax
        )

        # 设置图形属性
        ax.set_xlabel('Algorithm', fontsize=16)
        ax.set_ylabel('Deviation (p.u.)', fontsize=16)
        ax.set_ylim(-0.01, 0.1)
        ax.tick_params(axis='x', labelsize=12, rotation=0)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(title='Metric', fontsize=16, title_fontsize=16, frameon=False, loc='center left', bbox_to_anchor=(0.06, 0.35))

        plt.tight_layout()

        # 保存图表
        self._save_figure(fig, "type1_statistics_enhanced")
        # 保存原始数据（不包含显示名称列，保持原样）
        data_path = os.path.join(self.plots_dir, "type1_statistics_data.csv")
        df.drop('algorithm_display', axis=1).to_csv(data_path, index=False)

        print(f"✓ 类型1统计数据已保存到: {data_path}")

    def plot_oscillation_control_comparison(self, episode_indices):
        """
        绘制多种算法在指定 episode 的振荡控制对比图（仅4行：阻尼比、频率、电压、功率）
        参数:
            episode_indices: int 或 list of int，要绘制的 episode 索引
        """
        if isinstance(episode_indices, int):
            episode_indices = [episode_indices]

        algos = ['MAPPO', 'MASAC', 'MATD3', 'NSGA2_MATD3', 'WSE_MATD3']
        colors = {
            'MAPPO': '#bcbd22',
            'MATD3': '#ff7f0e',
            'MASAC': '#d62728',
            'NSGA2_MATD3': '#9467bd',
            'WSE_MATD3': '#4d4d4d',
        }
        linestyles = {
            'MATD3': ':',
            'MAPPO': (0, (5, 1)),
            'MASAC': (0, (3, 1, 1, 1, 1, 1)),
            'NSGA2_MATD3': (0, (3, 1, 1, 1)),
            'WSE_MATD3': '-',
        }
        labels = {
            'WSE_MATD3': 'WSE-MATD3',
            'NSGA2_MATD3': 'NSGA2-MATD3',
            'MATD3': 'MATD3',
            'MASAC': 'MASAC',
            'MAPPO': 'MAPPO'
        }
        area_config = {
            'area_1': {'gen_indices': [0, 6, 8], 'bus_index': 38, 'gen_for_plot': 0},
            'area_2': {'gen_indices': [1, 9], 'bus_index': 31, 'gen_for_plot': 1},
            'area_3': {'gen_indices': [4, 7], 'bus_index': 37, 'gen_for_plot': 4},
            'area_4': {'gen_indices': [2, 3, 5], 'bus_index': 32, 'gen_for_plot': 2}
        }

        for episode_idx in episode_indices:
            print(f"\n生成振荡控制对比图 (Episode {episode_idx})...")
            episode_path = os.path.join(self.result_dir, f"episode_{episode_idx}")
            if not os.path.exists(episode_path):
                print(f"错误：Episode {episode_idx} 数据不存在，跳过")
                continue

            # 加载算法数据
            algo_data = {}
            for algo in algos:
                algo_dir = os.path.join(episode_path, algo)
                if not os.path.exists(algo_dir):
                    continue
                sim_file = os.path.join(algo_dir, 'simulation_data.pkl')
                if not os.path.exists(sim_file):
                    continue
                with open(sim_file, 'rb') as f:
                    sim_data = pickle.load(f)
                step_file = os.path.join(algo_dir, 'step_data.csv')
                step_df = pd.read_csv(step_file) if os.path.exists(step_file) else None
                info_file = os.path.join(algo_dir, 'test_info.json')
                test_info = json.load(open(info_file)) if os.path.exists(info_file) else {}
                algo_data[algo] = {'sim': sim_data, 'step': step_df, 'info': test_info}

            if len(algo_data) < 2:
                print("错误：至少需要两种算法的数据才能绘图，跳过此episode")
                continue

            # 收集所有算法的 time 数组，找出最小长度
            time_lengths = []
            for data in algo_data.values():
                sim = data['sim']
                t = sim.get('time')
                if t is not None:
                    time_lengths.append(len(t))
            if not time_lengths:
                print("错误：无法获取有效的时间轴")
                continue
            common_len = min(time_lengths)

            # 统一所有算法的 time 轴为公共长度，并截断相关数据
            for algo, data in algo_data.items():
                sim = data['sim']
                if 'time' in sim:
                    sim['time'] = sim['time'][:common_len]
                for key in ['generator_frequencies', 'generator_angles', 'generator_power',
                            'vsg_frequencies', 'vsg_power', 'bus_voltages',
                            'coi_frequencies', 'coi_angles', 'ace_signals', 'bus_frequencies']:
                    if key in sim and sim[key] is not None:
                        if sim[key].ndim == 1:
                            sim[key] = sim[key][:common_len]
                        elif sim[key].ndim == 2:
                            sim[key] = sim[key][:common_len, :]
                step_df = data['step']
                if step_df is not None and not step_df.empty:
                    max_time = sim['time'][-1]
                    step_df = step_df[step_df['time'] <= max_time].reset_index(drop=True)
                    data['step'] = step_df

            first_algo = list(algo_data.keys())[0]
            time = algo_data[first_algo]['sim'].get('time')
            if time is None:
                print("错误：仿真数据中缺少时间轴")
                continue
            time = np.array(time)

            # 创建画布：4行4列（移除动作行）
            fig, axes = plt.subplots(4, 4, figsize=(20, 14), sharex=True)
            fig.subplots_adjust(hspace=0.1, wspace=0.25)

            global_handles, global_labels = [], []

            # 绘制所有子图
            for col, (area, config) in enumerate(area_config.items()):
                gen_idx = config['gen_for_plot']
                bus_idx = config['bus_index']
                damping_col = f'damping_ratio_{area}'

                axes[0, col].set_title(area.replace('_', ' ').title(), fontsize=18)

                for row in range(4):  # 只有0,1,2,3四行
                    ax = axes[row, col]

                    for algo, data in algo_data.items():
                        sim = data['sim']
                        step_df = data['step']
                        color = colors.get(algo, '#000000')
                        ls = linestyles.get(algo, '-')
                        label = labels.get(algo, algo)

                        if row == 0:  # 阻尼比
                            if step_df is not None and damping_col in step_df.columns:
                                step_times = step_df['time'].values
                                damping_vals = step_df[damping_col].values
                                window = 3
                                if len(damping_vals) >= window:
                                    damping_smoothed = pd.Series(damping_vals).rolling(window, min_periods=1,
                                                                                       center=True).mean().values
                                else:
                                    damping_smoothed = damping_vals
                                ax.plot(step_times, damping_smoothed, color=color, linewidth=2.5,
                                        label=label if col == 0 and row == 0 else "")
                            else:
                                ax.text(0.5, 0.5, 'No damping data', transform=ax.transAxes,
                                        ha='center', va='center', fontsize=18, color='gray', alpha=0.5)
                        elif row == 1:  # 频率
                            if gen_idx < 9:
                                if 'generator_frequencies' in sim:
                                    freq_data = sim['generator_frequencies']
                                    if freq_data.ndim > 1 and gen_idx < freq_data.shape[1]:
                                        ax.plot(time, freq_data[:, gen_idx], color=color, linewidth=2.5)
                            else:
                                if 'vsg_frequencies' in sim:
                                    vsg_freq = sim['vsg_frequencies'].flatten()
                                    if len(vsg_freq) == len(time):
                                        ax.plot(time, vsg_freq, color=color, linewidth=2.5)
                        elif row == 2:  # 电压
                            if 'bus_voltages' in sim:
                                volt_data = sim['bus_voltages']
                                if volt_data.ndim > 1 and bus_idx < volt_data.shape[1]:
                                    ax.plot(time, volt_data[:, bus_idx], color=color, linewidth=2.5)
                        elif row == 3:  # 功率（原第4行）
                            if gen_idx < 9:
                                if 'generator_power' in sim:
                                    power_data = sim['generator_power']
                                    if power_data.ndim > 1 and gen_idx < power_data.shape[1]:
                                        ax.plot(time, power_data[:, gen_idx], color=color, linewidth=2.5)
                            else:
                                if 'vsg_power' in sim:
                                    vsg_power = sim['vsg_power'].flatten()
                                    if len(vsg_power) == len(time):
                                        ax.plot(time, vsg_power, color=color, linewidth=2.5)

                    # 行标签（仅左侧列）
                    if col == 0:
                        ylabels = ['Damping Ratio (smoothed)', 'Freq (pu)', 'Voltage (pu)', 'Power (pu)']
                        ax.set_ylabel(ylabels[row], fontsize=18)
                        ax.yaxis.set_label_coords(-0.2, 0.5)

                    if row == 3:  # 最后一行（原功率行）设置x轴标签
                        ax.set_xlabel('Time (s)', fontsize=18)

                    ax.grid(True, alpha=0.3)

                    # 收集图例句柄（只一次）
                    if row == 0 and col == 0:
                        for algo in algo_data.keys():
                            line = plt.Line2D([0], [0], color=colors[algo], linestyle='-', linewidth=2.5)
                            global_handles.append(line)
                            global_labels.append(labels.get(algo, algo))

            # 设置所有子图的 x 轴范围为 0-15
            for ax in axes.flat:
                ax.set_xlim(0, 15)
                ax.tick_params(labelsize=18)
                ax.xaxis.label.set_size(18)
                ax.yaxis.label.set_size(18)

            # 添加全局图例
            fig.legend(global_handles, global_labels,
                       loc='upper right',
                       ncol=5,
                       fontsize=18,
                       framealpha=0.9,
                       bbox_to_anchor=(0.99, 0.97),
                       frameon=False)

            # 添加扰动阴影
            info = list(algo_data.values())[0]['info']
            dist_start = 2.0
            dist_duration = info.get('disturbance_duration', 1.0)
            if dist_duration:
                dist_end = dist_start + dist_duration
                for ax in axes.flatten():
                    ax.axvspan(dist_start, dist_end, alpha=0.2, color='lightgray')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            self._save_figure(fig, f'oscillation_control_comparison_ep{episode_idx}')
            plt.close()

        print(f"所有指定episode的振荡控制对比图已保存到: {self.plots_dir}")

    def plot_type2_generator_timeseries(self, all_episode_data, episode_idx=0):
        """
        绘制类型2：10台发电单元在多种算法下的时域仿真结果图
        增强版：
          - 第1行：所属区域阻尼比的均值曲线 ± 标准差（基于所有episode）
          - 第2-5行：频率、电压、有功、动作（基于指定episode）
        """
        print(f"\n正在生成增强类型2图表（参考Episode {episode_idx}）...")

        if not all_episode_data or episode_idx not in all_episode_data:
            print(f"Episode {episode_idx} 数据不存在")
            return

        # 区域到发电机映射
        area_to_gens = {
            'area_1': [0, 6, 8],
            'area_2': [1, 9],
            'area_3': [4, 7],
            'area_4': [2, 3, 5]
        }
        gen_to_area = {}
        for area, gens in area_to_gens.items():
            for g in gens:
                gen_to_area[g] = area

        # 收集所有算法名称（取所有episode中出现的算法）
        algorithms = set()
        for ep_data in all_episode_data.values():
            algorithms.update(ep_data.get('algorithms', {}).keys())
        algorithms = sorted(algorithms)

        # 定义算法样式（键名保持原样带下划线）
        local_colors = {
            'MAPPO': '#bcbd22',
            'MATD3': '#ff7f0e',
            'MASAC': '#d62728',
            'NSGA2_MATD3': '#9467bd',
            'WSE_MATD3': '#4d4d4d',
        }
        local_linestyles = {
            'MATD3': ':',
            'MAPPO': (0, (5, 1)),
            'MASAC': (0, (3, 1, 1, 1, 1, 1)),
            'NSGA2_MATD3': (0, (3, 1, 1, 1)),
            'WSE_MATD3': '-',
        }
        colors = local_colors
        linestyles = local_linestyles

        # 对于每个发电机
        for gen_idx in range(10):
            area = gen_to_area.get(gen_idx, None)
            if area is None:
                continue
            damping_col = f'damping_ratio_{area}'

            # 创建画布
            fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
            ax_damping = axes[0]
            ax_freq = axes[1]
            ax_volt = axes[2]
            ax_power = axes[3]
            # ax_action = axes[4]

            # 收集全局图例句柄
            global_handles = []
            global_labels = []

            # 获取参考episode的扰动信息
            ref_ep_data = all_episode_data[episode_idx]
            disturbance_info = None
            for algo in algorithms:
                if algo in ref_ep_data.get('algorithms', {}):
                    test_info = ref_ep_data['algorithms'][algo].get('test_info', {})
                    if test_info:
                        disturbance_info = test_info
                        break

            # ===== 子图1：阻尼比（基于所有episode，绘制均值±标准差） =====
            damping_dict = {algo: [] for algo in algorithms}
            step_times = None
            time_len = None

            for ep, ep_data in all_episode_data.items():
                for algo in algorithms:
                    algo_data = ep_data.get('algorithms', {}).get(algo)
                    if not algo_data:
                        continue
                    step_df = algo_data.get('step_data')
                    if step_df is None or step_df.empty:
                        continue
                    if damping_col not in step_df.columns:
                        continue
                    # 获取时间轴和阻尼比值
                    times = step_df['time'].values
                    vals = step_df[damping_col].values
                    # 记录第一个有效的时间长度
                    if time_len is None:
                        time_len = len(times)
                    # 长度对齐（取所有序列中的最小长度）
                    if len(times) != time_len:
                        # 如果长度不一致，截断到较小长度
                        min_len = min(time_len, len(times))
                        times = times[:min_len]
                        vals = vals[:min_len]
                        time_len = min_len
                    if step_times is None:
                        step_times = times
                    damping_dict[algo].append(vals)

            # 确保 step_times 存在
            if step_times is not None and any(len(damping_dict[algo]) > 0 for algo in algorithms):
                for algo in algorithms:
                    if len(damping_dict[algo]) == 0:
                        continue
                    # 将所有episode的序列垂直堆叠
                    aligned_seqs = []
                    for seq in damping_dict[algo]:
                        if len(seq) >= len(step_times):
                            aligned_seqs.append(seq[:len(step_times)])
                        else:
                            padded = np.full(len(step_times), np.nan)
                            padded[:len(seq)] = seq
                            aligned_seqs.append(padded)
                    damping_array = np.vstack(aligned_seqs)
                    damping_mean = np.nanmean(damping_array, axis=0)
                    damping_std = np.nanstd(damping_array, axis=0)
                    color = colors.get(algo, '#000000')
                    ls = linestyles.get(algo, '-')
                    line, = ax_damping.plot(step_times, damping_mean, color=color, linewidth=2.5,
                                            label=algo if gen_idx == 0 else "")
                    ax_damping.fill_between(step_times,
                                            damping_mean - damping_std,
                                            damping_mean + damping_std,
                                            color=color, alpha=0.2)

                    # 修改点：将算法名称中的下划线替换为连字符，用于图例显示
                    display_name = algo.replace('_', '-')
                    global_handles.append(line)
                    global_labels.append(display_name)
            else:
                ax_damping.text(0.5, 0.5, 'No damping data', transform=ax_damping.transAxes,
                                ha='center', va='center', fontsize=18, color='gray')

            ax_damping.set_ylabel('Damping Ratio\n(mean ± std)', fontsize=20)

            # ===== 子图2-5：基于参考episode绘制频率、电压、功率、动作 =====
            ref_data = ref_ep_data.get('algorithms', {})
            if not ref_data:
                print(f"警告：参考episode {episode_idx} 无算法数据")
                plt.close(fig)
                continue

            for algo in algorithms:
                algo_data = ref_data.get(algo)
                if not algo_data:
                    continue
                sim_data = algo_data.get('simulation_data')
                step_df = algo_data.get('step_data')
                if sim_data is None or 'time' not in sim_data:
                    continue
                time = sim_data['time']
                color = colors.get(algo, '#000000')
                ls = linestyles.get(algo, '-')

                # 频率
                if gen_idx < 9:
                    if 'generator_frequencies' in sim_data:
                        freq_data = sim_data['generator_frequencies']
                        if freq_data.ndim == 2 and freq_data.shape[0] == len(time) and gen_idx < freq_data.shape[1]:
                            ax_freq.plot(time, freq_data[:, gen_idx], color=color, linewidth=2.5)
                else:
                    if 'vsg_frequencies' in sim_data:
                        vsg_freq = sim_data['vsg_frequencies'].flatten()
                        if len(vsg_freq) == len(time):
                            ax_freq.plot(time, vsg_freq, color=color, linewidth=2.5)

                # 电压
                if 'bus_voltages' in sim_data:
                    key_bus_map = {'area_1': 38, 'area_2': 31, 'area_3': 37, 'area_4': 32}
                    bus_idx = key_bus_map.get(area, 0)
                    volt_data = sim_data['bus_voltages']
                    if volt_data.ndim == 2 and volt_data.shape[0] == len(time) and bus_idx < volt_data.shape[1]:
                        ax_volt.plot(time, volt_data[:, bus_idx], color=color, linewidth=2.5)

                # 有功功率
                if gen_idx < 9:
                    if 'generator_power' in sim_data:
                        power_data = sim_data['generator_power']
                        if power_data.ndim == 2 and power_data.shape[0] == len(time) and gen_idx < power_data.shape[1]:
                            ax_power.plot(time, power_data[:, gen_idx], color=color, linewidth=2.5)
                else:
                    if 'vsg_power' in sim_data:
                        vsg_power = sim_data['vsg_power'].flatten()
                        if len(vsg_power) == len(time):
                            ax_power.plot(time, vsg_power, color=color, linewidth=2.5)

            for ax in axes:
                ax.set_xlim(0, 15)

            # 设置 y 轴标签和字号
            ax_freq.set_ylabel('Rotor Speed (pu)', fontsize=20)
            ax_volt.set_ylabel('Voltage (pu)', fontsize=20)
            ax_power.set_ylabel('Power (pu)', fontsize=20)
            ax_power.set_xlabel('Time (s)', fontsize=20)

            # 左对齐 y 轴标签
            for ax in axes:
                ax.yaxis.set_label_coords(-0.05, 0.5)

            # 设置网格和动作轴范围
            for ax in axes:
                ax.grid(True, alpha=0.6)
            # ax_action.set_ylim([-5.0, 5.0])

            # 增大坐标轴刻度字号
            for ax in axes:
                ax.tick_params(labelsize=18)

            # 全局图例（每个发电机图片都添加）
            if global_handles:
                # 使用已转换的 display_name 的图例
                fig.legend(global_handles, global_labels,
                           loc='upper right',
                           ncol=min(5, len(global_handles)),
                           fontsize=18,
                           bbox_to_anchor=(0.99, 0.98),
                           frameon=False)

            # 添加扰动阴影
            if disturbance_info and 'disturbance_duration' in disturbance_info:
                dist_start = 2.0
                dist_duration = float(disturbance_info.get('disturbance_duration', 1.0))
                dist_end = dist_start + dist_duration
                for ax in axes:
                    ax.axvspan(dist_start, dist_end, alpha=0.2, color='lightgray')

                dist_text = (f"Type: {disturbance_info.get('disturbance_type', 'Unknown')}\n"
                             f"Location: {disturbance_info.get('disturbance_location', 'Unknown')}\n"
                             f"Severity: {disturbance_info.get('disturbance_severity', 0.0):.3f}\n"
                             f"Duration: {dist_duration:.2f}s")
                ax_damping.text( 0.15, ax_damping.get_ylim()[1] * 0.42, dist_text,
                                fontsize=13,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray', alpha=0.8))

            gen_type = "VSG" if gen_idx == 9 else f"Gen_{gen_idx + 1}"
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            # 保存图片（使用类内 _save_figure 方法）
            self._save_figure(fig, f"type2_generator_{gen_type}_ep{episode_idx}_enhanced")
            plt.close()

            print(f"✓ 发电机 {gen_type} 增强图表已保存")

        print(f"✓ 类型2增强图表生成完成")

    def plot_damping_ratio_distribution(self, all_episode_data, algorithms=None):

        import seaborn as sns

        # 收集所有 episode 中每个区域每个时刻的阻尼比数据
        data = []
        for ep, ep_data in all_episode_data.items():
            for algo, algo_data in ep_data.get('algorithms', {}).items():
                step_data = algo_data.get('step_data')
                if step_data is None:
                    continue
                damping_cols = [col for col in step_data.columns if col.startswith('damping_ratio_')]
                for col in damping_cols:
                    area = col.replace('damping_ratio_', '')
                    damping_vals = step_data[col].dropna().values
                    for val in damping_vals:
                        data.append({
                            'algorithm': algo,
                            'episode': ep,
                            'area': area,
                            'damping_ratio': val
                        })

        if not data:
            print("无阻尼比数据")
            return

        df = pd.DataFrame(data)
        if algorithms is not None:
            df = df[df['algorithm'].isin(algorithms)]

        # 指定算法显示顺序（使用原始带下划线的名称，内部保持原样）
        algorithm_order = ['MAPPO', 'MASAC', 'MATD3', 'NSGA2_MATD3','WSE_MATD3']
        existing_algos = [a for a in algorithm_order if a in df['algorithm'].unique()]
        df['algorithm'] = pd.Categorical(df['algorithm'], categories=existing_algos, ordered=True)

        # 创建画布，尺寸 8x6 与 type1 图一致
        fig, ax = plt.subplots(figsize=(8, 6), dpi=1500)

        # 绘制箱线图，使用 hue 区分四个区域
        sns.boxplot(data=df, x='algorithm', y='damping_ratio', hue='area',
                    palette='Set2', linewidth=1.0, ax=ax)

        # 设置标签与字体（全局 rcParams 已设为 Times New Roman 14，此处显式再设一次）
        ax.set_ylabel('Damping Ratio', fontsize=16)
        ax.set_xlabel('Algorithm', fontsize=16)
        ax.tick_params(axis='x', labelsize=14, rotation=0)  # 横坐标标签水平不倾斜
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        # 修改 x 轴刻度标签：将下划线替换为连字符
        current_xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        new_labels = [label.replace('_', '-') for label in current_xtick_labels]
        ax.set_xticklabels(new_labels, rotation=0, fontsize=14)

        # 图例：放置在图片内部右上角，2列展示，无边框，适当调整位置避免重叠
        handles, labels = ax.get_legend_handles_labels()
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.legend(handles=handles, labels=labels, title='Area', title_fontsize=16,
                  fontsize=16, loc='upper right', bbox_to_anchor=(0.99, 0.89), ncol=2, frameon=False)

        plt.tight_layout()

        # 同时保存 PNG 和 PDF（依赖类内的 _save_figure 方法）
        self._save_figure(fig, "damping_ratio_distribution")

    def generate_performance_table(self, all_episode_data):
        """生成性能对比表格（CSV和LaTeX）"""
        rows = []
        for ep, ep_data in all_episode_data.items():
            for algo, algo_data in ep_data.get('algorithms', {}).items():
                metrics = algo_data.get('metrics', {}).get('performance_metrics', {})
                if not metrics:
                    continue
                # 从step_data计算平均动作幅度
                step_data = algo_data.get('step_data')
                avg_action = 0.0
                if step_data is not None:
                    action_cols = [c for c in step_data.columns if '_action_' in c]
                    if action_cols:
                        avg_action = step_data[action_cols].abs().mean().mean()
                rows.append({
                    'Algorithm': algo,
                    'Episode': ep,
                    'Avg Damping Ratio': metrics.get('avg_damping_ratio', np.nan),
                    'Max Freq Dev (pu)': metrics.get('overall_max_freq_dev', np.nan),
                    'Max Volt Dev (pu)': metrics.get('overall_max_volt_dev', np.nan),
                    'Worst Recovery Time (s)': metrics.get('worst_recovery_time', np.nan),
                    'Avg Action Magnitude': avg_action
                })
        if not rows:
            print("无性能数据")
            return
        df = pd.DataFrame(rows)
        # 按算法分组统计均值±标准差
        grouped = df.groupby('Algorithm').agg({
            'Avg Damping Ratio': ['mean', 'std'],
            'Max Freq Dev (pu)': ['mean', 'std'],
            'Max Volt Dev (pu)': ['mean', 'std'],
            'Worst Recovery Time (s)': ['mean', 'std'],
            'Avg Action Magnitude': ['mean', 'std']
        }).round(4)
        # 保存CSV
        grouped.to_csv(os.path.join(self.plots_dir, 'performance_table.csv'))
        # 保存LaTeX
        with open(os.path.join(self.plots_dir, 'performance_table.tex'), 'w') as f:
            f.write(grouped.to_latex())
        print("✓ 性能表格已保存")

    def generate_all_plots(self):
        """生成所有类型的图表"""
        print("开始生成所有图表...")

        # 加载数据
        all_episode_data = self.load_all_episode_data()
        if not all_episode_data:
            print("没有找到测试数据")
            return

        print(f"加载了 {len(all_episode_data)} 个episode的数据")

        self.plot_type1_statistics(all_episode_data)
        self.plot_damping_ratio_distribution(all_episode_data)

        # 生成类型2图表（使用第一个episode）
        if all_episode_data:
            first_episode = list(all_episode_data.keys())[0]
            self.plot_type2_generator_timeseries(all_episode_data, first_episode)
            self.plot_oscillation_control_comparison([8]) #self.plot_oscillation_control_comparison(list(all_episode_data.keys()))

        self.generate_performance_table(all_episode_data)


def compute_performance_table(result_dir: str, output_csv: str = None):

    if output_csv is None:
        output_csv = os.path.join(result_dir, "algorithms_performance_table.csv")

    # 存储每个算法所有 episode 的原始数据
    algo_episode_data = {}

    # 遍历所有 episode 目录
    episode_dirs = [d for d in os.listdir(result_dir)
                    if d.startswith('episode_') and os.path.isdir(os.path.join(result_dir, d))]

    for episode_dir in episode_dirs:
        episode_path = os.path.join(result_dir, episode_dir)
        episode_idx = int(episode_dir.split('_')[1])

        # 遍历该 episode 下的算法目录
        algo_dirs = [d for d in os.listdir(episode_path)
                     if os.path.isdir(os.path.join(episode_path, d))]

        for algo_name in algo_dirs:
            algo_path = os.path.join(episode_path, algo_name)

            # 1. 读取 system_metrics.json 获取频率、电压峰值
            metrics_file = os.path.join(algo_path, 'system_metrics.json')
            if not os.path.exists(metrics_file):
                continue
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # 提取同步机最大频率偏差
            freq_data = metrics.get('freq_deviation_data', {})
            sync_max_freq_devs = []
            for key, val in freq_data.items():
                if key.startswith('gen_') and key != 'vsg':
                    sync_max_freq_devs.append(val.get('max_freq_dev', 0.0))
            peak_freq_sync = max(sync_max_freq_devs) if sync_max_freq_devs else np.nan

            # 电压最大偏差
            perf_metrics = metrics.get('performance_metrics', {})
            peak_volt = perf_metrics.get('overall_max_volt_dev', np.nan)

            # 2. 读取 step_data.csv 获取各区域阻尼比数据
            step_file = os.path.join(algo_path, 'step_data.csv')
            if not os.path.exists(step_file):
                continue
            step_df = pd.read_csv(step_file)

            # 识别四个区域的阻尼比列（假设列名为 damping_ratio_area_1, ...）
            damping_cols = [col for col in step_df.columns if col.startswith('damping_ratio_')]
            # 提取区域编号
            areas = [col.replace('damping_ratio_', '') for col in damping_cols]
            area_order = ['area_1', 'area_2', 'area_3', 'area_4']
            # 确保四个区域都存在
            area_damping = {}
            for area in area_order:
                col = f'damping_ratio_{area}'
                if col in step_df.columns:
                    vals = step_df[col].dropna().values
                    area_damping[area] = vals
                else:
                    area_damping[area] = np.array([])

            # 计算每个区域的统计量
            area_median = {}
            area_whisker_min = {}
            for area, vals in area_damping.items():
                if len(vals) == 0:
                    area_median[area] = np.nan
                    area_whisker_min[area] = np.nan
                else:
                    # 中位数
                    area_median[area] = np.median(vals)
                    # whisker 下边缘最小值 (Q1 - 1.5*IQR)
                    q1 = np.percentile(vals, 25)
                    q3 = np.percentile(vals, 75)
                    iqr = q3 - q1
                    lower_limit = q1 - 1.5 * iqr
                    filtered = vals[vals >= lower_limit]
                    if len(filtered) > 0:
                        area_whisker_min[area] = np.min(filtered)
                    else:
                        area_whisker_min[area] = np.min(vals)

            # 存储该 episode 的记录
            record = {
                'peak_freq_dev': peak_freq_sync,
                'peak_volt_dev': peak_volt,
                'area_median': area_median,
                'area_whisker_min': area_whisker_min
            }

            if algo_name not in algo_episode_data:
                algo_episode_data[algo_name] = []
            algo_episode_data[algo_name].append(record)

    # 计算每个算法各指标的均值 / 标准差
    results = []
    for algo_name, episodes in algo_episode_data.items():
        if len(episodes) == 0:
            continue

        # 频率峰值
        peak_freqs = [e['peak_freq_dev'] for e in episodes if not np.isnan(e['peak_freq_dev'])]
        mean_freq = np.mean(peak_freqs) if peak_freqs else np.nan
        std_freq = np.std(peak_freqs) if peak_freqs else np.nan

        # 电压峰值
        peak_volts = [e['peak_volt_dev'] for e in episodes if not np.isnan(e['peak_volt_dev'])]
        mean_volt = np.mean(peak_volts) if peak_volts else np.nan
        std_volt = np.std(peak_volts) if peak_volts else np.nan

        # 分区域统计：每个区域的中位数（所有 episode 的平均）、whisker 最小值（所有 episode 的平均）
        area_order = ['area_1', 'area_2', 'area_3', 'area_4']
        area_median_means = {}
        area_whisker_means = {}
        for area in area_order:
            medians = [e['area_median'][area] for e in episodes if not np.isnan(e['area_median'][area])]
            whiskers = [e['area_whisker_min'][area] for e in episodes if not np.isnan(e['area_whisker_min'][area])]
            area_median_means[area] = np.mean(medians) if medians else np.nan
            area_whisker_means[area] = np.mean(whiskers) if whiskers else np.nan

        # 显示名称映射
        display_name = algo_name.replace('_', '-')
        if display_name == 'WSE-MATD3':
            display_name = 'WSE-MATD3'
        elif display_name == 'NSGA2-MATD3':
            display_name = 'NSGA2-MATD3'

        row = {
            'Algorithm': display_name,
            'Mean Peak Freq. Dev. (p.u.)': mean_freq,
            'Std Peak Freq. Dev. (p.u.)': std_freq,
            'Mean Peak Volt. Dev. (p.u.)': mean_volt,
            'Std Peak Volt. Dev. (p.u.)': std_volt,
            'Area1 Median Damping': area_median_means['area_1'],
            'Area1 Min Damping (whisker)': area_whisker_means['area_1'],
            'Area2 Median Damping': area_median_means['area_2'],
            'Area2 Min Damping (whisker)': area_whisker_means['area_2'],
            'Area3 Median Damping': area_median_means['area_3'],
            'Area3 Min Damping (whisker)': area_whisker_means['area_3'],
            'Area4 Median Damping': area_median_means['area_4'],
            'Area4 Min Damping (whisker)': area_whisker_means['area_4'],
        }
        results.append(row)

    if not results:
        print("错误：未找到任何有效数据，请检查 result_dir 路径是否正确。")
        return None

    df_result = pd.DataFrame(results)

    # 按指定顺序排序
    algo_order = ['MAPPO', 'MASAC', 'MATD3', 'NSGA2-MATD3', 'WSE-MATD3']
    df_result['Algorithm'] = pd.Categorical(df_result['Algorithm'], categories=algo_order, ordered=True)
    df_result = df_result.sort_values('Algorithm').reset_index(drop=True)

    # 保存 CSV
    df_result.to_csv(output_csv, index=False)
    print(f"分区域性能表格已保存到: {output_csv}")
    return df_result

def main():
    """主函数"""
    print("IEEE 39-bus多智能体控制算法对比测试")
    print("=" * 60)

    # 定义要测试的算法
    algorithms = ['MAPPO', 'MASAC','MATD3', 'NSGA2_MATD3', 'WSE_MATD3']

    model_base_path = "MARL_Train"
    result_dir_name = "MARL_Test"

    tester = EnhancedMARL_Tester(algorithms, model_base_path=model_base_path, result_dir=result_dir_name)
    # 不再需要单独设置 tester.result_dir

    # 运行测试
    print(f"开始测试 {len(algorithms)} 种算法...")
    test_results, sim_data = tester.run_all_tests(
        n_episodes=10,  # 减少episodes数以加快测试
        seeds=list(range(42, 52))  #[42, 43, 44]
    )

    # 生成可视化图表
    print("\n开始生成可视化图表...")
    df_table = compute_performance_table(result_dir_name)
    if df_table is not None:
        print(df_table.to_string())

    visualizer = EnhancedMARL_Visualizer(result_dir=result_dir_name)
    visualizer.generate_all_plots()

    print("\n所有测试和可视化完成！")


if __name__ == "__main__":
    main()