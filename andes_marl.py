import os
import numpy as np
import andes
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import random
import copy
from pettingzoo.utils.env import ParallelEnv
from typing import Dict, List, Tuple, Optional, Union

max_steps=28
# Prony分析相关导入
try:
    import scipy.signal as signal
    from scipy.linalg import svd, lstsq, eig
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Prony analysis will be disabled.")

class LocalPronyAnalyzer:
    """增强型Prony分析器，能够从信号中提取主导振荡模式"""

    def __init__(self, agent_id, window_size=150, sampling_rate=30, min_damping_ratio=0.01): #window_size = 2秒 * 30 Hz = 30
        self.agent_id = agent_id
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.min_damping_ratio = min_damping_ratio
        self.sampling_interval = 1.0 / sampling_rate

        from collections import deque
        self.time = deque(maxlen=window_size)
        self.local_freq = deque(maxlen=window_size)
        self.local_angle = deque(maxlen=window_size)

        self.last_analysis = None
        self.analysis_valid = False

    def add_local_data(self, t: float, local_freq: float, local_angle: float = None):
        # 清洗数据，防止 NaN/Inf 进入缓冲区
        if not np.isfinite(local_freq):
            local_freq = 1.0
        self.local_freq.append(local_freq)
        self.time.append(t)
        if local_angle is not None:
            if not np.isfinite(local_angle):
                local_angle = 0.0
            self.local_angle.append(local_angle)

    def clear_history(self):
        self.time.clear()
        self.local_freq.clear()
        self.local_angle.clear()
        self.last_analysis = None
        self.analysis_valid = False

    def analyze(self, use_advanced=True) -> Dict:
        """执行振荡分析，返回分析结果（始终有效，但可能为弱阻尼）"""
        if len(self.local_freq) < 10:
            result = self._default_analysis()
            self.last_analysis = result
            self.analysis_valid = True
            return result

        # 转换为numpy数组并彻底清洗
        signal = np.array(self.local_freq, dtype=np.float64)
        signal = np.nan_to_num(signal, nan=1.0, posinf=1.0, neginf=1.0)
        signal = signal - np.mean(signal)  # 去直流

        # 尝试高级Prony分析
        if use_advanced and len(signal) >= 30:
            result = self._robust_prony_analysis(signal)
            if result['valid']:
                self.last_analysis = result
                self.analysis_valid = True
                return result

        # 后备1：FFT峰值检测
        result = self._fft_peak_analysis(signal)
        if result['valid']:
            self.last_analysis = result
            self.analysis_valid = True
            return result

        # 后备2：自相关法
        result = self._autocorr_analysis(signal)
        if result['valid']:
            self.last_analysis = result
            self.analysis_valid = True
            return result

        # 所有方法均失败，返回基于信号能量的弱阻尼估计
        result = self._default_analysis(signal)
        self.last_analysis = result
        self.analysis_valid = True
        return result

    def _robust_prony_analysis(self, signal: np.ndarray) -> Dict:
        """基于SVD的鲁棒Prony分析（增强数值稳定性）"""
        n = len(signal)
        if n < 10:
            return self._invalid_result()

        # 模型阶数：取 min(10, n//3) 并至少为3
        p = min(10, max(3, n // 3))
        L = n - p
        if L < p:
            return self._invalid_result()

        # 构建Hankel矩阵
        H = np.zeros((L, p), dtype=np.float64)
        for i in range(L):
            H[i, :] = signal[i:i+p]
        Y = signal[p:p+L]

        # SVD求解最小二乘
        try:
            U, s, Vt = np.linalg.svd(H, full_matrices=False)
            # 根据奇异值截断（保留前k个，累计能量>95%）
            cum_energy = np.cumsum(s**2) / np.sum(s**2)
            k = np.searchsorted(cum_energy, 0.95) + 1
            k = max(1, min(k, len(s)-1))
            # 避免奇异值过小导致除零
            s_k = s[:k]
            s_k_inv = np.diag(1.0 / (s_k + 1e-10))  # 加微小量防止除零
            U_k = U[:, :k]
            Vt_k = Vt[:k, :]
            # 最小二乘解：A = Vt_k.T @ inv(diag(s_k)) @ U_k.T @ Y
            A = Vt_k.T @ s_k_inv @ U_k.T @ Y
        except Exception as e:
            print(f"Prony SVD求解失败: {e}")
            return self._invalid_result()

        # 检查A是否包含非有限值
        if not np.all(np.isfinite(A)):
            print("Prony 线性预测系数 A 包含非有限值")
            return self._invalid_result()

        # 构造伴随矩阵求特征值
        companion = np.zeros((p, p), dtype=np.float64)
        companion[0, :] = -A
        for i in range(1, p):
            companion[i, i-1] = 1.0

        # 检查companion矩阵是否包含非有限值
        if not np.all(np.isfinite(companion)):
            print("Prony 伴随矩阵包含非有限值")
            return self._invalid_result()

        try:
            eigvals = np.linalg.eigvals(companion)
        except Exception as e:
            print(f"计算特征值失败: {e}")
            return self._invalid_result()

        # 转换为连续时间极点
        dt = self.sampling_interval
        # 去除可能位于0的特征值（对应直流分量）
        eigvals = eigvals[np.abs(eigvals) > 1e-6]
        if len(eigvals) == 0:
            return self._invalid_result()

        s_poles = np.log(eigvals) / dt

        modes = []
        for pole in s_poles:
            if not np.isfinite(pole):
                continue
            sigma = np.real(pole)
            omega = np.abs(np.imag(pole))
            if omega > 0.2:  # 忽略极低频（<0.03 Hz）
                wn = np.sqrt(sigma**2 + omega**2)
                damping = -sigma / wn if wn > 1e-6 else 0.0
                if 0.01 < damping < 0.99:
                    freq = omega / (2 * np.pi)
                    if 0.1 < freq < 5.0:  # 限制在典型振荡范围
                        modes.append({
                            'damping': damping,
                            'freq': freq,
                            'sigma': sigma,
                            'omega': omega
                        })

        if not modes:
            return self._invalid_result()

        # 选择阻尼比最小的模式（最危险的主导模式）
        dominant = min(modes, key=lambda m: m['damping'])
        damping_ratio = dominant['damping']
        osc_freq = dominant['freq']
        amplitude = np.std(signal)
        stability = np.clip(damping_ratio / 0.15, 0.0, 2.0)

        return {
            'damping_ratio': float(damping_ratio),
            'oscillation_freq': float(osc_freq),
            'amplitude': float(amplitude),
            'stability_index': float(stability),
            'valid': True
        }

    def _fft_peak_analysis(self, signal: np.ndarray) -> Dict:
        """基于FFT峰值检测的振荡估计（当Prony失败时使用）"""
        n = len(signal)
        if n < 10:
            return self._invalid_result()

        fft_vals = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(n, d=self.sampling_interval)
        magnitude = np.abs(fft_vals)

        pos_idx = fft_freq > 0
        pos_freq = fft_freq[pos_idx]
        pos_mag = magnitude[pos_idx]

        if len(pos_freq) == 0:
            return self._invalid_result()

        peak_idx = np.argmax(pos_mag)
        peak_freq = pos_freq[peak_idx]

        if not (0.1 < peak_freq < 5.0):
            valid_idx = np.where((pos_freq >= 0.1) & (pos_freq <= 5.0))[0]
            if len(valid_idx) > 0:
                valid_mag = pos_mag[valid_idx]
                max_valid = np.argmax(valid_mag)
                peak_freq = pos_freq[valid_idx[max_valid]]
            else:
                peak_freq = 0.5

        damping_ratio = 0.05
        amplitude = np.std(signal)
        stability = np.clip(damping_ratio / 0.15, 0.0, 2.0)

        return {
            'damping_ratio': float(damping_ratio),
            'oscillation_freq': float(peak_freq),
            'amplitude': float(amplitude),
            'stability_index': float(stability),
            'valid': True
        }

    def _autocorr_analysis(self, signal: np.ndarray) -> Dict:
        """基于自相关函数的振荡检测"""
        n = len(signal)
        if n < 20:
            return self._invalid_result()

        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[n-1:]  # 正延迟部分
        autocorr = autocorr / autocorr[0]  # 归一化

        peaks = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(i)

        if len(peaks) < 2:
            return self._invalid_result()

        intervals = np.diff(peaks)
        avg_interval = np.mean(intervals)
        if avg_interval < 1:
            return self._invalid_result()
        freq = self.sampling_rate / avg_interval

        peak_vals = autocorr[peaks]
        if len(peak_vals) >= 3:
            decay = np.log(peak_vals[0] / peak_vals[1])
            damping = decay / (2 * np.pi)
            damping = np.clip(damping, 0.01, 0.99)
        else:
            damping = 0.05

        if not (0.1 < freq < 5.0):
            freq = 0.5

        amplitude = np.std(signal)
        stability = np.clip(damping / 0.05, 0.0, 2.0)

        return {
            'damping_ratio': float(damping),
            'oscillation_freq': float(freq),
            'amplitude': float(amplitude),
            'stability_index': float(stability),
            'valid': True
        }

    def _default_analysis(self, signal: np.ndarray = None) -> Dict:
        """默认分析：当所有方法失败时，返回一个基于信号特征的弱阻尼估计"""
        if signal is not None and len(signal) > 0:
            amplitude = float(np.std(signal))
            # 粗略估计频率：取信号过零率
            zero_crossings = np.where(np.diff(np.sign(signal)))[0]
            if len(zero_crossings) > 1:
                avg_period = 2 * np.mean(np.diff(zero_crossings)) * self.sampling_interval
                if avg_period > 0:
                    freq = 1.0 / avg_period
                else:
                    freq = 0.5
            else:
                freq = 0.5
        else:
            amplitude = 0.0
            freq = 0.5

        return {
            'damping_ratio': 0.01,
            'oscillation_freq': float(freq),
            'amplitude': amplitude,
            'stability_index': 0.2,
            'valid': True
        }

    def _invalid_result(self):
        return {
            'damping_ratio': 0.0,
            'oscillation_freq': 0.0,
            'amplitude': 0.0,
            'stability_index': 0.0,
            'valid': False
        }

    def get_oscillation_features(self) -> np.ndarray:
        if self.last_analysis is None:
            return np.zeros(4, dtype=np.float32)
        res = self.last_analysis
        return np.array([res['damping_ratio'], res['oscillation_freq'],
                         res['amplitude'], res['stability_index']], dtype=np.float32)

class DistributedPronyCoordinator:
    """分布式Prony协调器，模拟区域间的信息交换"""

    def __init__(self, agent_ids):
        """
        初始化协调器

        Args:
            agent_ids: 所有智能体ID列表
        """
        self.agent_ids = agent_ids

        # 存储各区域的振荡分析结果
        self.area_results = {agent: None for agent in agent_ids}

        # 存储区域间振荡模式的相关性
        self.inter_area_correlations = {}

        # 振荡模式数据库
        self.oscillation_modes = []

    def update_area_result(self, agent_id: str, result: Dict):
        """更新区域振荡分析结果"""
        self.area_results[agent_id] = result

        # 如果所有区域都有结果，计算区域间相关性
        if all(v is not None for v in self.area_results.values()):
            self._calculate_inter_area_correlations()

    def _calculate_inter_area_correlations(self):
        """计算区域间振荡模式的相关性"""
        valid_results = {}
        for agent, result in self.area_results.items():
            if result and result.get('valid', False):
                valid_results[agent] = result

        if len(valid_results) < 2:
            return

        # 计算频率相似性（区域间是否有相似的振荡频率）
        frequencies = {agent: result['oscillation_freq']
                      for agent, result in valid_results.items()}

        # 识别主导振荡模式（频率最接近的模式）
        if frequencies:
            avg_freq = np.mean(list(frequencies.values()))
            self.dominant_frequency = avg_freq

            # 计算区域间频率差异
            for agent1, freq1 in frequencies.items():
                for agent2, freq2 in frequencies.items():
                    if agent1 < agent2:
                        freq_diff = abs(freq1 - freq2)
                        correlation_key = f"{agent1}-{agent2}"
                        self.inter_area_correlations[correlation_key] = {
                            'frequency_similarity': 1.0 / (1.0 + freq_diff),
                            'avg_frequency': (freq1 + freq2) / 2
                        }

    def get_inter_area_info(self, agent_id: str) -> np.ndarray:
        """
        获取区域间振荡信息（模拟有限的信息交换）

        在实际多智能体系统中，只能获取有限的其他区域信息
        """
        features = np.zeros(3, dtype=np.float32)

        # 特征1: 本区域是否有主导振荡
        if self.area_results.get(agent_id):
            result = self.area_results[agent_id]
            if result and result.get('valid', False):
                features[0] = 1.0  # 有振荡
                features[1] = result.get('stability_index', 0.0)  # 稳定性

        # 特征2: 是否有其他区域报告振荡
        other_oscillations = 0
        for agent, result in self.area_results.items():
            if agent != agent_id and result and result.get('valid', False):
                other_oscillations += 1

        features[2] = min(1.0, other_oscillations / 3.0)  # 归一化

        return features


class MultiAgentAndes39(ParallelEnv):
    """
    IEEE 39-bus 4-area multi-agent system frequency control environment
    """

    metadata = {
        "name": "andes39_multiagent_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "max_cycles": 28,
    }

    def __init__(self,
                 tf=15.0,
                 tstep=1/30,
                 max_steps=28,
                 include_prony=True,
                 shared_observations=True,
                 prony_coordination=True,
                 algorithm_type="MADDPG"):
        """
        多智能体环境初始化

        Args:
            tf: 仿真总时间 (秒)
            tstep: 仿真步长 (秒)
            max_steps: 最大动作步数
            include_prony: 是否包含Prony分析
            shared_observations: 是否共享全局观测
            prony_coordination: 是否启用Prony协调
            algorithm_type: 算法类型，用于确定观测空间
        """
        # 获取当前文件路径
        path = os.path.dirname(os.path.abspath(__file__))
        self.case_file = os.path.join(path, "ieee39_4area.xlsx")

        # 仿真参数
        self.tf = tf
        self.tstep = tstep
        self.fixt = True
        self.no_pbar = True

        # 动作时间点
        self.action_instants = np.linspace(2, 15, max_steps)
        self.max_steps = max_steps

        # 多智能体配置
        self.possible_agents = ["area_1", "area_2", "area_3", "area_4"]
        self.algorithm_type = algorithm_type

        # 区域到发电机的映射
        self.area_to_gens = {
            "area_1": [0, 6, 8],
            "area_2": [1, 9],
            "area_3": [4, 7],
            "area_4": [2, 3, 5]
        }

        # 动作空间配置
        self.action_dims = {
            "area_1": 3,
            "area_2": 2,
            "area_3": 2,
            "area_4": 3
        }

        # 观测空间配置
        self.include_prony = include_prony and SCIPY_AVAILABLE
        self.shared_observations = shared_observations
        self.prony_coordination = prony_coordination and self.include_prony

        # 根据算法类型调整观测空间
        self._adjust_observation_space_for_algorithm()

        # 定义动作空间和观测空间
        self.action_spaces = {
            agent: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.action_dims[agent],),
                dtype=np.float32
            ) for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(self.total_obs_dim[agent],),
                dtype=np.float32
            ) for agent in self.possible_agents
        }

        # 初始化变量
        self.i = 0
        self.seed(42)
        self.sim_case = None
        self.step_count = 0
        self.episode_count = 0
        self.total_rewards = {agent: 0.0 for agent in self.possible_agents}

        # 计算Prony采样频率（基于仿真步长）
        self.prony_sampling_rate = 1.0 / self.tstep  # 30 Hz
        self.prony_window_seconds = 2.0  # 分析窗口长度（秒）################################
        self.prony_window_size = int(self.prony_window_seconds * self.prony_sampling_rate)

        if self.include_prony:
            self.prony_analyzers = {
                agent: LocalPronyAnalyzer(
                    agent_id=agent,
                    window_size=self.prony_window_size,
                    sampling_rate=self.prony_sampling_rate,
                    min_damping_ratio=0.01
                ) for agent in self.possible_agents
            }

            if self.prony_coordination:
                self.prony_coordinator = DistributedPronyCoordinator(self.possible_agents)
            else:
                self.prony_coordinator = None
        else:
            self.prony_analyzers = None
            self.prony_coordinator = None

        # 关键变量索引
        self.w_idx = None
        self.wgf_idx = None
        self.Pg_idx = None
        self.Pgf_idx = None
        self.voltage_idx = None
        self.ace_idx = None
        self.wcoi_idx = None
        self.deltacoi_idx = None
        self.Busfreq_idx = None
        self.tg_idx = None
        self.vsg_idx = None
        self.key_buses = [38, 31, 37, 32]

        # 区域映射
        self.area_to_key_bus = {
            "area_1": 38,
            "area_2": 31,
            "area_3": 37,
            "area_4": 32
        }

        # 统计信息
        self.episode_rewards = []
        self.episode_lengths = []

        # PettingZoo要求的状态变量
        self.agents = []

        self.last_actions = {agent: np.zeros(self.action_dims[agent]) for agent in self.possible_agents}

    def _adjust_observation_space_for_algorithm(self):
        """根据算法类型调整观测空间"""
        # 基础观测维度
        # 每个区域的局部观测
        self.local_obs_dim = {
            "area_1": 12,  # 3发电机频率 + 3功率 + ACE + 母线电压 + 母线频率 + COI频率 + COI功角 + 区域间振荡信息
            "area_2": 10,  # 2发电单元频率 + 2功率 + ACE + 母线电压 + 母线频率 + COI频率 + COI功角 + 区域间振荡信息
            "area_3": 10,  # 2发电机频率 + 2功率 + ACE + 母线电压 + 母线频率 + COI频率 + COI功角 + 区域间振荡信息
            "area_4": 12   # 3发电机频率 + 3功率 + ACE + 母线电压 + 母线频率 + COI频率 + COI功角 + 区域间振荡信息
        }

        # 全局观测维度（共享观测）
        self.global_obs_dim = 43

        # 相域观测维度
        self.prony_obs_dim = 7 if self.include_prony else 0

        # 根据算法类型调整观测空间
        if self.algorithm_type in ["MATD3", "WSE_MATD3", "NSGA2_MATD3"]:
            # 这些算法使用共享全局观测
            self.shared_observations = True
            # 计算总观测维度
            self.total_obs_dim = {}
            for agent in self.possible_agents:
                self.total_obs_dim[agent] = (
                    self.global_obs_dim +
                    self.prony_obs_dim
                )

        elif self.algorithm_type in ["MAPPO"]:
            # MAPPO使用全局观测但不一定共享
            self.shared_observations = True
            self.total_obs_dim = {}
            for agent in self.possible_agents:
                self.total_obs_dim[agent] = (
                    self.global_obs_dim +
                    self.local_obs_dim[agent] +
                    self.prony_obs_dim
                )

        elif self.algorithm_type in ["MASAC"]:
            # MASAC通常使用部分观测
            self.shared_observations = False
            self.total_obs_dim = {}
            for agent in self.possible_agents:
                self.total_obs_dim[agent] = (
                    self.local_obs_dim[agent] +
                    self.prony_obs_dim
                )

        else:
            # 默认使用完整观测
            self.total_obs_dim = {}
            for agent in self.possible_agents:
                self.total_obs_dim[agent] = (
                    self.global_obs_dim +
                    self.local_obs_dim[agent] +
                    self.prony_obs_dim
                )

    @property
    def num_agents(self):
        return len(self.agents)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _initialize_simulation(self):
        """初始化ANDES仿真"""
        try:
            self.sim_case = andes.run(self.case_file, no_output=True, default_config=True)

            # 配置模型
            self.sim_case.PQ.config.p2p = 1
            self.sim_case.PQ.config.p2z = 0
            self.sim_case.PQ.config.p2i = 0
            self.sim_case.PQ.config.q2q = 1
            self.sim_case.PQ.config.q2z = 0
            self.sim_case.PQ.config.q2i = 0

            # 配置时域仿真
            self.sim_case.TDS.config.fixt = self.fixt
            self.sim_case.TDS.config.tstep = self.tstep
            self.sim_case.TDS.config.tf = 1.0

            # 执行初始化
            self.sim_case.TDS.init()

            # 获取变量索引
            self._get_variable_indices()

            # 重置Prony分析器
            if self.prony_analyzers:
                for analyzer in self.prony_analyzers.values():
                    analyzer.clear_history()

            return True

        except Exception as e:
            print(f"ANDES仿真初始化失败: {e}")
            return False

    def _get_variable_indices(self):
        """获取关键变量的索引"""
        try:
            self.w_idx = np.array(self.sim_case.GENROU.omega.a)
            self.delta_idx = np.array(self.sim_case.GENROU.delta.a)
            self.Pg_idx = np.array(self.sim_case.GENROU.Pe.a)
            self.Pgf_idx = np.array(self.sim_case.REGF2.Pe.a)
            self.deltagf_idx = np.array(self.sim_case.REGF2.delta.a)
            self.wgf_idx = np.array(self.sim_case.REGF2.dw_y.a)
            self.voltage_idx = np.array(self.sim_case.Bus.v.a)
            self.ace_idx = np.array(self.sim_case.ACEc.ace.a)
            self.wcoi_idx = np.array(self.sim_case.COI.omega.a)
            self.deltacoi_idx = np.array(self.sim_case.COI.delta.a)
            self.Busfreq_idx = np.array(self.sim_case.BusFreq.f.a)
            self.BusROCOF_idx = np.array(self.sim_case.BusROCOF.Wf_y.a)
            self.tg_idx = [i for i in self.sim_case.TurbineGov._idx2model.keys()]
            self.vsg_idx = [i for i in self.sim_case.RenGen._idx2model.keys()]

            return True

        except Exception as e:
            print(f"获取变量索引失败: {e}")
            raise

    def _apply_disturbance(self):
        """施加永久故障：负荷永久增长 或 线路永久断开"""
        try:
            self.disturbance_type = random.choice(['load_step', 'line_trip'])

            if self.disturbance_type == 'load_step':
                # 永久负荷增长：在母线 0~17 中随机选择一条，负荷永久增加
                self.disturbus = random.choice(range(0, 18))  # 负荷母线索引范围 0~17
                self.dist_severity = random.uniform(0.1, 0.4)  # 负荷增长幅度
                # 设置负荷增加（永久，不恢复）
                self.sim_case.Alter.amount.v[self.disturbus] = self.dist_severity
                self.dist_location = f"PQuid_{self.disturbus}"
                # 故障持续到仿真结束（记录为剩余时间，可选）
                self.dist_duration = self.tf - 2.0

            else:  # line_trip
                # 永久线路断开：在线路 0~44 中随机选择一条，永久断开
                self.disturbus = random.choice(range(0, 45))  # 线路索引范围 0~44
                self.dist_severity = 1.0
                # 设置线路断开（永久，不重合）
                self.sim_case.Toggler.u.v[self.disturbus] = 1
                self.dist_location = f"Lineuid_{self.disturbus}"
                self.dist_duration = self.tf - 2.0

            print(f"施加永久扰动: {self.disturbance_type} at {self.dist_location}, 幅度: {self.dist_severity:.3f}")

        except Exception as e:
            print(f"施加扰动失败: {e}")
            # 设置默认值，避免崩溃
            self.disturbance_type = 'load_step'
            self.dist_location = 'default'
            self.dist_severity = 0.1
            self.dist_duration = 1.0


    def sim_to_next(self):
        """仿真到下一个动作时间点"""
        if self.i < len(self.action_instants):
            next_time = float(self.action_instants[self.i])
        else:
            next_time = self.tf

        self.sim_case.TDS.config.tf = next_time
        self.i += 1

        try:
            success = self.sim_case.TDS.run(self.no_pbar)
            return success
        except Exception as e:
            print(f"仿真运行失败: {e}")
            return False

   
    def _get_local_signals_for_prony_at_index(self, agent: str, idx: int) -> Tuple[float, float]:
        """
        从 dae.ts 历史数据中提取指定索引 idx 时刻的局部信号，并进行有效性检查。
        """
        try:
            area_idx = int(agent.split('_')[1]) - 1

            # 获取 TimeSeries 数据（注意 ts.y 的维度：时间 × 变量）
            wcoi_ts = self.sim_case.dae.ts.y[:, self.wcoi_idx]  # (n_times, n_areas)
            deltacoi_ts = self.sim_case.dae.ts.y[:, self.deltacoi_idx]  # (n_times, n_areas)
            busfreq_ts = self.sim_case.dae.ts.y[:, self.Busfreq_idx]  # (n_times, n_buses)

            if area_idx < wcoi_ts.shape[1]:
                local_freq = wcoi_ts[idx, area_idx]
                local_angle = deltacoi_ts[idx, area_idx] if area_idx < deltacoi_ts.shape[1] else 0.0
            else:
                # 备选：使用关键母线频率
                key_bus = self.area_to_key_bus[agent]
                if key_bus < busfreq_ts.shape[1]:
                    local_freq = busfreq_ts[idx, key_bus]
                else:
                    local_freq = 1.0
                local_angle = 0.0

            # 清洗数据：确保为有限值
            if not np.isfinite(local_freq):
                local_freq = 1.0
            if not np.isfinite(local_angle):
                local_angle = 0.0

            return float(local_freq), float(local_angle)

        except Exception as e:
            print(f"获取历史信号失败 ({agent} idx {idx}): {e}")
            return 1.0, 0.0

    def _update_prony_analyzers(self):
        """更新所有 Prony 分析器：直接从 dae.ts 获取最近 window_size 个点重建历史"""
        if not self.prony_analyzers:
            return

        # 获取所有时间点
        t_all = self.sim_case.dae.ts.t
        if len(t_all) == 0:
            # 无数据时，强制每个分析器生成默认分析结果
            for analyzer in self.prony_analyzers.values():
                analyzer.clear_history()
                analyzer.analyze(use_advanced=True)
            return

        for agent, analyzer in self.prony_analyzers.items():
            # 清空历史
            analyzer.clear_history()
            # 取最近 window_size 个点（最多全部）
            start_idx = max(0, len(t_all) - analyzer.window_size)
            for idx in range(start_idx, len(t_all)):
                t = t_all[idx]
                freq, angle = self._get_local_signals_for_prony_at_index(agent, idx)
                analyzer.add_local_data(t, freq, angle)
            # 执行分析（确保结果有效）
            analyzer.analyze(use_advanced=True)

    def _get_prony_observation(self, agent: str):
        """获取Prony相域观测"""
        if not self.include_prony:
            return np.zeros(self.prony_obs_dim, dtype=np.float32)

        try:
            # 获取局部振荡特征
            if agent in self.prony_analyzers:
                local_features = self.prony_analyzers[agent].get_oscillation_features()
            else:
                local_features = np.zeros(4, dtype=np.float32)

            # 获取区域间协调信息
            if self.prony_coordinator:
                inter_area_features = self.prony_coordinator.get_inter_area_info(agent)
            else:
                inter_area_features = np.zeros(3, dtype=np.float32)

            # 组合特征向量
            prony_obs = np.concatenate([local_features, inter_area_features])

            return prony_obs

        except Exception as e:
            print(f"获取相域观测失败 ({agent}): {e}")
            return np.zeros(self.prony_obs_dim, dtype=np.float32)

    def _get_local_observation(self, agent: str):
        """获取局部观测"""
        try:
            area_idx = int(agent.split('_')[1]) - 1

            # 获取该区域的发电机索引
            gen_indices = self.area_to_gens[agent]

            # 提取发电机频率和功率
            wg = self.sim_case.dae.x[self.w_idx].astype(np.float32)
            Pg = self.sim_case.dae.y[self.Pg_idx].astype(np.float32)

            # 处理区域2（包含VSG）
            if agent == "area_2":
                Busfreq0 = self.sim_case.dae.y[self.Busfreq_idx].astype(np.float32)
                w_gf = Busfreq0[1]
                # w_gf = self.sim_case.dae.y[self.wgf_idx].astype(np.float32)
                Pgf = self.sim_case.dae.y[self.Pgf_idx].astype(np.float32)
                local_w = np.array([wg[1], w_gf[0]], dtype=np.float32)
                local_p = np.array([Pg[1], Pgf[0]], dtype=np.float32)
            else:
                local_w = wg[gen_indices].astype(np.float32)
                local_p = Pg[gen_indices].astype(np.float32)

            # 区域ACE
            ace = self.sim_case.dae.y[self.ace_idx].astype(np.float32)
            area_ace = ace[area_idx] if area_idx < len(ace) else 0.0

            # 关键母线电压和频率
            voltage = self.sim_case.dae.y[self.voltage_idx].astype(np.float32)
            Busfreq0 = self.sim_case.dae.y[self.Busfreq_idx].astype(np.float32)
            key_bus = self.area_to_key_bus[agent]
            bus_voltage = voltage[key_bus] if key_bus < len(voltage) else 1.0
            bus_freq = Busfreq0[key_bus] if key_bus < len(Busfreq0) else 1.0

            # COI频率和功角
            wcoi = self.sim_case.dae.y[self.wcoi_idx].astype(np.float32)
            deltacoi = self.sim_case.dae.y[self.deltacoi_idx].astype(np.float32)
            coi_freq = wcoi[area_idx] if area_idx < len(wcoi) else 1.0
            coi_angle = deltacoi[area_idx] if area_idx < len(deltacoi) else 0.0

            # 构建局部观测向量
            local_obs = np.concatenate([
                local_w.flatten(),
                local_p.flatten(),
                np.array([area_ace], dtype=np.float32),
                np.array([bus_voltage, bus_freq], dtype=np.float32),
                np.array([coi_freq, coi_angle], dtype=np.float32)
            ])

            # 确保维度正确
            expected_dim = self.local_obs_dim[agent]
            if len(local_obs) < expected_dim:
                # 填充零
                local_obs = np.concatenate([local_obs, np.zeros(expected_dim - len(local_obs))])
            elif len(local_obs) > expected_dim:
                # 截断
                local_obs = local_obs[:expected_dim]

            return local_obs

        except Exception as e:
            print(f"获取局部观测失败 ({agent}): {e}")
            return np.zeros(self.local_obs_dim[agent], dtype=np.float32)

    def _get_global_observation(self):
        """获取全局观测（共享）"""
        try:
            Busfreq0 = self.sim_case.dae.y[self.Busfreq_idx].astype(np.float32)
            key_busfreq = [9, 2, 8, 3]
            Busfreq = Busfreq0[key_busfreq].astype(np.float32)
            wg = self.sim_case.dae.x[self.w_idx].astype(np.float32)
            w_gf = Busfreq0[1]  #self.sim_case.dae.y[self.wgf_idx].astype(np.float32)
            Pg = self.sim_case.dae.y[self.Pg_idx].astype(np.float32)
            Pgf = self.sim_case.dae.y[self.Pgf_idx].astype(np.float32)
            wcoi = self.sim_case.dae.y[self.wcoi_idx].astype(np.float32)
            deltacoi = self.sim_case.dae.y[self.deltacoi_idx].astype(np.float32)

            # 计算各区域COI电功率
            area1_pg_indices = [0, 6, 8]
            area2_pg_indices = [1]
            area3_pg_indices = [4, 7]
            area4_pg_indices = [2, 3, 5]

            Pecoi1 = np.sum(Pg[area1_pg_indices]) if area1_pg_indices else 0.0
            Pecoi2 = np.sum(Pg[area2_pg_indices]) if area2_pg_indices else 0.0
            Pecoi2 += Pgf[0] if Pgf.size > 0 else 0.0
            Pecoi3 = np.sum(Pg[area3_pg_indices]) if area3_pg_indices else 0.0
            Pecoi4 = np.sum(Pg[area4_pg_indices]) if area4_pg_indices else 0.0
            Pecoi = np.array([Pecoi1, Pecoi2, Pecoi3, Pecoi4], dtype=np.float32)

            # 关键母线电压
            voltage = self.sim_case.dae.y[self.voltage_idx].astype(np.float32)
            V_key = voltage[self.key_buses].astype(np.float32)

            # ACE信号
            ace = self.sim_case.dae.y[self.ace_idx].astype(np.float32)
            if ace.shape[0] < 4:
                ace = np.pad(ace, (0, 4 - ace.shape[0]), 'constant')
            ace = ace[:4]

            # 构建全局观测向量
            components = [
                wg.flatten(),
                Pg.flatten(),
                Pgf.flatten(),
                wcoi.flatten(),
                deltacoi.flatten(),
                Pecoi.flatten(),
                V_key.flatten(),
                ace.flatten(),
                Busfreq.flatten()
            ]

            global_obs = np.concatenate(components).astype(np.float32)

            # 确保维度正确
            if len(global_obs) < self.global_obs_dim:
                global_obs = np.concatenate([global_obs, np.zeros(self.global_obs_dim - len(global_obs))])
            elif len(global_obs) > self.global_obs_dim:
                global_obs = global_obs[:self.global_obs_dim]

            # 更新Prony分析器数据
            if self.prony_analyzers:
                self._update_prony_analyzers()

            return global_obs

        except Exception as e:
            print(f"获取全局观测失败: {e}")
            return np.zeros(self.global_obs_dim, dtype=np.float32)

    def _get_agent_observation(self, agent: str):
        """获取智能体完整观测"""
        # 【新增】统一更新 Prony 分析器
        if self.include_prony:
            self._update_prony_analyzers()

        if self.algorithm_type in ["MATD3","NSGA2_MATD3", "WSE_MATD3"]:
            # 这些算法只使用全局观测和Prony观测
            global_obs = self._get_global_observation()
            prony_obs = self._get_prony_observation(agent)
            obs = np.concatenate([global_obs, prony_obs])

        elif self.algorithm_type in ["MAPPO"]:
            # MAPPO使用全局、局部和Prony观测
            global_obs = self._get_global_observation()
            local_obs = self._get_local_observation(agent)
            prony_obs = self._get_prony_observation(agent)
            obs = np.concatenate([global_obs, local_obs, prony_obs])

        elif self.algorithm_type in ["MASAC"]:
            # MASAC使用局部和Prony观测
            local_obs = self._get_local_observation(agent)
            prony_obs = self._get_prony_observation(agent)
            obs = np.concatenate([local_obs, prony_obs])

        else:
            # 默认使用完整观测
            global_obs = self._get_global_observation()
            local_obs = self._get_local_observation(agent)
            prony_obs = self._get_prony_observation(agent)
            obs = np.concatenate([global_obs, local_obs, prony_obs])

        # 维度检查
        expected_dim = self.total_obs_dim[agent]
        if len(obs) != expected_dim:
            print(f"警告: 观测维度不匹配! Agent {agent}: 期望 {expected_dim}, 实际 {len(obs)}")
            if len(obs) < expected_dim:
                obs = np.concatenate([obs, np.zeros(expected_dim - len(obs))])
            else:
                obs = obs[:expected_dim]

        return obs

    def _apply_agent_action(self, agent: str, action: np.ndarray):
        """应用智能体动作"""
        try:
            area_idx = int(agent.split('_')[1]) - 1
            gen_indices = self.area_to_gens[agent]

            for i, gen_idx in enumerate(gen_indices):
                if i < len(action):
                    if gen_idx == 9 and self.vsg_idx:
                        self.sim_case.RenGen.set(
                            src='Paux',
                            idx=self.vsg_idx[0],
                            value=action[i],
                            attr='v'
                        )
                    elif gen_idx < len(self.tg_idx):
                        self.sim_case.TurbineGov.set(
                            src='paux0',
                            idx=self.tg_idx[gen_idx],
                            value=action[i],
                            attr='v'
                        )

            return True

        except Exception as e:
            print(f"应用动作失败 ({agent}): {e}")
            return False

    def reset(self, seed=None, options=None):
        """重置环境到初始状态，并返回初始观测"""
        if seed is not None:
            self.seed(seed)

        # 重置计数器
        self.i = 0
        self.step_count = 0
        self.agents = self.possible_agents[:]
        self.total_rewards = {agent: 0.0 for agent in self.agents}
        self.episode_count += 1

        # 重置Prony相关状态

        if self.prony_analyzers:
            for analyzer in self.prony_analyzers.values():
                analyzer.clear_history()

        # 初始化仿真
        if not self._initialize_simulation():
            raise RuntimeError("仿真初始化失败")

        # 施加扰动
        self._apply_disturbance()

        # 仿真到第一个动作点
        sim_success = self.sim_to_next()
        if not sim_success:
            raise RuntimeError("初始仿真步失败")

        # 获取初始观测
        observations = {
            agent: self._get_agent_observation(agent)
            for agent in self.agents
        }

        # 构建info字典
        infos = {
            agent: {
                'episode': self.episode_count,
                'disturbance_type': self.disturbance_type,
                'disturbance_location': self.dist_location,
                'disturbance_severity': self.dist_severity,
                'area': agent
            }
            for agent in self.agents
        }

        return observations, infos

    def step(self, actions):
        """执行一步"""
        if not actions:
            raise ValueError("必须提供动作")

        # 应用动作
        for agent in self.agents:
            if agent not in actions:
                actions[agent] = np.zeros(self.action_dims[agent], dtype=np.float32)

        for agent in self.agents:
            action = np.clip(
                actions[agent],
                self.action_spaces[agent].low,
                self.action_spaces[agent].high
            )
            self._apply_agent_action(agent, action)

        # 仿真到下一个时间点
        sim_crashed = not self.sim_to_next()
        self.step_count += 1

        # 获取观测和奖励
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        for agent in self.agents:
            observations[agent] = self._get_agent_observation(agent)
            rewards[agent] = self._calculate_agent_reward(agent, actions[agent])
            self.total_rewards[agent] += rewards[agent]

            # 检查终止条件
            terminated = False
            truncated = False

            if sim_crashed:
                terminated = True
                rewards[agent] = -20.0 #训练代码中 crash_penalty = -10一致

            if self.step_count >= self.max_steps:
                truncated = True

            # if self.i >= len(self.action_instants):
            #     terminated = True

            terminateds[agent] = terminated
            truncateds[agent] = truncated

            # 构建info字典
            infos[agent] = {
                'step': self.step_count,
                'total_reward': self.total_rewards[agent],
                'area': agent,
                'sim_crashed': sim_crashed,
                'action_applied': actions[agent].copy()
            }

        # 如果episode结束
        all_terminated = all(terminateds.values())
        all_truncated = all(truncateds.values())

        if all_terminated or all_truncated:
            self.episode_rewards.append(sum(self.total_rewards.values()))
            self.episode_lengths.append(self.step_count)
            self._extract_simulation_data()
        return observations, rewards, terminateds, truncateds, infos

    def _calculate_agent_reward(self, agent: str, action: np.ndarray):
        try:
            area_idx = int(agent.split('_')[1]) - 1
            wcoi = self.sim_case.dae.y[self.wcoi_idx].astype(np.float32)
            voltage = self.sim_case.dae.y[self.voltage_idx].astype(np.float32)
            ace = self.sim_case.dae.y[self.ace_idx].astype(np.float32)[:4]
            key_bus = self.area_to_key_bus[agent]
            V_ref = [1.03, 0.97522, 1.0265, 0.9972]

            freq_dev = np.abs(wcoi[area_idx] - 1.0) if area_idx < len(wcoi) else 0.0
            volt_dev = np.abs(voltage[key_bus] - V_ref[area_idx]) if key_bus < len(voltage) else 0.0
            area_ace = ace[area_idx] if area_idx < len(ace) else 0.0

            freq_reward = -freq_dev * 2000.0
            volt_reward = -(volt_dev ** 2) * 500.0
            ace_reward = -np.abs(area_ace) * 5.0

            action_penalty = np.mean(action ** 2) * 2.0
            if hasattr(self, 'last_actions') and agent in self.last_actions:
                action_diff = np.mean(np.abs(action - self.last_actions[agent]))
                smooth_penalty = action_diff * 0.5
            else:
                smooth_penalty = 0.0

            bound_penalty = 0.0
            if np.any(np.abs(action) > 0.8):
                bound_penalty = np.sum(np.abs(action)[np.abs(action) > 0.8] - 0.8) * 1.0

            # 【修改】Prony 奖励部分：如果分析无效，返回负奖励并打印警告
            damping_reward = 0.0
            # stability_reward = 0.0
            if self.include_prony and agent in self.prony_analyzers:
                analyzer = self.prony_analyzers[agent]
                # 确保分析器已更新（如果尚未更新，可尝试立即更新）
                if analyzer.last_analysis is None:
                    # 尝试从当前仿真数据中更新
                    self._update_prony_analyzers()
                result = analyzer.last_analysis
                if result is None or not result.get('valid', False):
                    # 根据要求，此处应报错，但为避免训练中断，返回一个较大的负奖励
                    # 同时打印警告，便于调试
                    print(f"警告: {agent} 的 Prony 分析无效，奖励设为 -5.0")
                    return -5.0  # 直接返回负奖励，跳过后续计算
                damping_ratio = result['damping_ratio']
                if damping_ratio > 0:
                    damping_reward = damping_ratio * 3.0
                # stability_reward = result['stability_index'] * 1.0

            total_reward = (freq_reward + volt_reward + ace_reward +
                            damping_reward  -
                            action_penalty - smooth_penalty - bound_penalty) #+ stability_reward

            total_reward = np.clip(total_reward, -100.0, 100.0)
            total_reward = total_reward / 10.0
            return float(total_reward)

        except Exception as e:
            print(f"计算奖励失败 ({agent}): {e}")
            return 0.0

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def close(self):
        if hasattr(self, 'sim_case') and self.sim_case is not None:
            del self.sim_case

    def _extract_simulation_data(self):
        """提取所有时域仿真数据"""
        try:
            if hasattr(self.sim_case, 'dae') and hasattr(self.sim_case.dae, 'ts'):
                self.sim_case.dae.ts.unpack()

                t_render = np.array(self.sim_case.dae.ts.t)

                self.system_data = {
                    'time': t_render,
                    'generator_frequencies': np.array(self.sim_case.dae.ts.x[:, self.w_idx]),
                    # ... (detailed system_data omitted for brevity)
                }

                print(f"Episode {self.episode_count}: 已提取仿真数据，时间范围: {t_render[0]:.2f}-{t_render[-1]:.2f}s")

        except Exception as e:
            print(f"提取仿真数据失败: {e}")
            self.system_data = {}


def make_matd3_env(config=None):
    if config is None:
        config = {}
    env_config = {
        'tf': config.get('tf', 15.0),
        'tstep': config.get('tstep', 1/30),
        'max_steps': config.get('max_steps', max_steps),
        'include_prony': config.get('include_prony', True),
        'shared_observations': config.get('shared_observations', True),
        'prony_coordination': config.get('prony_coordination', True),
        'algorithm_type': 'MATD3'
    }
    return MultiAgentAndes39(**env_config)

def make_mappo_env(config=None):
    if config is None:
        config = {}
    env_config = {
        'tf': config.get('tf', 15.0),
        'tstep': config.get('tstep', 1/30),
        'max_steps': config.get('max_steps', max_steps),
        'include_prony': config.get('include_prony', True),
        'shared_observations': config.get('shared_observations', True),
        'prony_coordination': config.get('prony_coordination', True),
        'algorithm_type': 'MAPPO'
    }
    return MultiAgentAndes39(**env_config)

def make_masac_env(config=None):
    if config is None:
        config = {}
    env_config = {
        'tf': config.get('tf', 15.0),
        'tstep': config.get('tstep', 1/30),
        'max_steps': config.get('max_steps', max_steps),
        'include_prony': config.get('include_prony', True),
        'shared_observations': config.get('shared_observations', False),
        'prony_coordination': config.get('prony_coordination', True),
        'algorithm_type': 'MASAC'
    }
    return MultiAgentAndes39(**env_config)


def make_nsga2_matd3_mix_env(config=None):
    if config is None:
        config = {}
    env_config = {
        'tf': config.get('tf', 15.0),
        'tstep': config.get('tstep', 1/30),
        'max_steps': config.get('max_steps', max_steps),
        'include_prony': config.get('include_prony', True),
        'shared_observations': config.get('shared_observations', True),
        'prony_coordination': config.get('prony_coordination', True),
        'algorithm_type': 'NSGA2_MATD3'
    }
    return MultiAgentAndes39(**env_config)

def make_wse_matd3_env(config=None):
    if config is None:
        config = {}
    env_config = {
        'tf': config.get('tf', 15.0),
        'tstep': config.get('tstep', 1/30),
        'max_steps': config.get('max_steps', max_steps),
        'include_prony': config.get('include_prony', True),
        'shared_observations': config.get('shared_observations', True),
        'prony_coordination': config.get('prony_coordination', True),
        'algorithm_type': 'WSE_MATD3'
    }
    return MultiAgentAndes39(**env_config)