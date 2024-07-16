#第二版
import random
import math
from math import sqrt
import numpy as np
from src.envs.multiagentenv import MultiAgentEnv
from src.utils.dict2namedtuple import convert

import torch

import numpy as np
from scipy.optimize import minimize

class CodertaskEnv(MultiAgentEnv):

    def __init__(self,**kwargs):
        
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        self.num_users = getattr(args, "num_users", 0.0)
        self.t_max = getattr(args, "t_max", 0.0)
        # self.transmission_rates = np.random.rand(self.num_users)#随机初始化单位传输速率和单位计算速率
        # self.computation_rates = np.random.rand(self.num_users)

        self.transmission_rates = np.random.uniform(0.1,1,(self.num_users))#随机初始化单位传输速率和单位计算速率
        self.computation_rates = np.random.uniform(0.1,1,(self.num_users))
        # # 理想化
        # self.computational_tasks = np.array([0.2, 0.4, 0.6, 0.8, 1.0])  # 这个只是个例子，你得根据resnet实际的计算量和传输量来设置
        # self.transmitted_data    = np.array([1.0, 0.8, 0.6, 0.4, 0.2])  # Example values
        # self.accuracy            = np.array([1.0, 0.8, 0.6, 0.4, 0.2])  # 不同的编码深度会导致不同给的恢复精度

        # # Resnet化
        # self.computational_tasks = np.array([0.16, 0.27, 0.39, 0.50, 1.00, 0])  # 这个只是个例子，你得根据resnet实际的计算量和传输量来设置
        # # self.transmitted_data    = np.array([1.00, 0.50, 0.25, 0.13, 0.03])  # Example1 values
        # # self.transmitted_data    = np.array([1.00, 0.70, 0.53, 0.28, 0.12])  # Example2 values
        # # self.transmitted_data    = np.array([1.00, 0.88, 0.76, 0.64, 0.52])  # Example3 values 取得了不错的效果
        # self.transmitted_data    = np.array([1.00, 0.9, 0.8, 0.7, 0.6, 1.5])  # 第二大类图就采用这个比例 values
        # self.accuracy            = np.array([0.60, 0.79, 0.84, 0.87, 0.95, 1])  # 不同的编码深度会导致不同给的恢复精度


        # 第四一版
        self.computational_tasks = np.array([0.07, 0.35, 0.55, 0.73, 1.01, 0])
        self.transmitted_data    = np.array([1.15, 0.66, 0.48, 0.39, 0.29, 2.2])  # Example3 values
        self.accuracy = np.array([0.79, 0.83, 0.87, 0.91, 0.95, 1])



        self.time_factor = 0.5 # 为了同时考虑精度和时延，需要设置一个比例系数，这个数可以调节
        self.acc_factor  = 0.5
        self.bandwidth = getattr(args, "bandwidth", 0.0)

        self.n_agents = self.num_users
        self.n_actions = 6
        self.episode_limit = 1
        self.cur_timestep = 0
        self.steps = 0
        self.p = 0

    def reward_task(self, index_vector):
        if len(index_vector) != self.num_users:
            raise ValueError("Index vector length must match the number of users")
        
        # index_vector = np.random.randint(0, 5, size=(self.num_users))

        # Compute computation and transmission times for each user
        computation_time = self.computational_tasks[index_vector] / self.computation_rates
        transmission_time = self.transmitted_data[index_vector] / self.transmission_rates
        _, transmission_time = self.find_optimal_b(transmission_time)
        latency = np.average(computation_time) + transmission_time
        accuracy = np.average(self.accuracy[index_vector])

        # # 全用大模型
        # # Compute computation and transmission times for each user
        # computation_time = self.computational_tasks[4] / self.computation_rates
        # transmission_time = self.transmitted_data[4] / self.transmission_rates
        # _, transmission_time = self.find_optimal_b(transmission_time)
        # latency = np.average(computation_time) + transmission_time
        # accuracy = np.average(self.accuracy[4])

        # # 全用直接传输模型
        # # Compute computation and transmission times for each user
        # computation_time = self.computational_tasks[5] / self.computation_rates
        # transmission_time = self.transmitted_data[5] / self.transmission_rates
        # _, transmission_time = self.find_optimal_b(transmission_time)
        # latency = np.average(computation_time) + transmission_time
        # accuracy = np.average(self.accuracy[5])

        # # # 全用小模型
        # # Compute computation and transmission times for each user
        # computation_time = self.computational_tasks[0] / self.computation_rates
        # transmission_time = self.transmitted_data[0] / self.transmission_rates
        # _, transmission_time = self.find_optimal_b(transmission_time)
        # latency = np.average(computation_time) + transmission_time
        # accuracy = np.average(self.accuracy[0])

        # print(self.computational_tasks[4])
        # print(self.computation_rates)
        # print(computation_time)
        # print(self.cur_timestep) # 除以8倍
        # print(np.average(computation_time))
        # print(transmission_time)
        # print(latency * 10)
        # print(accuracy * 50)
        total_reward = -self.time_factor * latency * 15 + self.acc_factor * accuracy * 30

        return total_reward, transmission_time, np.average(computation_time), latency, accuracy #最小化时延的同时，最大化精度

    def objective(self,b, x):
        return np.average(x / b)

    def constraint(self, b):
        return np.sum(b) - self.bandwidth
    
    def find_optimal_b(self, x):
        b_initial = np.full(len(x), self.bandwidth / len(x))

        con = {'type': 'eq', 'fun': lambda b: self.constraint(b)}

        # Bounds to ensure b_i is positive
        bounds = [(0, None) for _ in x]

        # Run the optimization
        result = minimize(self.objective, b_initial, args=(x,), constraints=con, bounds=bounds, method='SLSQP')
        optimal_value = self.objective(result.x, x)
        return result.x, optimal_value

    # # 平均带宽
    # def find_optimal_b(self, x):
    #     average_bandwith = self.bandwidth/self.num_users
    #     # Run the optimization
    #     result = [average_bandwith] * self.num_users
    #     optimal_value = self.objective(result, x)
    #     return result, optimal_value

    # # 贪心带宽
    # def find_optimal_b(self, x):
    #     # 预定义的带宽资源值，从大到小排列
    #     bandwidth_resources = [1.9, 1.6, 1.2, 0.8, 0.5]
    #     # 确保x的长度与带宽资源列表长度一致
    #     if len(x) != len(bandwidth_resources):
    #         raise ValueError("The length of x must match the number of bandwidth resources")
    #     # 对x进行排序，并获取排序后的索引
    #     sorted_indices = np.argsort(x)
    #     # 创建结果列表，初始化为全零
    #     result = [0] * len(x)
    #     # 根据x的排序结果分配带宽资源
    #     for i, index in enumerate(sorted_indices):
    #         result[index] = bandwidth_resources[i]
    #     # 计算最优值
    #     optimal_value = self.objective(result, x)
    #     return result, optimal_value

    

    def step(self, actions):
        # print(actions)
        # print(test_mode)

        self.cur_timestep += 1
        # print(self.cur_timestep) # 除以8倍
        # print(self.steps)
        self.steps += 1

        # 原action
        real_action = actions[:self.num_users]
        # 新引入的test_mode 进行检验
        test_mode = actions[-1]

        #判断是否需要中止
        terminated = False
        if self.steps >= self.episode_limit and not terminated:
            terminated = True
        reward, transmission_time, computation_time, latency, accuracy = self.reward_task(real_action)

        # 环境中均值计算
        info_n = {}
        info_n["Reward"] = reward
        info_n["trans_time"] = transmission_time
        info_n["accuracy"] = accuracy
        info_n["comp_time"] = computation_time
        info_n["latency"] = latency

        if test_mode == 1:
            # print(test_mode)
            # 环境中单例打印
            # if self.cur_timestep > 123500:     #100万次的打印
            # if self.cur_timestep > 98500:    #80万次的打印
            if self.cur_timestep > 28000:   #30万次打印 原先35000次、可能20万次就足够训练好了，可以尝试32000次，更加节约，hh
            # if self.cur_timestep > 47000:     # 40万次打印
            # if self.cur_timestep > 73000:     # 60万次打印
            # if self.cur_timestep > 56000:     # 50万次打印
            # if self.cur_timestep > 56000:     # 50万次打印
                print(real_action)
                print(self.transmission_rates)
                print(self.computation_rates)
                print(computation_time)
                print(transmission_time)
                print(latency)
                print(accuracy)

        return reward, terminated, info_n

    def reset(self):
        self.steps = 0
        self.transmission_rates = np.random.uniform(0.1,1,(self.num_users))#随机初始化单位传输速率和单位计算速率
        self.computation_rates = np.random.uniform(0.1,1,(self.num_users))
        return self.get_obs()
    
    def get_obs(self):
        """ Returns all agent observations in a list """
        obs_n = []
        for i in range(self.num_users):
            obs = self.get_obs_agent(i)
            obs_n.append(obs)
        return obs_n

    def get_obs_agent(self, agent_id):

        # 方案一，直接获取全局信息
        obs = [0]*self.num_users*2
        # obs[0] = self.transmission_rates[agent_id]
        # obs[1] = self.computation_rates[agent_id]
        for i in range(self.num_users):
            obs[2*i] = self.transmission_rates[i]
            obs[2*i+1] = self.computation_rates[i]
        obs = np.array(obs)
        return obs

    def get_obs_size(self):
        return len(self.get_obs_agent(0))

    def get_state(self, team=None):
        state = np.concatenate(self.get_obs())
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        state_size = len(self.get_state())
        return state_size

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.get_total_actions()))
    
    def get_avail_agent_actions(self, agent_id):
        pass

    def get_total_actions(self):
        return self.n_actions

    def get_stats(self):
        return None

    def close(self):
        pass

    def render(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        pass
