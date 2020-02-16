import numpy as np
from numpy import random as rnd
import networkx as nx
from enum import Enum, auto


class Agent:
    def __init__(self):
        self.state: HealthState = HealthState.SUSCEPTIBLE
        self.next_state: HealthState = HealthState.SUSCEPTIBLE
        self.location: int = None
        self.next_location: int = None


class HealthState(Enum):
    SUSCEPTIBLE = auto()
    INFECTED = auto()
    RECOVERD = auto()


class Simulation:
    def __init__(
        self,
        num_agent: int = 1000,
        num_island: int = 20,
        num_initial_infected_agent: int = 1,
        average_degree: int = 8,
        beta: float = 0.833,
        gamma: float = 0.33,
        m: float = 0.2,
    ):

        self.agents = [Agent() for _ in range(num_agent)]
        self.topology = nx.barabasi_albert_graph(num_island, average_degree // 2)
        self.num_initial_infected_agent = num_initial_infected_agent
        self.beta = beta
        self.gamma = gamma
        self.m = m

    def _initialize_agents(self):
        # 初期感染者を選択し、病気に感染させる
        initial_infected_agent = rnd.choice(self.agents, size=self.num_initial_infected_agent)
        for agent in initial_infected_agent:
            agent.state = HealthState.INFECTED
            agent.next_state = HealthState.INFECTED

        # それぞれの島に同数のエージェントを配置する
        initial_island_population = len(self.agents) // len(self.topology.nodes)
        for island_id in self.topology.nodes:
            start = initial_island_population * island_id
            end = initial_island_population * (island_id + 1)

            for agent in self.agents[start:end]:
                agent.location = island_id
                agent.next_location = island_id

    def _run_epidemics(self):
        for t in range(10000):
            # それぞれの島の人口と感染者比率を計算
            infected_fraction = np.zeros(len(self.topology.nodes))
            island_population = np.zeros(len(self.topology.nodes))
            for agent in self.agents:
                island_population[agent.location] += 1
                if agent.state == HealthState.INFECTED:
                    infected_fraction[agent.location] += 1
            infected_fraction /= island_population

            # 感染者がいなければ島ごとに累積感染者比率を計算して終了する
            if np.sum(infected_fraction) == 0:
                FES = np.zeros(len(self.topology.nodes))  # FES: Final Epidemic Size
                for agent in self.agents:
                    if agent.state == HealthState.RECOVERD:
                        FES[agent.location] += 1

                FES /= island_population
                print(f"Finished calculation at time {t} with Final Epidemic Size: {FES}")
                break

            # 状態遷移と移動
            rand_nums = rnd.random(len(self.agents))
            infection_probabilitis = self.beta * infected_fraction
            migration_probabilities = self.m * infected_fraction
            destinations = [list(self.topology[island_id]) for island_id in self.topology.nodes]  # それぞれの島が隣接する島
            willingnesses = np.array([[1-fraction for i, fraction in enumerate(infected_fraction) if i in destinations[island_id]] for island_id in self.topology.nodes])  # 感染者が少ない島に移動しやすいように移動しやすさを計算
            willingnesses = [w/np.sum(w) for w in willingnesses]  # 重み付きサンプリング用に規格化

            for agent_id, agent in enumerate(self.agents):
                # 未感染者
                if agent.state == HealthState.SUSCEPTIBLE:
                    # 感染
                    if rand_nums[agent_id] < infection_probabilitis[agent.location]:
                        agent.next_state = HealthState.INFECTED
                    # 移動
                    if rand_nums[agent_id] < migration_probabilities[agent.location]:
                        agent.next_location = rnd.choice(destinations[agent.location], p=willingnesses[agent.location])
                # 感染者
                elif agent.state == HealthState.INFECTED and rand_nums[agent_id] < self.gamma:
                    agent.next_state = HealthState.RECOVERD
                else:
                    agent.next_state = agent.state
                    agent.next_location = agent.location

            # 健康状態と位置をシンクロ更新
            for agent in self.agents:
                agent.state, agent.location = agent.next_state, agent.next_location

    def run(self):
        self._initialize_agents()
        self._run_epidemics()


num_agent = 10000
num_island = 40
average_degree = 4
num_initial_infected_agent = 1
beta = 0.833
gamma = 0.33
m = 0.1

simulation = Simulation(
    num_agent=num_agent,
    num_island=num_island,
    average_degree=average_degree,
    num_initial_infected_agent=num_initial_infected_agent,
    m=m,
)
simulation.run()
