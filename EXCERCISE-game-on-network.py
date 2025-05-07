
#IMPORTAZIONE DEI PACCHETTI NECESSARI
import matplotlib.pyplot as plt
from mesa import Model, Agent
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import random

#DEFINIZIONE DELL'AGENTE
class PlayerAgent(Agent):
    def __init__(self, unique_id, model, strategy=None):
        super().__init__(unique_id, model)
        self.strategy = strategy if strategy is not None else random.choice([0, 1])  
        self.strategy_after_revision = self.strategy         
        self.payoff = 0                                      

    def update_payoff(self):
        self.payoff = 0
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        for neighbor_id in neighbors:
            neighbor = self.model.schedule.agents[neighbor_id]
            self.payoff += self.model.payoff_matrix[self.strategy][neighbor.strategy]

    def decide_strategy(self):
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        best_neighbor = max(
            (self.model.schedule.agents[n_id] for n_id in neighbors),  # Ottieni direttamente gli agenti
            key=lambda a: a.payoff,
            default=None
        )
        if best_neighbor and best_neighbor.payoff > self.payoff:
            self.strategy_after_revision = best_neighbor.strategy

    def step(self):
        self.update_payoff()

    def advance(self):
      if not self.model.first_step:
          if random.random() < self.model.prob_revision:
              self.decide_strategy()
      self.strategy = self.strategy_after_revision

#DEFINIZIONE DEL MODELLO
class StrategyModel(Model):
    def __init__(self, num_players=20, prob_revision=1.0, initial_strategies=None):
        super().__init__()
        self.num_agents = num_players
        self.prob_revision = prob_revision
        self.first_step = True

        self.schedule = SimultaneousActivation(self)     #Creazione grafo e network
        self.G = nx.Graph()
        self.grid = NetworkGrid(self.G)

        self.payoff_matrix = [[0, 1.2], [-0.4, 1]]
        self.n_of_strategies = len(self.payoff_matrix)

        if initial_strategies is None:
            initial_strategies = [random.choice([0, 1]) for _ in range(self.num_agents)]

        for i in range(self.num_agents):
            agent = PlayerAgent(i, self, strategy=initial_strategies[i])
            self.G.add_node(i, agent=[])
            self.grid.place_agent(agent, i)
            self.schedule.add(agent)

        connections = [                                  #Definisco i collegami tra i nodi
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
            (1, 8), (1, 9), (1, 10), (1, 11),
            (2, 12), (2, 13), (2, 14), (2, 15),
            (3, 16), (3, 17), (3, 18), (3, 19)
        ]
        self.G.add_edges_from(connections)

        self.datacollector = DataCollector(
            model_reporters={
                "Strategy_0": lambda m: self.count_strategy(0),
                "Strategy_1": lambda m: self.count_strategy(1)
            },
            agent_reporters={"Strategy": "strategy"}
        )

        self.running = True

    def count_strategy(self, strategy):                                                              #Calcola proporzione agenti con xStrategia
        return sum(1 for a in self.schedule.agents if a.strategy == strategy) / self.num_agents

    def step(self):
      self.schedule.step()
      self.datacollector.collect(self)
      if self.first_step:
          self.first_step = False

#VISUALIZZAZIONE GRAFICA
def plot_network(model):
    G = model.G
    pos = nx.spring_layout(G, seed=42)
    colors = []

    for node in G.nodes:
        agent = G.nodes[node]["agent"]
        strategy = agent.strategy if isinstance(agent, PlayerAgent) else agent[0].strategy
        color = "green" if strategy == 0 else "red"
        colors.append(color)

    fig = plt.figure(figsize=(6, 4))
      
    nx.draw(G, pos, with_labels=False, node_color=colors, node_size=500, edge_color='gray')
    plt.title("Network di partenza", color='white')
    plt.show()


def run_simulation(model, steps):
    for _ in range(steps):
        model.step()

    model_data = model.datacollector.get_model_vars_dataframe()
    return model_data["Strategy_0"].values, model_data["Strategy_1"].values

#SIMULAZIONE
initial_strategies = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
model = StrategyModel(num_players=20, prob_revision=1.0, initial_strategies=initial_strategies)
plot_network(model)

steps = 1000
strategy_0, strategy_1 = run_simulation(model, steps)
plot_network(model)

