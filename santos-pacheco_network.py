import networkx as nx
import numpy as np
import random
from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector


############## Player Agent Class ############
class PlayerAgent(Agent):
    def __init__(self, model, strategy):
        super().__init__(model)
        self.strategy = strategy
        self.strategy_after_revision = strategy
        self.payoff = 0

    def calculate_payoff(self):
        neighbors = self.model.graph.neighbors(self.pos)
        
        agents = []
        for n in neighbors:
            contents = self.model.grid.get_cell_list_contents([n])
            if contents:
                agents.append(contents[0])

        if not agents:
            self.payoff = 0
            return

        if self.model.play_with == "one-random-nbr":
            partner = self.random.choice(agents)
            self.payoff = self.model.payoff_matrix[self.strategy][partner.strategy]
        else:
            counts = [0] * self.model.n_strategies
            for a in agents:
                counts[a.strategy] += 1
            my_row = self.model.payoff_matrix[self.strategy]
            total = sum(my_row[i] * counts[i] for i in range(self.model.n_strategies))
            self.payoff = total if self.model.play_with.endswith("TOTAL-payoff") else total / len(agents)

    def decide_strategy(self):
        neighbors = self.model.graph.neighbors(self.pos)

        agents = []
        for n in neighbors:
            contents = self.model.grid.get_cell_list_contents([n])
            if contents:
                agents.append(contents[0])

        if not agents:
            self.strategy_after_revision = self.strategy
            return

        if self.random.random() < self.model.noise:
            self.strategy_after_revision = self.random.randint(0, self.model.n_strategies - 1)
            return

        dr = self.model.decision_rule
        if dr == "best-neighbor":
            best = max(agents + [self], key=lambda a: a.payoff)
            self.strategy_after_revision = best.strategy
        elif dr == "imitate-if-better":
            n = self.random.choice(agents)
            self.strategy_after_revision = n.strategy if n.payoff > self.payoff else self.strategy
        elif dr == "imitative-pairwise-difference":
            n = self.random.choice(agents)
            diff = n.payoff - self.payoff
            prob = max(0, diff / self.model.max_payoff_difference)
            self.strategy_after_revision = n.strategy if self.random.random() < prob else self.strategy
        elif dr == "imitative-positive-proportional-m":
            weights = [(a.payoff ** self.model.m) for a in agents + [self]]
            chosen = self.random.choices(agents + [self], weights=weights, k=1)[0]
            self.strategy_after_revision = chosen.strategy
        elif dr == "Fermi-m":
            n = self.random.choice(agents)
            diff = n.payoff - self.payoff
            prob = 1 / (1 + np.exp(-self.model.m * diff))
            self.strategy_after_revision = n.strategy if self.random.random() < prob else self.strategy
        elif dr == "Santos-Pacheco":
            n = self.random.choice(agents)
            diff = n.payoff - self.payoff
            ki = len(list(neighbors))
            kj = len(list(self.model.grid.get_neighbors(n.pos, include_center=False)))
            denom = self.model.max_payoff_diff_matrix * max(ki, kj)
            prob = max(0, diff / denom)
            self.strategy_after_revision = n.strategy if self.random.random() < prob else self.strategy
        else:
            self.strategy_after_revision = self.strategy

    def update_strategy(self):
        self.strategy = self.strategy_after_revision


############# Game Model Class ############
class GameModel(Model):
    def __init__(self, n_players, strategies, payoff_matrix,
                 play_with="all-nbrs-TOTAL-payoff", decision_rule="best-neighbor",
                 m=0.1, noise=0.0, seed=None):
        super().__init__(seed=seed)
        self.strategies = strategies
        self.n_strategies = len(strategies)
        self.payoff_matrix = payoff_matrix
        self.play_with = play_with
        self.decision_rule = decision_rule
        self.noise = noise
        self.m = m

        self.graph = nx.barabasi_albert_graph(n_players, 2)
        self.grid = NetworkGrid(self.graph)

        pool = []
        for i, count in enumerate(strategies):
            pool.extend([i] * count)
        self.random.shuffle(pool)

        for i, node in enumerate(self.graph.nodes()):
            s = pool[i % len(pool)]
            agent = PlayerAgent(self, s)
            self.grid.place_agent(agent, node)

        self.max_payoff_diff_matrix = np.max(payoff_matrix) - np.min(payoff_matrix)
        self.max_n_neighbors = max(len(list(self.graph.neighbors(n))) for n in self.graph.nodes)
        self.max_payoff_difference = self.max_payoff_diff_matrix * (self.max_n_neighbors if "TOTAL" in play_with else 1)

        self.datacollector = DataCollector(
            model_reporters={
                "strategy_0": lambda m: m.strategy_distribution().get(0, 0),
                "strategy_1": lambda m: m.strategy_distribution().get(1, 0),
                "avg_clustering": lambda m: nx.average_clustering(m.graph),
                "largest_component": lambda m: len(max(nx.connected_components(m.graph), key=len)),
            }
        )

    def strategy_distribution(self):
        dist = {i: 0 for i in range(self.n_strategies)}
        for agent in self.agents:
            dist[agent.strategy] += 1
        total = len(self.agents)
        return {k: v / total for k, v in dist.items()}

    def step(self):
        self.agents.do("calculate_payoff")
        self.agents.do("decide_strategy")
        self.agents.do("update_strategy")
        self.datacollector.collect(self)


########## Main Function to Run the Model ##########
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    import matplotlib.patches as mpatches


    model = GameModel(
        n_players=1000,
        strategies=[500, 500],
        payoff_matrix=[[0, 1.1875], [-0.1875, 1]],
        play_with="all-nbrs-TOTAL-payoff",
        decision_rule="Santos-Pacheco",
        m=0.5,
        noise=0.0,
        seed=42
    )

    for _ in range(10000):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()


################# Plotting Results ####################

    plt.figure(figsize=(12, 6))
    plt.stackplot(
        df.index,
        df["strategy_1"],
        df["strategy_0"],
        labels=["Cooperation (1)", "Defection (0)"],
        colors=["green", "orange"]
    )
    plt.legend(loc="upper right")
    plt.title("Strategy Distribution Over Time")
    plt.ylabel("Proportion of Population")
    plt.xlabel("Time Step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


########### Final Network Visualization ##########
#  plt.figure(figsize=(10, 8))
#   pos = nx.spring_layout(model.graph, seed=42)

#   node_colors = []
#   node_sizes = []

#   for node in model.graph.nodes():
#       agent = model.grid.get_cell_list_contents([node])[0]
#       node_colors.append("red" if agent.strategy == 0 else "blue")
#       node_sizes.append(50 + 200 * agent.payoff)

#   nx.draw_networkx_edges(model.graph, pos, alpha=0.2)
#   nx.draw_networkx_nodes(model.graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)

#    legend_handles = [
#      mpatches.Patch(color="red", label="Strategy 0"),
#      mpatches.Patch(color="blue", label="Strategy 1")
    
#    plt.legend(handles=legend_handles, loc="upper right")
#    plt.title("Final Network: Node Color = Strategy, Size = Payoff")
#    plt.axis("off")
#    plt.tight_layout()
#    plt.show()