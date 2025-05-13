import mesa
import networkx as nx
import matplotlib.pyplot as plt
from decimal import Decimal

payoffs = {
    "A": {
        "A": (Decimal('1.0'), Decimal('1.0')),
        "B": (Decimal('-0.4'), Decimal('1.2'))
    },
    "B": {
        "A": (Decimal('1.2'), Decimal('-0.4')),
        "B": (Decimal('0.0'), Decimal('0.0'))
    }
}

#DEFINIZIONE DELL'AGENTE
class nxnImitationAgent(mesa.Agent):
    def __init__(self, unique_id, model, strategy=None):
        super().__init__(model)
        self.strategy = strategy if strategy else self.random.choice(["A", "B"])
        self.strategy_after_revision = self.strategy
        self.payoff = Decimal('0.0')

    def decide_strategy(self):
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        if not neighbors:
            self.strategy_after_revision = self.strategy
            return

        if self.model.decision_rule == "imitate_if_better":
            observed = self.random.choice(neighbors)
            if observed.payoff > self.payoff:
                self.strategy_after_revision = observed.strategy
            else:
                self.strategy_after_revision = self.strategy

        if self.model.decision_rule == "imitate_better_neighbor":
            candidates = [self] + self.model.grid.get_neighbors(self.pos, include_center=False)
            if not candidates:
                self.strategy_after_revision = self.strategy
                return
            max_payoff = max(agent.payoff for agent in candidates)
            best_agents = [agent for agent in candidates if agent.payoff == max_payoff]
            chosen = self.random.choice(best_agents)
            self.strategy_after_revision = chosen.strategy


    def step(self):
        if not self.model.first_step:
            self.decide_strategy()

    def advance(self):
        if self.strategy != self.strategy_after_revision:
            print(f"Agente {self.unique_id}: {self.strategy} → {self.strategy_after_revision}")
        self.strategy = self.strategy_after_revision


#DEFINIZIONE DEL MODELLO
class nxnImitationModel(mesa.Model):
    def __init__(self, n, decision_rule="imitate_if_better", seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        self.decision_rule = decision_rule
        self.steps = 0
        self.first_step = True

        self.agent_list = []
        self.grid = Network(n, "tree_network")

        for node in self.grid.G.nodes:
            strategy = "B" if "Nodo Centrale" in node else "A"

            agent = nxnImitationAgent(len(self.agent_list), self, strategy=strategy)
            self.grid.place_agent(agent, node)
            self.agent_list.append(agent)

        self.datacollector = mesa.DataCollector(
            agent_reporters={"Strategy": "strategy", "Payoff": "payoff"}
        )

    def compute_all_payoffs(self):
        for agent in self.agent_list:
            agent.payoff = Decimal('0.0')

        for node1, node2 in self.grid.G.edges:
            agents1 = self.grid.get_cell_list_contents([node1])
            agents2 = self.grid.get_cell_list_contents([node2])
            if agents1 and agents2:
                a1, a2 = agents1[0], agents2[0]
                p1, p2 = payoffs[a1.strategy][a2.strategy]
                a1.payoff += p1
                a2.payoff += p2

    def step(self):
        self.compute_all_payoffs()
        print(f"\nPAYOFF STEP {self.steps}:" if not self.first_step else "\nPAYOFF INIZIALI:")
        for i, agent in enumerate(self.agent_list):
            print(f"Agente {i+1}: Payoff = {agent.payoff}")

        self.datacollector.collect(self)

        for agent in self.agent_list:
            agent.step()

        for agent in self.agent_list:
            agent.advance()

        self.steps += 1
        if self.first_step:
            self.first_step = False


#DEFINIZIONE DEL NETWORK
class Network(mesa.space.NetworkGrid):
    def __init__(self, num_nodes, network_model_type):
        self.num_nodes = num_nodes
        self.network_model_type = network_model_type
        super().__init__(self.build_network())

    def build_network(self):
        return self.tree_network(self.num_nodes) if self.network_model_type == "tree_network" else None

    def tree_network(self, num_nodes):
        G = nx.Graph()
        nodo_centrale = "Nodo Centrale"
        G.add_node(nodo_centrale)

        for i in range(1, 5):
            leaf_node = f"Foglia {i} Nodo Centrale"
            G.add_node(leaf_node)
            G.add_edge(nodo_centrale, leaf_node)

        peripheral_nodes = ["Nodo Periferico 1", "Nodo Periferico 2", "Nodo Periferico 3"]
        for p_node in peripheral_nodes:
            G.add_node(p_node)
            G.add_edge(nodo_centrale, p_node)

            for i in range(1, 5):
                leaf_node = f"Foglia {i} {p_node.split()[2]}"
                G.add_node(leaf_node)
                G.add_edge(p_node, leaf_node)

        return G

#GRAFICO DEL NETWORK
def plot_network(model, ax, title="Network"):
    G = model.grid.G
    pos = nx.spring_layout(G, seed=42)

    colors = []
    for node in G.nodes:
        agents = model.grid.get_cell_list_contents([node])
        color = "green" if agents and agents[0].strategy == "A" else "red" if agents else "gray"
        colors.append(color)

    nx.draw(G, pos, node_color=colors, with_labels=False, edge_color="gray", node_size=500, ax=ax, font_size=8)
    ax.set_title(title)

#ESECUZIONE
def run_simulation(model, steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_network(model, ax1, title="Configurazione Iniziale")
    for _ in range(steps):
        model.step()
    plot_network(model, ax2, title=f"Configurazione Dopo {steps} Passi")
    plt.tight_layout()
    plt.show()

decision_rule = "imitate_if_better"
#decision_rule = "imitate_better_neighbor"
model = nxnImitationModel(15, decision_rule=decision_rule)
run_simulation(model, steps=10)
