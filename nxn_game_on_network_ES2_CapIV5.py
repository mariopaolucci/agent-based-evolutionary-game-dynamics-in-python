import mesa
import networkx as nx
from decimal import Decimal
from random import shuffle

# PAYOFFS
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
        """Decide if to revise the strategy based on neighbors."""
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)

        shuffle(neighbors)

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
            candidates = [self] + neighbors
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
        self.strategy = self.strategy_after_revision

#DEFINIZIONE DEL MODELLO
class nxnImitationModel(mesa.Model):
    def __init__(self, n, decision_rule="imitate_better_neighbor", seed=None):
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
        for agent in self.agent_list:
            agent.step()
        for agent in self.agent_list:
            agent.advance()
        self.steps += 1
        if self.first_step:
            self.first_step = False

    def get_cooperator_count(self):
        return sum(1 for agent in self.agent_list if agent.strategy == "A")

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
        G.add_node("Nodo Centrale")
        for i in range(1, 5):
            G.add_node(f"Foglia {i} Nodo Centrale")
            G.add_edge("Nodo Centrale", f"Foglia {i} Nodo Centrale")

        for j in range(1, 4):
            p_node = f"Nodo Periferico {j}"
            G.add_node(p_node)
            G.add_edge("Nodo Centrale", p_node)
            for i in range(1, 5):
                leaf_node = f"Foglia {i} {j}"
                G.add_node(leaf_node)
                G.add_edge(p_node, leaf_node)
        return G

#FUNZIONE PER SIMULAZIONI MULTIPLE
def simulate_and_calculate_probabilities(num_simulations=100, steps=5):
    counts = {20: 0, 11: 0, 6: 0, 0: 0}

    for _ in range(num_simulations):
        model = nxnImitationModel(15, decision_rule="imitate_better_neighbor")
        for _ in range(steps):
            model.step()

        cooperators = model.get_cooperator_count()
        if cooperators in counts:
            counts[cooperators] += 1

    total = num_simulations
    for key in counts:
        percent = (counts[key] / total) * 100
        print(f"Probabilità che i cooperatori siano {key}: {percent:.2f} %")

from collections import Counter

def simulate_and_analyze(num_simulations=100, steps=5):
    results = []

    for _ in range(num_simulations):
        model = nxnImitationModel(15, decision_rule="imitate_better_neighbor")
        for _ in range(steps):
            model.step()

        cooperators = sum(1 for agent in model.agent_list if agent.strategy == "A")
        results.append(cooperators)

    freq = Counter(results)

    print("\nRisultati su", num_simulations, "simulazioni:")
    for x in sorted(freq.keys()):
        percentage = (freq[x] / num_simulations) * 100
        print(f"Collaboratori = {x:2d} → {percentage:.2f}%")

    return freq

#ESECUZIONE
simulate_and_analyze(num_simulations=100, steps=5)