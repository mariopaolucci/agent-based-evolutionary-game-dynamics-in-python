import mesa
import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt

##funzioanante
# === CLASSE AGENTE ===
class Player(Agent):
    def __init__(self, unique_id, model, strategy):
        self.unique_id = unique_id
        self.model = model
        self.strategy = strategy
        self.strategy_after_revision = strategy
        self.payoff = 0

    def update_payoff(self):
        self.payoff = self.model.payoff_to_use(self)

    def update_strategy_after_revision(self):
        if random.random() < self.model.noise:
            self.strategy_after_revision = random.choice(range(self.model.n_of_strategies))
        else:
            decision_rule = self.model.decision_rule
            rule_method = f"{decision_rule}_rule"
            getattr(self, rule_method)()    

    def update_strategy(self):
        self.strategy = self.strategy_after_revision

    # === Regole imitative ===
    def imitate_if_better_rule(self):
        other = random.choice(self.model.players)
        if other.payoff > self.payoff:
            self.strategy_after_revision = other.strategy

    def imitative_pairwise_difference_rule(self):
        other = random.choice(self.model.players)
        diff = other.payoff - self.payoff
        if diff > 0 and self.model.max_payoff_difference > 0:
            prob = diff / self.model.max_payoff_difference
            if random.random() < prob:
                self.strategy_after_revision = other.strategy

    def imitative_linear_attraction_rule(self):
        other = random.choice(self.model.players)
        diff = other.payoff - self.model.min_payoff
        if self.model.max_payoff_difference > 0:
            prob = diff / self.model.max_payoff_difference
            if random.random() < prob:
                self.strategy_after_revision = other.strategy

    def imitative_linear_dissatisfaction_rule(self):
        diff = self.model.max_payoff - self.payoff
        if self.model.max_payoff_difference > 0:
            prob = diff / self.model.max_payoff_difference
            if random.random() < prob:
                other = random.choice(self.model.players)
                self.strategy_after_revision = other.strategy

    # === Regole dirette ===
    def direct_best_rule(self):
        best = max(
            [(s, self.model.expected_payoff(s) if self.model.use_expected_payoff else self.model.realized_payoff(s)) for s in range(self.model.n_of_strategies)],
            key=lambda x: x[1]
        )
        self.strategy_after_revision = best[0]

    def direct_pairwise_difference_rule(self):
        s = random.choice(range(self.model.n_of_strategies))
        new_payoff = self.model.expected_payoff(s) if self.model.use_expected_payoff else self.model.realized_payoff(s)
        diff = new_payoff - self.payoff
        if diff > 0 and self.model.max_payoff_difference > 0:
            prob = diff / self.model.max_payoff_difference
            if random.random() < prob:
                self.strategy_after_revision = s

    def direct_positive_proportional_m_rule(self):
        if self.model.use_expected_payoff:
            payoffs = [max(self.model.expected_payoff(s), 0) for s in range(self.model.n_of_strategies)]
        else:
            payoffs = [max(self.model.realized_payoff(s), 0) for s in range(self.model.n_of_strategies)]
        weights = [p ** self.model.m for p in payoffs]
        total = sum(weights)
        if total > 0:
            r = random.random() * total
            acc = 0
            for i, w in enumerate(weights):
                acc += w
                if acc >= r:
                    self.strategy_after_revision = i
                    break

    def step(self):
        self.update_payoff()
        self.update_strategy_after_revision()
        self.update_strategy()


# === CLASSE MODELLO ===
class GameModel(Model):
    def __init__(self, n_of_players, payoff_matrix, initial_distribution,
                 decision_rule, noise, use_expected_payoff=False, m=1):
        super().__init__()
        self.num_agents = n_of_players
        self.payoff_matrix = payoff_matrix
        self.n_of_strategies = len(payoff_matrix)
        self.players = []
        self.decision_rule = decision_rule
        self.noise = noise
        self.use_expected_payoff = use_expected_payoff
        self.m = m

        # Costanti di normalizzazione
        self.min_payoff = np.min(payoff_matrix)
        self.max_payoff = np.max(payoff_matrix)
        self.max_payoff_difference = self.max_payoff - self.min_payoff

        # Inizializzazione degli agenti
        agent_id = 0
        for i, count in enumerate(initial_distribution):
            for _ in range(count):
                player = Player(agent_id, self, i)
                self.players.append(player)
                agent_id += 1

        self.data_collector = DataCollector(
            agent_reporters={"strategy": "strategy", "payoff": "payoff"}
        )

    def payoff_to_use(self, player):
        return self.expected_payoff(player.strategy) if self.use_expected_payoff else self.realized_payoff_vs(player)

    def realized_payoff_vs(self, player):
        other = self.random.choice(self.players)
        return self.payoff_matrix[player.strategy][other.strategy]

    def expected_payoff(self, strategy):
        counts = [sum(1 for p in self.players if p.strategy == s) for s in range(self.n_of_strategies)]
        total = sum(counts)
        if total == 0:
            return 0
        probs = [c / total for c in counts]
        return sum(self.payoff_matrix[strategy][s] * p for s, p in enumerate(probs))

    def realized_payoff(self, strategy):
        sample = random.choice(self.players)
        return self.payoff_matrix[strategy][sample.strategy]

    def step(self):
        self.data_collector.collect(self)
        self.random.shuffle(self.players)
        for agent in self.players:
            agent.step()

## CONFRONTO TRA STRATEGIE DECSIONALI USANDO PAYOFF MEDIO
# Lista delle regole decisionali da testare
rules = [
    "imitate_if_better",
    "imitative_pairwise_difference",
    "imitative_linear_attraction",
    "imitative_linear_dissatisfaction",
    "direct_pairwise_difference",
    "direct_best",
    "direct_positive_proportional_m"
]

# Parametri del modello
n_of_players = 300
payoff_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])  # Sasso-Carta-Forbici
initial_distribution = [100, 100, 100]
noise = 0.05
m = 5
use_expected = True

# Ciclo per testare tutte le regole
for rule in rules:
    model = GameModel(
        n_of_players, payoff_matrix, initial_distribution,
        decision_rule=rule, noise=noise, use_expected_payoff=use_expected, m=m
    )
    
    history = []  # Lista per memorizzare l'evoluzione delle strategie
    
    # Esecuzione del modello per 100 step
    for step in range(100):
        model.step()
        strategies = [agent.strategy for agent in model.players]
        counts = {s: strategies.count(s) for s in range(model.n_of_strategies)}
        history.append(counts)
        
        # Stampa dei valori ogni 10 step
        if step % 10 == 0:
            print(f"Step {step}: {counts}")
    
    # Plot dei risultati per ogni regola
    plt.figure(figsize=(14, 7))
    custom_colors = ['skyblue', 'red', 'green']
    for s in range(model.n_of_strategies):
        plt.plot(
            range(100),
            [h.get(s, 0) for h in history],
            label=f"Strategia {s}",
            color=custom_colors[s % len(custom_colors)] 
        )
    
    plt.title(f"Distribuzione delle strategie con payoff medio - Regola: {rule}")
    plt.xlabel("Passi temporali")
    plt.ylabel("Numero di agenti per strategia")
    plt.legend()
    plt.grid(True)
    plt.show()

## CONFRONTO TRA STRATEGIE DECSIONALI USANDO PAYOFF CAUSALE TRA DUE AGENTI

# Lista delle regole decisionali da testare
rules = [
    "imitate_if_better",
    "imitative_pairwise_difference",
    "imitative_linear_attraction",
    "imitative_linear_dissatisfaction",
    "direct_pairwise_difference",
    "direct_best",
    "direct_positive_proportional_m"
]

# Parametri 
n_of_players = 300
payoff_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])  # Sasso-Carta-Forbici
initial_distribution = [100, 100, 100]
noise = 0.05
m = 5
use_expected = True

# Ciclo per testare tutte le regole
for rule in rules:
    model = GameModel(
        n_of_players, payoff_matrix, initial_distribution,
        decision_rule=rule, noise=noise, use_expected_payoff=False, m=m
    )
    
    history = []  
    
    # Esecuzione del modello per 100 step
    for step in range(100):
        model.step()
        strategies = [agent.strategy for agent in model.players]
        counts = {s: strategies.count(s) for s in range(model.n_of_strategies)}
        history.append(counts)
        
        # Stampa dei valori ogni 10 step
        if step % 10 == 0:
            print(f"Step {step}: {counts}")
    
    # Plot dei risultati per ogni regola
    plt.figure(figsize=(14, 7))
    custom_colors = ['skyblue', 'red', 'green']
    for s in range(model.n_of_strategies):
        plt.plot(
            range(100),
            [h.get(s, 0) for h in history],
            label=f"Strategia {s}",
            color=custom_colors[s % len(custom_colors)]  
        )
    
    plt.title(f"Distribuzione delle Strategie con payoff casuale tra due agenti - Regola: {rule}")
    plt.xlabel("Passi Temporali")
    plt.ylabel("Numero di Agenti per Strategia")
    plt.legend()
    plt.grid(True)
    plt.show()


