import mesa
import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt


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
        other = random.choice(self.model.players) #Seleziona a caso un agente del modello.
        if other.payoff > self.payoff:
            self.strategy_after_revision = other.strategy

    def imitative_pairwise_difference_rule(self):
        other = random.choice(self.model.players)
        diff = other.payoff - self.payoff #variabile per calcolare la differenza di payoff tra l’altro agente e se stesso.
        if diff > 0 and self.model.max_payoff_difference > 0:  #L’agente considera solo l’altro se ha avuto un payoff migliore.
            prob = diff / self.model.max_payoff_difference #Calcola la probabilità di imitazione in modo proporzionale alla differenza

            if random.random() < prob: #Estrae un numero casuale tra 0 e 1. Se è inferiore alla probabilità calcolata, copia la strategia dell’altro agente.
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
        self.update_payoff() #agente calcola il proprio payoff attuale
        self.update_strategy_after_revision() #decide la strategia che adotterebbe al prossimo turno
        self.update_strategy() #aggiorna la propria strategia


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

# CONFRONTO TRA STRATEGIE DECISIONALI USANDO PAYOFF MEDIO - 300 PLAYERS 

rules = [
    "imitate_if_better",
    "imitative_pairwise_difference",
    "imitative_linear_attraction",
    "imitative_linear_dissatisfaction",
    "direct_pairwise_difference",
    "direct_best",
    "direct_positive_proportional_m"
]

# Colori personalizzati
custom_colors = ['skyblue', 'red', 'green']

# Parametri del modello
n_of_players = 300
payoff_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])  # Sasso-Carta-Forbici
initial_distribution = [100, 100, 100]
noise = 0.05
m = 5
use_expected = True

# Imposta la griglia per i subplot
n_rules = len(rules)
cols = 2  # puoi aumentare per fare figure più larghe
rows = int(np.ceil(n_rules / cols))
fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows))
axs = axs.flatten()


# Ciclo per ogni regola
for idx, rule in enumerate(rules):
    model = GameModel(
        n_of_players, payoff_matrix, initial_distribution,
        decision_rule=rule, noise=noise, use_expected_payoff=use_expected, m=m
    )

    history = []

    for step in range(100):
        model.step()
        strategies = [agent.strategy for agent in model.players]
        counts = {s: strategies.count(s) for s in range(model.n_of_strategies)}
        history.append(counts)

    # Line plot nel subplot corrispondente
    ax = axs[idx]
    for s in range(model.n_of_strategies):
        ax.plot(
            range(100),
            [h.get(s, 0) for h in history],
            label=f"Strategia {s}",
            color=custom_colors[s % len(custom_colors)]
        )
    ax.set_title(f"Regola: {rule}", fontsize=10)
    ax.set_xlabel("Passi temporali")
    ax.set_ylabel("Numero di agenti")
    ax.set_xticks(np.arange(0, 101, 20))
    ax.grid(True)
    ax.legend(fontsize=8)

# Rimuovi subplot vuoti se presenti
for i in range(len(rules), len(axs)):
    fig.delaxes(axs[i])

# Titolo generale
plt.suptitle("Distribuzione delle strategie (payoff medio) per ogni regola", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

## GRAFICO RISULTATI FINALI DIVISI PER STRATEGIA CON PAYOFF MEDIO 300 AGENTI
# Parametri
n_of_players = 300
payoff_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])  # Sasso-Carta-Forbici
initial_distribution = [100, 100, 100]
noise = 0.05
m = 5
use_expected_payoff = True

# Imposta griglia dinamica in base al numero di regole
n_rules = len(rules)
cols = 3
rows = int(np.ceil(n_rules / cols))

# Crea la figura con sottoplot
fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axs = axs.flatten()

# Ciclo per ogni regola e subplot
for idx, rule in enumerate(rules):
    model = GameModel(
        n_of_players, payoff_matrix, initial_distribution,
        decision_rule=rule, noise=noise, use_expected_payoff=use_expected, m=m
    )

    history = []
    for step in range(100):
        model.step()
        strategies = [agent.strategy for agent in model.players]
        counts = {s: strategies.count(s) for s in range(model.n_of_strategies)}
        history.append(counts)

    # Dati finali
    final_counts = history[-1]
    strategies = list(final_counts.keys())
    values = list(final_counts.values())
 
    ax = axs[idx]
    ax.bar(strategies, values, color=custom_colors)
    ax.set_title(f"{rule}", fontsize=10)
    ax.set_xticks(strategies)
    ax.set_xticklabels([f"Strategia {s}" for s in strategies])
    ax.set_ylim(0, n_of_players)
    ax.grid(True, axis='y')

# Disattiva eventuali subplot vuoti
for i in range(len(rules), len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.suptitle("Distribuzione finale delle strategie per ogni regola decisionale con payoff medio", fontsize=16, y=1.02)
plt.show()


## CONFRONTO TRA STRATEGIE DECSIONALI USANDO PAYOFF MEDIO - 600 players - 500 step

# Parametri del modello
n_of_players = 600
payoff_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])  # Sasso-Carta-Forbici
initial_distribution = [200, 200, 200]
noise = 0.05
m = 5
use_expected = True

# Imposta la griglia per i subplot
n_rules = len(rules)
cols = 2  # puoi aumentare per fare figure più larghe
rows = int(np.ceil(n_rules / cols))
fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows))
axs = axs.flatten()


# Ciclo per ogni regola
for idx, rule in enumerate(rules):
    model = GameModel(
        n_of_players, payoff_matrix, initial_distribution,
        decision_rule=rule, noise=noise, use_expected_payoff=use_expected, m=m
    )

    history = []

    for step in range(100):
        model.step()
        strategies = [agent.strategy for agent in model.players]
        counts = {s: strategies.count(s) for s in range(model.n_of_strategies)}
        history.append(counts)

    # Line plot nel subplot corrispondente
    ax = axs[idx]
    for s in range(model.n_of_strategies):
        ax.plot(
            range(100),
            [h.get(s, 0) for h in history],
            label=f"Strategia {s}",
            color=custom_colors[s % len(custom_colors)]
        )
    ax.set_title(f"Regola: {rule}", fontsize=10)
    ax.set_xlabel("Passi temporali")
    ax.set_ylabel("Numero di agenti")
    ax.set_xticks(np.arange(0, 101, 20))
    ax.grid(True)
    ax.legend(fontsize=8)

# Rimuovi subplot vuoti se presenti
for i in range(len(rules), len(axs)):
    fig.delaxes(axs[i])

# Titolo generale
plt.suptitle("Distribuzione delle strategie (payoff medio) per ogni regola", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


##GRAFICO FINALE PAYOFF MEDIO 600 AGENTI
# Imposta griglia dinamica in base al numero di regole
n_rules = len(rules)
cols = 3
rows = int(np.ceil(n_rules / cols))

# Crea la figura con sottoplot
fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axs = axs.flatten()

# Ciclo per ogni regola e subplot
for idx, rule in enumerate(rules):
    model = GameModel(
        n_of_players, payoff_matrix, initial_distribution,
        decision_rule=rule, noise=noise, use_expected_payoff=use_expected, m=m
    )

    history = []
    for step in range(500):
        model.step()
        strategies = [agent.strategy for agent in model.players]
        counts = {s: strategies.count(s) for s in range(model.n_of_strategies)}
        history.append(counts)

    # Dati finali
    final_counts = history[-1]
    strategies = list(final_counts.keys())
    values = list(final_counts.values())
 
    ax = axs[idx]
    ax.bar(strategies, values, color=custom_colors)
    ax.set_title(f"{rule}", fontsize=10)
    ax.set_xticks(strategies)
    ax.set_xticklabels([f"Strategia {s}" for s in strategies])
    ax.set_ylim(0, n_of_players)
    ax.grid(True, axis='y')

# Disattiva eventuali subplot vuoti
for i in range(len(rules), len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.suptitle("Distribuzione finale delle strategie per ogni regola decisionale con payoff medio", fontsize=16, y=1.02)
plt.show()


## CONFRONTO TRA STRATEGIE DECSIONALI USANDO PAYOFF CAUSALE TRA DUE AGENTI  - 300 players 

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
use_expected_payoff = False

# Imposta la griglia per i subplot
n_rules = len(rules)
cols = 2  # puoi aumentare per fare figure più larghe
rows = int(np.ceil(n_rules / cols))
fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows))
axs = axs.flatten()


# Ciclo per ogni regola
for idx, rule in enumerate(rules):
    model = GameModel(
        n_of_players, payoff_matrix, initial_distribution,
        decision_rule=rule, noise=noise, use_expected_payoff=use_expected, m=m
    )

    history = []

    for step in range(100):
        model.step()
        strategies = [agent.strategy for agent in model.players]
        counts = {s: strategies.count(s) for s in range(model.n_of_strategies)}
        history.append(counts)

    # Line plot nel subplot corrispondente
    ax = axs[idx]
    for s in range(model.n_of_strategies):
        ax.plot(
            range(100),
            [h.get(s, 0) for h in history],
            label=f"Strategia {s}",
            color=custom_colors[s % len(custom_colors)]
        )
    ax.set_title(f"Regola: {rule}", fontsize=10)
    ax.set_xlabel("Passi temporali")
    ax.set_ylabel("Numero di agenti")
    ax.set_xticks(np.arange(0, 101, 20))
    ax.grid(True)
    ax.legend(fontsize=8)

# Rimuovi subplot vuoti se presenti
for i in range(len(rules), len(axs)):
    fig.delaxes(axs[i])

# Titolo generale
plt.suptitle("Distribuzione delle strategie (payoff causale) per ogni regola", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

##GRAFICO FINALE PAYOFF CAUSALE 300 AGENTI

# Imposta griglia dinamica in base al numero di regole
n_rules = len(rules)
cols = 3
rows = int(np.ceil(n_rules / cols))

# Crea la figura con sottoplot
fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axs = axs.flatten()

# Ciclo per ogni regola e subplot
for idx, rule in enumerate(rules):
    model = GameModel(
        n_of_players, payoff_matrix, initial_distribution,
        decision_rule=rule, noise=noise, use_expected_payoff=use_expected, m=m
    )

    history = []
    for step in range(100):
        model.step()
        strategies = [agent.strategy for agent in model.players]
        counts = {s: strategies.count(s) for s in range(model.n_of_strategies)}
        history.append(counts)

    # Dati finali
    final_counts = history[-1]
    strategies = list(final_counts.keys())
    values = list(final_counts.values())
    custom_colors = ['skyblue', 'red', 'green']

    ax = axs[idx]
    ax.bar(strategies, values, color=custom_colors)
    ax.set_title(f"{rule}", fontsize=10)
    ax.set_xticks(strategies)
    ax.set_xticklabels([f"Strategia {s}" for s in strategies])
    ax.set_ylim(0, n_of_players)
    ax.grid(True, axis='y')

# Disattiva eventuali subplot vuoti
for i in range(len(rules), len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.suptitle("Distribuzione finale delle strategie per ogni regola decisionale con payoff causale", fontsize=16, y=1.02)
plt.show()


## CONFRONTO TRA STRATEGIE DECSIONALI USANDO PAYOFF CAUSALE TRA DUE AGENTI  - 600 players - 500 step

# Parametri del modello
n_of_players = 600
payoff_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])  # Sasso-Carta-Forbici
initial_distribution = [200, 200, 200]
noise = 0.05
m = 5
use_expected = False

# Imposta la griglia per i subplot
n_rules = len(rules)
cols = 2  # puoi aumentare per fare figure più larghe
rows = int(np.ceil(n_rules / cols))
fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows))
axs = axs.flatten()


# Ciclo per ogni regola
for idx, rule in enumerate(rules):
    model = GameModel(
        n_of_players, payoff_matrix, initial_distribution,
        decision_rule=rule, noise=noise, use_expected_payoff=use_expected, m=m
    )

    history = []

    for step in range(500):
        model.step()
        strategies = [agent.strategy for agent in model.players]
        counts = {s: strategies.count(s) for s in range(model.n_of_strategies)}
        history.append(counts)

    # Line plot nel subplot corrispondente
    ax = axs[idx]
    for s in range(model.n_of_strategies):
        ax.plot(
            range(500),
            [h.get(s, 0) for h in history],
            label=f"Strategia {s}",
            color=custom_colors[s % len(custom_colors)]
        )
    ax.set_title(f"Regola: {rule}", fontsize=10)
    ax.set_xlabel("Passi temporali")
    ax.set_ylabel("Numero di agenti")
    ax.set_xticks(np.arange(0, 501, 100))
    ax.grid(True)
    ax.legend(fontsize=8)

# Rimuovi subplot vuoti se presenti
for i in range(len(rules), len(axs)):
    fig.delaxes(axs[i])

# Titolo generale
plt.suptitle("Distribuzione delle strategie (payoff causale) per ogni regola", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

##GRAFICO FINALE PAYOFF CAUSALE 600 AGENTI

# Imposta griglia dinamica in base al numero di regole
n_rules = len(rules)
cols = 3
rows = int(np.ceil(n_rules / cols))

# Crea la figura con sottoplot
fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axs = axs.flatten()

# Ciclo per ogni regola e subplot
for idx, rule in enumerate(rules):
    model = GameModel(
        n_of_players, payoff_matrix, initial_distribution,
        decision_rule=rule, noise=noise, use_expected_payoff=use_expected, m=m
    )

    history = []
    for step in range(500):
        model.step()
        strategies = [agent.strategy for agent in model.players]
        counts = {s: strategies.count(s) for s in range(model.n_of_strategies)}
        history.append(counts)

    # Dati finali
    final_counts = history[-1]
    strategies = list(final_counts.keys())
    values = list(final_counts.values())
    custom_colors = ['skyblue', 'red', 'green']

    ax = axs[idx]
    ax.bar(strategies, values, color=custom_colors)
    ax.set_title(f"{rule}", fontsize=10)
    ax.set_xticks(strategies)
    ax.set_xticklabels([f"Strategia {s}" for s in strategies])
    ax.set_ylim(0, n_of_players)
    ax.grid(True, axis='y')

# Disattiva eventuali subplot vuoti
for i in range(len(rules), len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.suptitle("Distribuzione finale delle strategie per ogni regola decisionale con payoff causale", fontsize=16, y=1.02)
plt.show()
