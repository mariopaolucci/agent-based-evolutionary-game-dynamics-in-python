import mesa
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Matrice dei payoff per il gioco Carta-Forbice-Sasso
# 0 = Carta, 1 = Forbice, 2 = Sasso
payoffs = np.array([[0, -1, 1],
                    [1, 0, -1],
                    [-1, 1, 0]])

class EGD3x3ImitateModel(mesa.Model):
    """Modello con agenti che adottano strategie in stile Carta-Forbice-Sasso."""

    def __init__(self, n, width, height, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        self.running = True
        self.prob_revision = 0.1  # Probabilità di revisione della strategia

        # Crea gli agenti
        agents = EGD3x3ImitateAgent.create_agents(model=self, n=n)

        self.datacollector = mesa.DataCollector(
            agent_reporters={"Current_strategy": "strategy"},
        )

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("update_payoff")
        self.agents.shuffle_do("update_strategy")

class EGD3x3ImitateAgent(mesa.Agent):
    """Agente con strategia iniziale casuale."""

    def __init__(self, model):
        super().__init__(model)
        self.strategy = self.random.choice([0, 1, 2])  # 0 = Carta, 1 = Forbice, 2 = Sasso

    def update_payoff(self):
        others = [x for x in self.model.agents if x != self]
        if others:
            other = self.random.choice(others)
            self.payoff = payoffs[self.strategy, other.strategy]
            other.payoff = payoffs[other.strategy, self.strategy]

    def update_strategy(self):
        if self.random.random() < self.model.prob_revision:
            others = [x for x in self.model.agents if x != self]
            if others:
                other = self.random.choice(others)
                if other.payoff > self.payoff:
                    self.strategy = other.strategy

# Inizializzazione del modello
model = EGD3x3ImitateModel(n=100, width=10, height=10)

# Numero di passi temporali
steps = 1000

# Esecuzione del modello
for step in range(steps):
    model.step()

# Raccolta dei dati dal DataCollector
df_strategy = model.datacollector.get_agent_vars_dataframe().reset_index()
df_strategy = df_strategy.rename(columns={"Current_strategy": "Strategy"})

# Mappa strategia numerica → etichetta testuale
strategy_labels = {0: "Carta", 1: "Forbice", 2: "Sasso"}
df_strategy["StrategyLabel"] = df_strategy["Strategy"].map(strategy_labels)

# Grafico
plt.figure(figsize=(10, 6))
sns.histplot(data=df_strategy, x='Step', hue='StrategyLabel',
             multiple='stack', palette='Set2', discrete=True)

plt.title("Evoluzione delle Strategie degli Agenti (Carta, Forbice, Sasso)")
plt.xlabel("Passo")
plt.ylabel("Numero di Agenti")
plt.tight_layout()
plt.show()
