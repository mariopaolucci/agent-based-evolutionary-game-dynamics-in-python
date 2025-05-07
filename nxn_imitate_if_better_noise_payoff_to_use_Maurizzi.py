import numpy as np
import random
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.datacollection import DataCollector

class Player(Agent):

    def __init__(self, unique_id, strategy, model):
        self.unique_id = unique_id  # Identificativo univoco dell'agente
        self.model = model
        self.strategy = strategy  # Strategia attuale
        self.strategy_after_revision = strategy  # Strategia dopo la revisione
        self.payoff = 0  # Guadagno attuale

    def update_payoff(self):
        # Aggiorna il guadagno dell'agente in base alla strategia
        self.payoff = self.model.payoff_to_use(self)

    def update_strategy_after_revision(self):
        # Con una certa probabilità (rumore), cambia strategia casualmente
        if random.random() < self.model.noise:
            self.strategy_after_revision = random.randint(0, self.model.n_of_strategies - 1)
        else:
            # Osserva un altro agente (diverso da sé stesso)
            others = [p for p in self.model.players if p.unique_id != self.unique_id]
            if others:  # Controlla che esistano altri agenti
                observed_player = self.model.random.choice(others)
                if observed_player.payoff > self.payoff:
                    self.strategy_after_revision = observed_player.strategy
    
    def update_strategy(self):
        # Aggiorna la strategia attuale con quella scelta dopo la revisione
        self.strategy = self.strategy_after_revision


class GameModel(Model):

    def __init__(self, n_of_players, payoff_matrix, initial_distribution, noise, payoff_method):
        super().__init__()
        self.num_agents = n_of_players  # Numero di agenti nel modello
        self.payoff_matrix = payoff_matrix  # Matrice dei payoff
        self.n_of_strategies = len(payoff_matrix)  # Numero di strategie possibili
        self.players = []  # Lista dei giocatori
        self.noise = noise  # Livello di rumore
        self.payoff_method = payoff_method  # Metodo di calcolo del guadagno: "expected" o "random"
        self.current_id = 0  # ID iniziale per gli agenti

        self.setup_players(initial_distribution)
        self.data_collector = DataCollector(
            agent_reporters={"strategy": "strategy", "payoff": "payoff"}
        )
        self.strategy_history = []  # Storico della distribuzione delle strategie a ogni passo

    def next_id(self):
        """Restituisce un ID unico per il prossimo agente."""
        self.current_id += 1
        return self.current_id - 1

    def setup_players(self, initial_distribution):
        # Verifica che la distribuzione iniziale sia coerente con il numero di strategie
        if len(initial_distribution) != self.n_of_strategies:
            raise ValueError("La lunghezza della distribuzione iniziale deve corrispondere al numero di strategie.")

        # Crea gli agenti con le strategie iniziali
        for i, count in enumerate(initial_distribution):
            for _ in range(count):
                player = Player(self.next_id(), i, self)
                self.players.append(player)

    def payoff_to_use(self, player):
        # Seleziona il metodo di calcolo del guadagno
        if self.payoff_method == "expected":
            return self.expected_payoff(player.strategy)
        elif self.payoff_method == "random":
            return self.realized_payoff(player)
        else:
            raise ValueError(f"Metodo di guadagno non valido: {self.payoff_method}")

    def expected_payoff(self, strategy):
        # Calcola il guadagno atteso della strategia rispetto alla distribuzione attuale
        counts = [sum(1 for p in self.players if p.strategy == s) for s in range(self.n_of_strategies)]
        total = sum(counts)
        if total == 0:
            return 0
        probs = [c / total for c in counts]
        return sum(self.payoff_matrix[strategy][s] * p for s, p in enumerate(probs))

    def realized_payoff(self, player):
        # Calcola un guadagno realistico simulando un'interazione casuale con un altro giocatore
        others = [p for p in self.players if p.unique_id != player.unique_id]
        if not others:
            return 0
        other = self.random.choice(others)
        return self.payoff_matrix[player.strategy][other.strategy]

    def step(self):
        # Fase 1: aggiornamento dei guadagni
        for player in self.players:
            player.update_payoff()

        # Fase 2: revisione delle strategie
        for player in self.players:
            player.update_strategy_after_revision()

        # Fase 3: adozione delle nuove strategie
        for player in self.players:
            player.update_strategy()

        # Calcolo dei guadagni attesi per ogni strategia (per analisi interna)
        self.update_strategy_expected_payoffs()

        # Raccolta dati con DataCollector (per ogni agente)
        self.data_collector.collect(self)
        
        # Calcola la distribuzione attuale delle strategie
        strategy_counts = [0] * self.n_of_strategies
        for player in self.players:
            strategy_counts[player.strategy] += 1

        # Calcola e salva le frequenze relative
        strategy_frequencies = [count / self.num_agents for count in strategy_counts]
        self.strategy_history.append(strategy_frequencies)

        # Stampa le frequenze delle strategie in percentuale
        percentages = [f"{freq * 100:.2f}%" for freq in strategy_frequencies]
        print(f"Passo {len(self.strategy_history)}: Frequenze strategie = {percentages}")

    def update_strategy_expected_payoffs(self):
        # Calcola le frequenze di ciascuna strategia
        self.strategy_frequencies = [0] * self.n_of_strategies
        for player in self.players:
            self.strategy_frequencies[player.strategy] += 1
        self.strategy_frequencies = [f / self.num_agents for f in self.strategy_frequencies]

        # Calcola i guadagni attesi per ogni strategia
        self.strategy_expected_payoffs = [
            sum(self.payoff_matrix[strategy][s] * self.strategy_frequencies[s] for s in range(self.n_of_strategies))
            for strategy in range(self.n_of_strategies)
        ]


if __name__ == "__main__":
    # Matrice dei guadagni per il gioco Carta-Forbice-Sasso
    payoff_matrix = np.array([[0, -1, 1],
                              [1, 0, -1],
                              [-1, 1, 0]])

    n_of_players = 500
    initial_distribution = [167, 167, 166]  # Distribuzione quasi equa
    noise = 0.05  # Probabilità di errore casuale
    payoff_method = "expected"  # Metodo di calcolo del guadagno

    
    model = GameModel(n_of_players, payoff_matrix, initial_distribution, noise, payoff_method)

    
    steps = 100
    for _ in range(steps):
        model.step()

    # Grafico dell'evoluzione delle strategie nel tempo
    strategy_history = np.array(model.strategy_history)
    plt.figure(figsize=(10, 6))
    for i in range(model.n_of_strategies):
        plt.plot(strategy_history[:, i], label=f"Strategia {i}")
    plt.xlabel("Passi")
    plt.ylabel("Frequenza")
    plt.title(f"Evoluzione della Distribuzione delle Strategie in {steps} Passi")
    plt.legend()
    plt.show()

    # Distribuzione finale delle strategie
    strategies = [agent.strategy for agent in model.players]
    counts = {s: strategies.count(s) for s in range(model.n_of_strategies)}

    # Mapping delle etichette desiderate
    labels = {0: "0", 1: "1", 2: "-1"}  

    # Grafico finale della distribuzione delle strategie
    plt.figure(figsize=(10, 4))
    plt.bar(range(model.n_of_strategies), [counts[i] for i in range(model.n_of_strategies)],
            color=["skyblue", "salmon", "lightgreen"])
    plt.xticks(ticks=range(model.n_of_strategies), labels=[labels[i] for i in range(model.n_of_strategies)])
    plt.xlabel("Strategie")
    plt.ylabel("Numero di Giocatori")
    plt.title(f"Distribuzione Finale delle Strategie dopo {steps} Passi\n(metodo guadagno: {payoff_method})")
    plt.show()
