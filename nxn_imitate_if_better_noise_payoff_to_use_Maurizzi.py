import random
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt


# PARAMETRI - imposta tutti i parametri qui
N_OF_PLAYERS = 500
N_OF_STRATEGIES = 3
PAYOFF_MATRIX = np.array([[0, -1, 1],
                         [1, 0, -1],
                         [-1, 1, 0]])
PROB_REVISION = 0.1
NOISE = 0.01
PAYOFF_TO_USE = "use-strategy-expected-payoff"  # o "play-with-one-rd-agent"
INITIAL_DISTRIBUTION = [167, 167, 166]  # Distribuzione iniziale opzionale


class Player(Agent):
   """Un agente che rappresenta un giocatore nel gioco."""
  
   def __init__(self, unique_id, model, strategy):
       super().__init__(unique_id, model)
       self.strategy = strategy
       self.payoff = 0
       self.strategy_after_revision = strategy


   def update_payoff(self, payoff_to_use):
       if payoff_to_use == "play-with-one-rd-agent":
           self.play_with_one_rd_agent()
       elif payoff_to_use == "use-strategy-expected-payoff":
           self.use_strategy_expected_payoff()


   def play_with_one_rd_agent(self):
       mate = self.random.choice(self.model.schedule.agents)
       self.payoff = self.model.payoff_matrix[self.strategy][mate.strategy]


   def use_strategy_expected_payoff(self):
       strategy_frequencies = [0] * self.model.n_of_strategies
       for agent in self.model.schedule.agents:
           strategy_frequencies[agent.strategy] += 1
       strategy_frequencies = [freq / self.model.num_agents for freq in strategy_frequencies]
       self.payoff = sum(self.model.payoff_matrix[self.strategy][i] * strategy_frequencies[i] for i in range(self.model.n_of_strategies))


   def update_strategy_after_revision(self):
       if random.random() < self.model.noise:
           self.strategy_after_revision = random.choice(range(self.model.n_of_strategies))
       else:
           others = [a for a in self.model.schedule.agents if a != self]
           observed_player = self.random.choice(others) if others else self
           if observed_player.payoff > self.payoff:
               self.strategy_after_revision = observed_player.strategy


   def update_strategy(self):
       self.strategy = self.strategy_after_revision


class GameModel(Model):
   """Classe del modello per il gioco."""


   def __init__(self, n_of_players, n_of_strategies, payoff_matrix, prob_revision, noise, payoff_to_use, initial_distribution=None):
       self.num_agents = n_of_players
       self.n_of_strategies = n_of_strategies
       self.payoff_matrix = payoff_matrix
       self.prob_revision = prob_revision
       self.noise = noise
       self.payoff_to_use = payoff_to_use


       self.schedule = RandomActivation(self)
       self.datacollector = DataCollector(
           model_reporters={"Strategy_frequencies": self.get_strategy_frequencies},
           agent_reporters={"Strategy": "strategy"}
       )


       if initial_distribution:
           if len(initial_distribution) != n_of_strategies:
               raise ValueError("La lunghezza della distribuzione iniziale deve essere uguale al numero di strategie")
           agent_id = 0
           for strategy_index, count in enumerate(initial_distribution):
               for _ in range(count):
                   agent = Player(agent_id, self, strategy_index)
                   self.schedule.add(agent)
                   agent_id += 1
       else:
           for i in range(n_of_players):
               strategy = random.choice(range(n_of_strategies))
               agent = Player(i, self, strategy)
               self.schedule.add(agent)


   def step(self):
       self.datacollector.collect(self)


       for agent in self.schedule.agents:
           agent.update_payoff(self.payoff_to_use)


       for agent in self.schedule.agents:
           if random.random() < self.prob_revision:
               agent.update_strategy_after_revision()


       for agent in self.schedule.agents:
           agent.update_strategy()


   def get_strategy_frequencies(self):
       frequencies = [0] * self.n_of_strategies
       for agent in self.schedule.agents:
           frequencies[agent.strategy] += 1
       return [freq / self.num_agents for freq in frequencies]


if __name__ == "__main__":
   model = GameModel(
       N_OF_PLAYERS,
       N_OF_STRATEGIES,
       PAYOFF_MATRIX,
       PROB_REVISION,
       NOISE,
       PAYOFF_TO_USE,
       INITIAL_DISTRIBUTION
   )


   steps = 100
   for _ in range(steps):
       model.step()


   data = model.datacollector.get_model_vars_dataframe()


   # Estrai le frequenze delle strategie dal DataFrame
   # "Strategy_frequencies" è una lista salvata in ogni riga, la convertiamo in colonne separate
   freqs = data['Strategy_frequencies'].tolist()
   freqs_array = np.array(freqs)


   # Grafico delle frequenze di ogni strategia nel tempo
   plt.figure(figsize=(10, 6))
   for i in range(N_OF_STRATEGIES):
       plt.plot(range(steps), freqs_array[:, i], label=f'Strategy {i}')
   plt.xlabel('Step')
   plt.ylabel('Frequenza Strategie')
   plt.title('Evoluzione delle Frequenze delle Strategie nel Tempo')
   plt.legend()
   plt.grid(True)
   plt.show()