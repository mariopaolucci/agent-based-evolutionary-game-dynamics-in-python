#import
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

#parametri
WIDTH = 50
HEIGHT = 50
STEPS = 100
PROB_REVISION = 0.5  #probabilità che un agente consideri un cambio di strategia
NOISE = 0.1          #probabilità di scelta casuale
PAYOFF_MATRIX = np.array([
    [0, -1, 1],  #sasso
    [1, 0, -1],  #carta
    [-1, 1, 0]   #forbici
])
N_STRATEGIES = PAYOFF_MATRIX.shape[0]


#scheduler perchè mesa.time non funziona
class CustomRandomActivation:
    def __init__(self, model):
        self.model = model
        self.agents = []

    def add(self, agent):
        self.agents.append(agent)

    def step(self):
        for agent in random.sample(self.agents, len(self.agents)):
            agent.step()
        for agent in self.agents:
            agent.advance()


#creazione agente
class RPSAgent:
    def __init__(self, unique_id, model, strategy):
        self.unique_id = unique_id
        self.model = model
        self.strategy = strategy
        self.strategy_after_revision = strategy
        self.payoff = 0.0
        self.pos = None

    def step(self):
        mate = random.choice([
            agent for agent in self.model.schedule.agents if agent != self
        ])
        self.payoff = self.model.payoff_matrix[self.strategy, mate.strategy]

    def advance(self):
        if random.random() < PROB_REVISION:
            if random.random() < NOISE:
                self.strategy_after_revision = random.randint(0, N_STRATEGIES - 1)
            else:
                observed = random.choice([
                    agent for agent in self.model.schedule.agents if agent != self
                ])
                if observed.payoff > self.payoff:
                    self.strategy_after_revision = observed.strategy
        self.strategy = self.strategy_after_revision


#modello generale con rumore
class RPSModel(Model):
    def __init__(self, width, height, initial_distribution=None):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = CustomRandomActivation(self)
        self.payoff_matrix = PAYOFF_MATRIX

        uid = 0
        total_agents = width * height

        if initial_distribution is None:
            initial_distribution = [1 / N_STRATEGIES] * N_STRATEGIES

        strategy_counts = [int(p * total_agents) for p in initial_distribution]
        while sum(strategy_counts) < total_agents:
            for i in range(len(strategy_counts)):
                strategy_counts[i] += 1
                if sum(strategy_counts) == total_agents:
                    break

        strategies = []
        for i, count in enumerate(strategy_counts):
            strategies += [i] * count
        random.shuffle(strategies)

        for x in range(width):
            for y in range(height):
                strategy = strategies.pop()
                agent = RPSAgent(uid, self, strategy)
                self.grid.place_agent(agent, (x, y))
                agent.pos = (x, y)
                self.schedule.add(agent)
                uid += 1

    def step(self):
        self.schedule.step()

    def get_distribution(self):
        counts = [0] * N_STRATEGIES
        for agent in self.schedule.agents:
            counts[agent.strategy] += 1
        total = len(self.schedule.agents)
        return [c / total for c in counts]


#animazione grafico
model = RPSModel(WIDTH, HEIGHT)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

grid_data = np.zeros((WIDTH, HEIGHT))
for x in range(WIDTH):
    for y in range(HEIGHT):
        agents = model.grid.get_cell_list_contents([(x, y)])
        if agents:
            grid_data[x, y] = agents[0].strategy

im = ax1.imshow(grid_data.T, origin="lower", cmap="tab10", vmin=0, vmax=N_STRATEGIES - 1)
ax1.set_title("Distribuzione Strategie")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

x_data = []
y_data = [[] for _ in range(N_STRATEGIES)]
lines = [
    ax2.plot([], [], label=f"Strategia {i}")[0] for i in range(N_STRATEGIES)
]
ax2.set_xlim(0, STEPS)
ax2.set_ylim(0, 1)
ax2.set_xlabel("Passi")
ax2.set_ylabel("Frequenza")
ax2.set_title("Evoluzione Strategie")
ax2.legend()

def update(frame):
    model.step()
    dist = model.get_distribution()

    for x in range(WIDTH):
        for y in range(HEIGHT):
            agents = model.grid.get_cell_list_contents([(x, y)])
            if agents:
                grid_data[x, y] = agents[0].strategy
    im.set_array(grid_data.T)

    x_data.append(frame)
    for i in range(N_STRATEGIES):
        y_data[i].append(dist[i])
        lines[i].set_data(x_data, y_data[i])

    ax2.set_xlim(0, max(50, frame + 1))
    return [im] + lines

ani = FuncAnimation(fig, update, frames=STEPS, interval=300, blit=False, repeat=False)
plt.tight_layout()
plt.show()

#####################################################esercizio 1#####################################################
#parametri per l'esercizio
WIDTH = 20
HEIGHT = 10
STEPS = 100
PROB_REVISION = 0.1 #probabilità della revisione della strategia settata a 0.1
NOISE = 0.0 #rumore a zero
PAYOFF_MATRIX = np.array([
    [2, 4],  # Defect vs Defect, Defect vs Cooperate
    [1, 3]   # Cooperate vs Defect, Cooperate vs Cooperate
])
N_STRATEGIES = PAYOFF_MATRIX.shape[0]


#scheduler
class CustomRandomActivation:
    def __init__(self, model):
        self.model = model
        self.agents = []

    def add(self, agent):
        self.agents.append(agent)

    def step(self):
        for agent in random.sample(self.agents, len(self.agents)):
            agent.step()
        for agent in self.agents:
            agent.advance()


#agente
class RPSAgent:
    def __init__(self, unique_id, model, strategy):
        self.unique_id = unique_id
        self.model = model
        self.strategy = strategy
        self.strategy_after_revision = strategy
        self.payoff = 0.0
        self.pos = None

    def step(self):
        mate = random.choice([
            agent for agent in self.model.schedule.agents if agent != self
        ])
        self.payoff = self.model.payoff_matrix[self.strategy, mate.strategy]

    def advance(self):
        if random.random() < PROB_REVISION:
            if random.random() < self.model.noise:
                self.strategy_after_revision = random.randint(0, N_STRATEGIES - 1)
            else:
                observed = random.choice([
                    agent for agent in self.model.schedule.agents if agent != self
                ])
                if observed.payoff > self.payoff:
                    self.strategy_after_revision = observed.strategy
        self.strategy = self.strategy_after_revision


#modello
class RPSModel(Model):
    def __init__(self, width, height, initial_distribution, noise):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = CustomRandomActivation(self)
        self.payoff_matrix = PAYOFF_MATRIX
        self.noise = noise

        uid = 0
        total_agents = width * height
        strategy_counts = [int(p * total_agents) for p in initial_distribution]
        while sum(strategy_counts) < total_agents:
            for i in range(len(strategy_counts)):
                strategy_counts[i] += 1
                if sum(strategy_counts) == total_agents:
                    break

        strategies = []
        for i, count in enumerate(strategy_counts):
            strategies += [i] * count
        random.shuffle(strategies)

        for x in range(width):
            for y in range(height):
                strategy = strategies.pop()
                agent = RPSAgent(uid, self, strategy)
                self.grid.place_agent(agent, (x, y))
                agent.pos = (x, y)
                self.schedule.add(agent)
                uid += 1

    def step(self):
        self.schedule.step()

    def get_distribution(self):
        counts = [0] * N_STRATEGIES
        for agent in self.schedule.agents:
            counts[agent.strategy] += 1
        total = len(self.schedule.agents)
        return [c / total for c in counts]


#animazione grafico
model = RPSModel(WIDTH, HEIGHT, initial_distribution=[0.0, 1.0], noise=NOISE)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
grid_data = np.zeros((WIDTH, HEIGHT))
for x in range(WIDTH):
    for y in range(HEIGHT):
        agents = model.grid.get_cell_list_contents([(x, y)])
        if agents:
            grid_data[x, y] = agents[0].strategy

im = ax1.imshow(grid_data.T, origin="lower", cmap="coolwarm", vmin=0, vmax=N_STRATEGIES - 1)
ax1.set_title("Strategie sullo spazio")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

x_data = []
y_data = [[] for _ in range(N_STRATEGIES)]
lines = [
    ax2.plot([], [], label=f"Strategia {i}")[0] for i in range(N_STRATEGIES)
]
ax2.set_xlim(0, STEPS)
ax2.set_ylim(0, 1)
ax2.set_xlabel("Passi")
ax2.set_ylabel("Frequenza")
ax2.set_title("Evoluzione delle strategie")
ax2.legend()

def update(frame):
    #modificare il livello di rumore
    if frame == 10:
        model.noise = 0.02  #rumore minimo

    model.step()
    dist = model.get_distribution()

    for x in range(WIDTH):
        for y in range(HEIGHT):
            agents = model.grid.get_cell_list_contents([(x, y)])
            if agents:
                grid_data[x, y] = agents[0].strategy
    im.set_array(grid_data.T)

    x_data.append(frame)
    for i in range(N_STRATEGIES):
        y_data[i].append(dist[i])
        lines[i].set_data(x_data, y_data[i])

    ax2.set_xlim(0, max(50, frame + 1))
    return [im] + lines

ani = FuncAnimation(fig, update, frames=STEPS, interval=300, blit=False, repeat=False)
plt.tight_layout()
plt.show()

#################################################esercizio 2########################################################
#parametri
WIDTH = 15
HEIGHT = 20
STEPS = 100
PROB_REVISION = 0.1
NOISE_INITIAL = 0.0 
PAYOFF_MATRIX = np.array([
    [ 0, -1,  1],  #sasso
    [ 1,  0, -1],  #carta
    [-1,  1,  0]   #forbici
])
N_STRATEGIES = PAYOFF_MATRIX.shape[0]


#scheduler
class CustomRandomActivation:
    def __init__(self, model):
        self.model = model
        self.agents = []

    def add(self, agent):
        self.agents.append(agent)

    def step(self):
        for agent in random.sample(self.agents, len(self.agents)):
            agent.step()
        for agent in self.agents:
            agent.advance()


#agente
class RPSAgent:
    def __init__(self, unique_id, model, strategy):
        self.unique_id = unique_id
        self.model = model
        self.strategy = strategy
        self.strategy_after_revision = strategy
        self.payoff = 0.0
        self.pos = None

    def step(self):
        mate = random.choice([
            agent for agent in self.model.schedule.agents if agent != self
        ])
        self.payoff = self.model.payoff_matrix[self.strategy, mate.strategy]

    def advance(self):
        if random.random() < PROB_REVISION:
            if random.random() < self.model.noise:
                self.strategy_after_revision = random.randint(0, N_STRATEGIES - 1)
            else:
                observed = random.choice([
                    agent for agent in self.model.schedule.agents if agent != self
                ])
                if observed.payoff > self.payoff:
                    self.strategy_after_revision = observed.strategy
        self.strategy = self.strategy_after_revision


#modello
class RPSModel(Model):
    def __init__(self, width, height, initial_distribution, noise):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = CustomRandomActivation(self)
        self.payoff_matrix = PAYOFF_MATRIX
        self.noise = noise

        uid = 0
        total_agents = width * height
        strategy_counts = [int(p * total_agents) for p in initial_distribution]
        while sum(strategy_counts) < total_agents:
            for i in range(len(strategy_counts)):
                strategy_counts[i] += 1
                if sum(strategy_counts) == total_agents:
                    break

        strategies = []
        for i, count in enumerate(strategy_counts):
            strategies += [i] * count
        random.shuffle(strategies)

        for x in range(width):
            for y in range(height):
                strategy = strategies.pop()
                agent = RPSAgent(uid, self, strategy)
                self.grid.place_agent(agent, (x, y))
                agent.pos = (x, y)
                self.schedule.add(agent)
                uid += 1

    def step(self):
        self.schedule.step()

    def get_distribution(self):
        counts = [0] * N_STRATEGIES
        for agent in self.schedule.agents:
            counts[agent.strategy] += 1
        total = len(self.schedule.agents)
        return [c / total for c in counts]


#grafico
model = RPSModel(WIDTH, HEIGHT, initial_distribution=[1/3, 1/3, 1/3], noise=NOISE_INITIAL)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
grid_data = np.zeros((WIDTH, HEIGHT))
for x in range(WIDTH):
    for y in range(HEIGHT):
        agents = model.grid.get_cell_list_contents([(x, y)])
        if agents:
            grid_data[x, y] = agents[0].strategy

im = ax1.imshow(grid_data.T, origin="lower", cmap="tab10", vmin=0, vmax=N_STRATEGIES - 1)
ax1.set_title("Strategie sullo spazio")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

x_data = []
y_data = [[] for _ in range(N_STRATEGIES)]
lines = [
    ax2.plot([], [], label=f"Strategia {i}")[0] for i in range(N_STRATEGIES)
]
ax2.set_xlim(0, STEPS)
ax2.set_ylim(0, 1)
ax2.set_xlabel("Passi")
ax2.set_ylabel("Frequenza")
ax2.set_title("Evoluzione strategie")
ax2.legend()

def update(frame):
    #rumore a 0.001
    if frame == 30:
        model.noise = 0.001

    model.step()
    dist = model.get_distribution()

    for x in range(WIDTH):
        for y in range(HEIGHT):
            agents = model.grid.get_cell_list_contents([(x, y)])
            if agents:
                grid_data[x, y] = agents[0].strategy
    im.set_array(grid_data.T)

    x_data.append(frame)
    for i in range(N_STRATEGIES):
        y_data[i].append(dist[i])
        lines[i].set_data(x_data, y_data[i])

    ax2.set_xlim(0, max(50, frame + 1))
    return [im] + lines

ani = FuncAnimation(fig, update, frames=STEPS, interval=300, blit=False, repeat=False)
plt.tight_layout()
plt.show()

###################################################esercizio 3#######################################################
#parametri
WIDTH = 25
HEIGHT = 40
STEPS = 100
PROB_REVISION = 0.1
NOISE_INITIAL = 0.05
PAYOFF_MATRIX = np.array([
    [1, 1, 0],  # Strategy 0
    [1, 1, 1],  # Strategy 1
    [0, 1, 1]   # Strategy 2
])
N_STRATEGIES = PAYOFF_MATRIX.shape[0]

#scheduler
class CustomRandomActivation:
    def __init__(self, model):
        self.model = model
        self.agents = []

    def add(self, agent):
        self.agents.append(agent)

    def step(self):
        for agent in random.sample(self.agents, len(self.agents)):
            agent.step()
        for agent in self.agents:
            agent.advance()

#agente
class RPSAgent:
    def __init__(self, unique_id, model, strategy):
        self.unique_id = unique_id
        self.model = model
        self.strategy = strategy
        self.strategy_after_revision = strategy
        self.payoff = 0.0
        self.pos = None

    def step(self):
        mate = random.choice([
            agent for agent in self.model.schedule.agents if agent != self
        ])
        self.payoff = self.model.payoff_matrix[self.strategy, mate.strategy]

    def advance(self):
        if random.random() < PROB_REVISION:
            if random.random() < self.model.noise:
                self.strategy_after_revision = random.randint(0, N_STRATEGIES - 1)
            else:
                observed = random.choice([
                    agent for agent in self.model.schedule.agents if agent != self
                ])
                if observed.payoff > self.payoff:
                    self.strategy_after_revision = observed.strategy
        self.strategy = self.strategy_after_revision

#modello
class RPSModel(Model):
    def __init__(self, width, height, initial_distribution, noise):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = CustomRandomActivation(self)
        self.payoff_matrix = PAYOFF_MATRIX
        self.noise = noise

        uid = 0
        total_agents = width * height
        strategy_counts = [int(p * total_agents) for p in initial_distribution]
        while sum(strategy_counts) < total_agents:
            for i in range(len(strategy_counts)):
                strategy_counts[i] += 1
                if sum(strategy_counts) == total_agents:
                    break

        strategies = []
        for i, count in enumerate(strategy_counts):
            strategies += [i] * count
        random.shuffle(strategies)

        for x in range(width):
            for y in range(height):
                strategy = strategies.pop()
                agent = RPSAgent(uid, self, strategy)
                self.grid.place_agent(agent, (x, y))
                agent.pos = (x, y)
                self.schedule.add(agent)
                uid += 1

    def step(self):
        self.schedule.step()

    def get_distribution(self):
        counts = [0] * N_STRATEGIES
        for agent in self.schedule.agents:
            counts[agent.strategy] += 1
        total = len(self.schedule.agents)
        return [c / total for c in counts]


#update modello
def create_model():
    return RPSModel(WIDTH, HEIGHT, initial_distribution=[0.5, 0.0, 0.5], noise=NOISE_INITIAL)

#primo modello
model = create_model()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
grid_data = np.zeros((WIDTH, HEIGHT))
for x in range(WIDTH):
    for y in range(HEIGHT):
        agents = model.grid.get_cell_list_contents([(x, y)])
        if agents:
            grid_data[x, y] = agents[0].strategy

im = ax1.imshow(grid_data.T, origin="lower", cmap="tab20", vmin=0, vmax=N_STRATEGIES - 1)
ax1.set_title("Strategie sullo spazio")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

x_data = []
y_data = [[] for _ in range(N_STRATEGIES)]
lines = [
    ax2.plot([], [], label=f"Strategia {i}")[0] for i in range(N_STRATEGIES)
]
ax2.set_xlim(0, STEPS)
ax2.set_ylim(0, 1)
ax2.set_xlabel("Passi")
ax2.set_ylabel("Frequenza")
ax2.set_title("Evoluzione strategie")
ax2.legend()

#aggiornamento della simulazione
def update(frame):
    global model

    #a metà simulazione si fa il setup
    if frame == 50:
        model = create_model()
        print(" SETUP NUOVO MODELLO ESEGUITO a metà simulazione")

    model.step()
    dist = model.get_distribution()

    for x in range(WIDTH):
        for y in range(HEIGHT):
            agents = model.grid.get_cell_list_contents([(x, y)])
            if agents:
                grid_data[x, y] = agents[0].strategy
    im.set_array(grid_data.T)

    x_data.append(frame)
    for i in range(N_STRATEGIES):
        y_data[i].append(dist[i])
        lines[i].set_data(x_data, y_data[i])

    ax2.set_xlim(0, max(50, frame + 1))
    return [im] + lines

ani = FuncAnimation(fig, update, frames=STEPS, interval=300, blit=False, repeat=False)
plt.tight_layout()
plt.show()