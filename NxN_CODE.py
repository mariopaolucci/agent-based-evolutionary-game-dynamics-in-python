import mesa
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable  # Import ScalarMappable for color mapping
from mesa import Agent, Model  # Import Agent and Model classes from Mesa
from mesa.space import NetworkGrid  # Import NetworkGrid for managing agents on a network
from mesa.datacollection import DataCollector  # Import DataCollector for collecting simulation data
import numpy as np  # Import NumPy for numerical operations
import random  # Import random for random number generation
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt


payoffs = {'A': {'A': 1, 'B': 0}, 'B': {'A': 0, 'B': 2}}  # Define the payoff matrix for strategies A and B

class GameModel(Model):
    """A model with a fixed number of players."""
    
    def __init__(self, N, width, height, seed=None):
        super().__init__(seed=seed)  # Initialize the model with a random seed
        self.num_agents = N  # Number of agents in the model
        self.running = True  # Flag to indicate if the model is running
        self.prob_revision = 0.1  # Probability of strategy revision

        self.G = nx.erdos_renyi_graph(n=self.num_agents, p=0.05, seed=seed)
        self.net = NetworkGrid(self.G)

        # Create agents
        #for node in self.G.nodes():
            #a = Player(
                #self, model=self,  # Pass the model instance to the agent
            #)
        for i, node in enumerate(self.G.nodes()):
             strategy = "A" if i < 70 else "B"
             a = Player(i, self, strategy)
             #self.agents.append(a)
             self.net.place_agent(a, node)

           
            # Add the agent to the node

        self.datacollector = mesa.DataCollector(
            #model_reporters={"Gini": compute_gini},
            agent_reporters={"Wealth": "wealth", "Current_strategy": "strategy"},
        )

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("update_payoff")
        self.agents.shuffle_do("update_strategy")


class Player(Agent):
    """An agent representing a player in the game."""
    
    def __init__(self, unique_id, model, strategy):
        """Initialize the player with a unique ID, model, and strategy."""
        super().__init__(model)  # Initialize the agent with a unique ID and the model
        self.strategy = strategy   #self.random.choice(["A", "B"]) - Randomly assign an initial strategy ("A" or "B")
        self.payoff = 0  # Initialize the agent's payoff to 0
        self.xcor = 0  # Placeholder for the x-coordinate of the agent
        self.ycor = 0  # Placeholder for the y-coordinate of the agent
        self.wealth = 1  # Initialize the agent's wealth to 1
        self.steps_not_given = 0  # Initialize the number of steps not given to 0
        self.name = unique_id  # Assign a name to the agent       

    def update_payoff(self):
        neighbors_nodes = self.model.net.get_neighborhood(
            self.pos, include_center=False
        )
        others=self.model.net.get_cell_list_contents(neighbors_nodes)
        #print(self.model.net.get_cell_list_contents(neighbors_nodes))
        if len(others) > 0 :
            other = self.random.choice(others)
            other.wealth += payoffs[other.strategy][self.strategy]
            self.wealth += payoffs[self.strategy][other.strategy]

    def update_strategy(self):
        if self.random.random() < self.model.prob_revision:  # Fire with probability prob_revision
            others = [x for x in self.model.agents if x != self]
            if len(others) > 0:
                other = self.random.choice(others)
                if other.wealth > self.wealth:
                    self.strategy = other.strategy



# Initialize the game model
model = GameModel(N=100, width=10, height=10)  # Create an instance of the GameModel

# Run the model for 10 steps
for step in range(100):  # Loop for 10 steps
    model.step()  # Advance the model by one step
    print(f"Step {step + 1} completed.")  # Print a message indicating the step is completed

# Collect data for plotting
data = model.datacollector.get_agent_vars_dataframe()

# Calculate average wealth over time
#average_wealth = data.groupby('Step')['Wealth'].mean().reset_index()

# Calculate strategy usage over time
strategy_usage = data.groupby(['Step', 'Current_strategy']).size().unstack(fill_value=0).reset_index()

# Plotting the average wealth over time
#plt.figure(figsize=(10, 6))
#sns.lineplot(data=average_wealth, x='Step', y='Wealth')
#plt.title('Average Wealth Over Time')
#plt.xlabel('Step')
#plt.ylabel('Average Wealth')
#plt.grid()
#plt.show()

# Plotting the usage of strategies A and B over time
plt.figure(figsize=(20, 12))
sns.lineplot(data=strategy_usage.melt(id_vars='Step', value_vars=['A', 'B']), 
             x='Step', y='value', hue='Current_strategy', marker='o')
plt.title('Usage of Strategies A and B Over Time')
plt.xlabel('Step')
plt.ylabel('Number of Agents')
plt.grid()
plt.legend(title='Strategy')
plt.show()

# Create a heatmap of agent strategies on the network (final step)
# Plot strategy heatmap on the network (final step)
# Slideshow of strategies over all steps
# Slideshow of strategies over all steps
# Gather strategy data over time
df = model.datacollector.get_agent_vars_dataframe()
steps = df.index.get_level_values(0).unique()
strategy_map = {"A": 0, "B": 1}
strategy_history = []

for step in steps:
    step_data = df.xs(step, level="Step")
    strategies = step_data["Current_strategy"].map(strategy_map)
    node_colors = [strategies.get(node, 0) for node in model.G.nodes()]
    strategy_history.append(node_colors)

# Setup figure
fig, ax = plt.subplots(figsize=(10, 6))
pos = nx.spring_layout(model.G, seed=42)
nodes = nx.draw(
    model.G, pos,
    node_color=strategy_history[0],
    with_labels=True,
    cmap=plt.cm.coolwarm,
    node_size=300,
    edge_color="gray",
    ax=ax
)

sm = ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
cbar = fig.colorbar(sm, ax=ax, label='Strategy')
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["A", "B"])

# Animation update function
def update(frame):
    ax.clear()
    nx.draw(
        model.G, pos,
        node_color=strategy_history[frame],
        with_labels=True,
        cmap=plt.cm.coolwarm,
        node_size=300,
        edge_color="gray",
        ax=ax
    )
    ax.set_title(f"Agent Strategies – Step {frame}")

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(strategy_history), interval=500)

# Save as GIF (requires ImageMagick or Pillow)
ani.save("strategy_evolution.gif", writer="pillow", fps=2)
