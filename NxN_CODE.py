from mesa import Agent, Model  # Import Agent and Model classes from Mesa
from mesa.space import NetworkGrid  # Import NetworkGrid for managing agents on a network
from mesa.datacollection import DataCollector  # Import DataCollector for collecting simulation data
import numpy as np  # Import NumPy for numerical operations
import random  # Import random for random number generation
import networkx as nx

class Player(Agent):
    """An agent representing a player in the game."""
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)  # Initialize the agent with a unique ID and the model
        self.strategy = self.random.choice(["A", "B"])  # Randomly assign an initial strategy ("A" or "B")
        self.payoff = 0  # Initialize the agent's payoff to 0
        self.xcor = 0  # Placeholder for the x-coordinate of the agent
        self.ycor = 0  # Placeholder for the y-coordinate of the agent

    def update_payoff(self):
        """Update the player's payoff based on interactions with neighbors."""
        neighbors = self.model.grid.get_neighbors(self)  # Get the agent's neighbors from the grid
        for neighbor in neighbors:  # Iterate through each neighbor
            if self.strategy == "A" and neighbor.strategy == "A":  # If both have strategy "A"
                self.payoff += 1  # Increase payoff by 1
            elif self.strategy == "B" and neighbor.strategy == "B":  # If both have strategy "B"
                self.payoff += 2  # Increase payoff by 2
            else:  # If strategies differ
                self.payoff += 0  # No payoff

    def update_strategy_after_revision(self):
        """Update strategy based on neighbors' strategies and payoffs."""
        neighbors = self.model.grid.get_neighbors(self)  # Get the agent's neighbors from the grid
        best_payoff = self.payoff  # Start with the agent's current payoff
        best_strategy = self.strategy  # Start with the agent's current strategy
        
        for neighbor in neighbors:  # Iterate through each neighbor
            if neighbor.payoff > best_payoff:  # If a neighbor has a higher payoff
                best_payoff = neighbor.payoff  # Update the best payoff
                best_strategy = neighbor.strategy  # Update the best strategy
        
        # Introduce noise in strategy selection
        if random.random() < self.model.noise:  # With a probability equal to the model's noise
            self.strategy = random.choice(["A", "B"])  # Randomly choose a new strategy
        else:  # Otherwise
            self.strategy = best_strategy  # Adopt the best strategy

class GameModel(Model):
    """A model with a fixed number of players."""
    
    def __init__(self, N=100, prob_revision=0.5, noise=0.1):
        self.num_agents = N  # Number of agents in the model
        self.G = nx.Graph()
        self.grid = NetworkGrid(self.G)  # Initialize a NetworkGrid (this is incorrect and needs a graph as input)
        self.prob_revision = prob_revision  # Probability of strategy revision
        self.noise = noise  # Noise level in strategy selection
        
        # Create players
        for i in range(self.num_agents):  # Loop through the number of agents
            player = Player(i, self)  # Create a new player agent
            self.schedule.add(player)  # Add the player to the scheduler
            self.grid.place_agent(player, (0, 0))  # Place the player on the grid at position (0, 0)

        self.build_erdos_renyi_network()  # Build an Erdos-Renyi network
        self.relax_network()  # Adjust the network layout for visualization

    def build_erdos_renyi_network(self):
        """Build an Erdos-Renyi network."""
        for i in range(self.num_agents):  # Loop through all agents
            for j in range(i + 1, self.num_agents):  # Loop through pairs of agents
                if random.random() < 0.1:  # With a 10% probability
                    self.grid.add_edge(i, j)  # Add an edge between the two agents

    def step(self):
        """Advance the model by one step."""
        for agent in self.schedule.agents:  # Loop through all agents
            agent.update_payoff()  # Update each agent's payoff
        
        for agent in self.schedule.agents:  # Loop through all agents again
            if random.random() < self.prob_revision:  # With a probability of strategy revision
                agent.update_strategy_after_revision()  # Update the agent's strategy
        
        # Reset payoffs for the next round
        for agent in self.schedule.agents:  # Loop through all agents
            agent.payoff = 0  # Reset the payoff to 0

        self.relax_network()  # Adjust the network layout for visualization

    def relax_network(self):
        """Adjust the layout of the network for better visualization."""
        for _ in range(3):  # Repeat 3 times for better layout
            factor = np.sqrt(len(self.schedule.agents))  # Calculate a scaling factor
            self.layout_spring(factor)  # Apply a spring layout algorithm

        # Don't bump the edges of the world
        x_offset = max([player.xcor for player in self.schedule.agents]) + min([player.xcor for player in self.schedule.agents])  # Calculate x offset
        y_offset = max([player.ycor for player in self.schedule.agents]) + min([player.ycor for player in self.schedule.agents])  # Calculate y offset
        
        # Adjust offsets to limit large jumps
        x_offset = self.limit_magnitude(x_offset, 0.1)  # Limit the x offset
        y_offset = self.limit_magnitude(y_offset, 0.1)  # Limit the y offset

        for player in self.schedule.agents:  # Loop through all agents
            player.xcor -= x_offset / 2  # Adjust the x-coordinate
            player.ycor -= y_offset / 2  # Adjust the y-coordinate

    def layout_spring(self, factor):
        """Placeholder for spring layout algorithm."""
        # Implement the spring layout algorithm to position players
        # This is a placeholder; you would need to implement the actual layout logic
        pass

    def limit_magnitude(self, number, limit):
        """Limit the magnitude of a number to a specified limit."""
        if number > limit:  # If the number exceeds the positive limit
            return limit  # Return the positive limit
        if number < -limit:  # If the number exceeds the negative limit
            return -limit  # Return the negative limit
        return number  # Otherwise, return the number as is

# Initialize the game model
model = GameModel(N=100, prob_revision=0.5, noise=0.1)  # Create an instance of the GameModel

# Run the model for 10 steps
for step in range(10):  # Loop for 10 steps
    model.step()  # Advance the model by one step
    print(f"Step {step + 1} completed.")  # Print a message indicating the step is completed
