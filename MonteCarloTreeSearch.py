import gymnasium as gym 
import math
from copy import deepcopy
from statistics import mean


def UCB1(v, n, parent_n):
    return v/n + 2*math.sqrt(math.log(parent_n)/n)

class MonteCarloTreeSearch():
    def __init__(self, max_iterations=1000, env="ALE/Breakout-v5"):
        # Set up the environment 
        self.env = gym.make(env)

        # Start the Environment
        initial_state = self.env.reset()

        # Set the root_state to be a clone of the initial environment state
        self.root_state = self.env.ale.cloneState()

        # Create Root Node
        self.root_node = Node(None, False, 'root', self.root_state)

        # Count for naming of Nodes
        self.counter = 1

        # Total steps taken so far
        self.iterations = 0

        # Maximum number of iterations
        self.max_iterations = max_iterations

    def start_search(self):
        while True:

            # Check if you have reached the max iterations 
            if self.iterations >= self.max_iterations:

                # Return the root node
                return self.root_node
            
            # Choose the next node to explore 
            next_node = self.choose_next_node(self.root_node)

            # Complete random search 
            self.random_search(next_node)

            if self.iterations % 20 == 1:
                print('Explored for ' + str(self.iterations) + ' iterations')

    
    def choose_next_node(self, node):
        # If you are at a terminal state you should end algorithm
        if node.terminal_state == True: 
            # Edge condition for when leaf node is reached
            return node.parent
        
        # Check if the node is new or if it has children already
        elif len(node.children) < 1:
            # If it is a new Node then get all children
            self.get_child_nodes(node)


        max_UCB1 = None
        final_child = None

        explored_child_count = 0

        for child in node.children:
            # If it's the first time going through the node then you take the first child as the final node
            if node.times_visited == 0:
                final_child = child
                break

            # If you haven't explored the child then explore it
            if child.times_visited == 0:
                final_child = child
                break
            
            # Else use the value with the highest UCB1 Score
            else:
                # Calculate the UCB1 Score 
                child_UCB1 = UCB1(child.value, child.times_visited, node.times_visited)
                explored_child_count += 1

            # Initial max UCB1 
            if max_UCB1 == None:
                max_UCB1 = child_UCB1
                final_child = child
            
            # This makes the Algorithm choose which state-action pair is the best
            # If the current child UCB1 score is better than the best UCB1 then it is chosen
            elif child_UCB1 > max_UCB1:
                max_UCB1 = child_UCB1
                final_child = child
        
        # If you have explored all states from the parent class explore child Node
        if explored_child_count == len(node.children):
            final_child = self.choose_next_node(final_child)

        # Return the child to be explored 
        return final_child 

    def random_search(self, node):
        # Set state to explore from
        self.env.ale.restoreState(node.state)

        done = False

        final_reward = 0

        # Randomly traverse until you reach a terminal state
        while not done:
            # Choose random action
            action = self.env.action_space.sample()

            observation, reward, done, truncated, _ = self.env.step(action)
            
            final_reward += reward

            if done == True:
                break
        
        if node.times_visited < 1:
            node.value = final_reward

        # Increase the number of times that the node has been visited 
        node.times_visited += 1

        # Back Prop reward to root 
        while True:

            # Since the root node is initialized to be None this will be our break condition 
            if node.parent == None:
                break

            else:
                # Increase the number of times that the node has been visited 
                node.parent.times_visited += 1
                # Add the reward for the current exploration to the parent node 
                node.parent.value += final_reward
                # Make the parent node the current node
                node = node.parent
        
        # After this you have completed one full iteration 
        self.iterations += 1 


    # Get all children nodes for a parent (take all possible actions from the parent state)
    def get_child_nodes(self, node):

        # Go through all possible actions that can be taken in the state
        for i in range(self.env.action_space.n):
            
            # Load in state informantion
            self.env.ale.restoreState(node.state)

            # Perform action
            observation, reward, done, truncated, _ = self.env.step(i)

            # Create child node 
            child_node = Node(node, done, str(self.counter), self.env.ale.cloneState())

            # If there is a reward in that state add it to the value 
            child_node.value += reward
            
            # Append child to children list in the parent node
            node.children.append(child_node)

            # Increase counter for name 
            self.counter += 1 
                
class Node():
    def __init__(self, parent=False, terminal_state=False, name=None, state=None):
        # Is there a parent Node or not
        self.parent = parent

        # List of all children from the node
        self.children = []

        # Are you in a Terminal state or not
        self.terminal_state = terminal_state

        # The number of times that the state has been visited 
        self.times_visited = 0

        # Value (reward for the Node)
        # Should be 0 and updates after doing a rollout 
        self.value = 0

        # Name for the node
        self.node_name = name

        # State for the node
        self.state = state

        return 
    
if __name__ == '__main__':
    root_node = MonteCarloTreeSearch()
    root_node.start_search()

    done = False

    node = root_node.root_node

    while True:
        print('Node ' + node.node_name +' has value: ' + str(node.value))
        next_value = None
        idx = 0

        for child in node.children:
            if next_value == None:
                next_value = child.value
                best_child = child
                best_idx = idx
                
            elif child.value > next_value:
                next_value = child.value
                best_child = child
                best_idx = idx

            idx += 1

        
        if node.node_name == best_child.node_name:
            break
        
        print('\nBest action to take is ' + str(best_idx))

        node = best_child

        if node.terminal_state == True:
            break