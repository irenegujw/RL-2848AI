# 1. The 2048 Game

__Game Objective__: The main goal of 2048 is to slide numbered tiles on a grid to combine them and create a tile with the number 2048.

__Game Setup__: The game begins with a 4x4 grid. Initially, two tiles with either the number 2 or 4 are placed in random positions on the grid.

__Moving Tiles__: Players can slide the tiles in one of four directions: up, down, left, or right. Every time the player makes a move, all tiles slide as far as possible in the chosen direction until they are stopped by either another tile or the edge of the grid.

__Combining Tiles__: If two tiles of the same number collide while moving, they will merge into a single tile. The value of the new tile is the sum of the two original tiles.

__New Tile Appearance__: After each swipe, a new tile with a number 2 or 4 appears randomly on an empty spot on the board. The placement is random and cannot be predicted.

__Game Over__: The game ends when the board is filled with tiles and there are no more moves left that can combine tiles. At this point, the player's score is calculated based on the value of the tiles on the board.

__Scoring__: The score starts at zero and increases by the value of the combined tiles at each merge. For instance, merging two tiles with the number 4 into an 8-tile scores 8 points.

# 2. Markov Decision Process (MDP)

## 2.1. State: 4x4 Grid State

The game state in 2048 can be represented as a 4x4 grid, where each cell contains either an empty space or a tile with a numbered value. The grid can be represented as a 2D array or matrix.

To preprocess the game state for input into a neural network, we can consider two common approaches:

1.  Logarithmic Encoding: We can apply a logarithmic transformation to each tile value. This helps to reduce the range of values and provides a more compact representation. For example, we can use $\log_2(x)$ for each tile value $x$. Empty cells can be represented by a small negative value (e.g., -1).
2.  One-Hot Encoding: We can use one-hot encoding to represent each tile value as a binary vector. In this approach, we define a fixed set of possible tile values and create a binary vector for each cell, where the corresponding value is 1 and all other values are 0.

## 2.2. Action: Up, Down, Left, Right
__Up__: Slide all tiles upward. As input of neural network, we can use 1 or [1,0,0,0] to represent this action.

__Down__: Slide all tiles downward. As input of neural network, we can use 2 or [0,1,0,0] to represent this action.

__Left__: Slide all tiles to the left. As input of neural network, we can use 3 or [0,0,1,0] to represent this action.

__Right__: Slide all tiles to the right. As input of neural network, we can use 4 or [0,0,0,1] to represent this action.


## 2.3. Reward: Based on Merging Numbers, Reaching the Goal, or Game Over
__Merging Numbers__: When two tiles merge, the model receives a positive reward equal to the value of the newly created tile. This encourages the model to make moves that lead to merging tiles.

__Invalid Move__: When no tile merge after move, the model recive a 0 or negative reaward. This penalty encourage the model merging tiles first.

__Reaching 2048__: When the model creates a tile with the value 2048, it receives a large positive reward . This reward signifies that the model has won the game and encourages it to reach the 2048 tile.

__Game Over__: When the game ends (there are no more possible moves), the model receives a negative reward.

In addition to these rewards, we can also consider giving small positive rewards for each successful move to encourage the model to make progress and avoid getting stuck.

## 2.4. Transition Probability: Based on Current State and Action

The transition probability ($P(s' | s, a)$) defines the probability of moving from one state to another when taking a specific action. In the 2048 game, the transition probability depends on the current state, the chosen action, and the probability of a new tile (either 2 or 4) appearing in an empty cell after the action is performed. Commonly, the probability of a new tile being 2 is usually higher than the probability of it being 4.

## 2.5. Discount Factor

The discount factor ($\gamma$), is a parameter that determines the importance of future rewards. we might choose $\gamma=0.995$ as the initial value, and then adjust it based on the training results.

# 3. Deep Q-Network (DQN)

## 3.1. Q-Network and Target Network

The Q-network takes the state as input and outputs the Q-values for each possible action and the target network has the same architecture as the Q-network but with frozen parameters. The target network is used to calculate the target Q-values during the training process. The parameters of the target network are periodically updated with the parameters of the Q-network.

## 3.3. Experience Replay
The replay buffer is a memory structure that stores the model's experiences in the form of transitions $(s, a, r, s')$. As the model interacts with the environment, new transitions are added to the replay buffer. 

## 3.4. ε-Greedy Policy
To encourage exploration of the environment, we set a probability $\epsilon$ (epsilon), the model selects a random action. With probability $1 - \epsilon$, the model selects the action with the highest Q-value according to the current Q-network.

## 3.5. Loss Function: Mean Squared Error (MSE)
We will use the data randomly chose from the replay buffer and calculate the loss function through Mean Square Error.

## 3.6. Updating the Value Networks
The Q-network parameters are updated using gradient descent to minimize the MSE loss. The target network parameters are updated periodically base on the Q-nerwork.


# 5. Training Process

## 5.1. Initializing Q-Network and Target Network

The first step in the training process is to initialize the Q-network and the target network. 

Both networks are initialized with the same architecture and random weights. We choose convolutional layers and fully connected layers as hidden layer. The input to the networks is the preprocessed game state (e.g., log-encoded or one-hot encoded), and the output is the predicted Q-values for each possible action.

## 5.2. Interacting with the Game Environment and Storing Experiences

During training, the DQN model interacts with the game environment to collect experiences. At each time step, the model observes the current state of the game and selects an action based on the ε-greedy policy. The model then performs the selected action and observes the resulting next state, reward, and whether the game has ended. Then, the experience tuple $(s, a, r, s', end)$ is stored in the replay buffer. 

## 5.3. Sampling Batches from Replay Buffer and Updating Q-Network

During training, the DQN model periodically samples batch of experiences from the replay buffer to update the Q-network. The sampled experiences are used to calculate the target Q-values and the loss for updating the Q-network parameters.

For each experience $(s, a, r, s', end)$ in the batch, the target Q-value is calculated using the Bellman equation and target network:

$y = r + \gamma \max_{a'} Q(s', a')$

where $Q(s', a')$ is the Q-value of the next state $s'$ calculated using the __target network__.

The loss function is then calculated between the predicted Q-values from the Q-network and the target Q-values:

$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

$y_i$ is the Q value from target network, and $\hat{y}_i$ is the predicted Q-value from the Q-network.

## 5.4. Periodically Soft Updating Target Network

Target network parameters are periodically updated from the Q-network parameters to the target network. We will update target network every 500 steps at the begining. It can be tuned based on the training setup.
