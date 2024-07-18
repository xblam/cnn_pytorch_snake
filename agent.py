import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Point
from model import Linear_QNet, QTrainer
from cnn_model import SimpleCNN, QTrainerCNN
from helper import plot

DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("USING CUDA")



#Defining the parameters
memory = deque(maxlen=100000)# popleft()
gamma = 0.9
batchSize = 32
nLastStates = 4
epsilon = 1
epsilonDecayRate = 0.001
minEpsilon = 0.001
learningRate = 0.001


game = SnakeGameAI()

class Agent:

    def __init__(self):
        self.n_games = 0
        
        self.model = SimpleCNN().to(DEVICE)
        self.trainer = QTrainerCNN(self.model, lr=learningRate, gamma=gamma)

    # change this to return the state that is the game matrix
    def get_state(self, game):
        state = game.game_matrix
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(memory) > batchSize:
            mini_sample = random.sample(memory, batchSize) # list of tuples
        else:
            mini_sample = memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        print(dones)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        print(done)
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        final_move = [0,0,0]
        if random.randint(0, 200) < epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(DEVICE)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.epsilon -= agent.epsilon_decay
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
                print('Game', agent.n_games, 'Score', score, 'Record:', record,'Epsilon:', agent.epsilon)

            elif agent.n_games % 100 == 0:
                print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Epsilon:', agent.epsilon)


            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()