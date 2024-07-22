import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI
from cnn_model import SimpleCNN, QTrainerCNN
import wandb
import pygame

DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("USING CUDA")



# set the parameters
gamma = 0.9
batchSize = 100
nLastStates = 4
minEpsilon = 0.001
learningRate = 0.001

# make the snake model


class Agent:

    # count games
    def __init__(self):
        self.epsilon = 1
        self.epsilon_decay = 0.0001
        self.min_eps = 0.001
        self.learning_rate = 0.001
        self.gamma = 0.9
        self.memory = deque(maxlen = 100000)

        self.n_games = 0
        # specify the model type and Q trainer
        self.model = SimpleCNN().to(DEVICE)
        self.trainer = QTrainerCNN(self.model, lr= self.learning_rate, gamma=self.gamma)

    # put the combination of current state, actino, next_state, and reward into memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    # take all of states in memory, and pass them to the trainer to tweak parameters
    def train_long_memory(self):
        if len(self.memory) > batchSize:
            mini_sample = random.sample(self.memory, batchSize) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    # only train the state we have at hand
    def train_short_memory(self, state, action, reward, next_state, done):
        
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        final_move = [0,0,0]
        if random.randint(0, 1) < max(self.epsilon, self.min_eps):
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(DEVICE)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def get_state(self, game):
        # find a way to stack multiple states, and then send them into the program as one
        # perhasp the best way would be to stack all of those, so the snake will ahve 4 frames of 3 layers each, for a total of channels of imformation 
        
        print('testmp')

def train(log, display):
    scores_list = []
    n_turns_no_improvement = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(display)
    
    if log:
        wandb.init(
        # Set the wandb project where this run will be logged
            project="convolutional_snake_ai"
        )
    while True:
        # get old state
        state_old = game.game_matrix

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)

        state_new = game.game_matrix

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        # if we are dead
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            n_turns_no_improvement += 1
            agent.epsilon -= agent.epsilon_decay
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
                n_turns_no_improvement = 0
                print('Game', agent.n_games, 'Score', score, 'Record:', record,'Epsilon:', agent.epsilon)

            elif agent.n_games % 100 == 0:
                print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Epsilon:', agent.epsilon)

            scores_list.append(score)
            running_mean_scores = np.sum(scores_list[-100:])/100

            # plot(plot_scores, plot_mean_scores)
            if log:
                wandb.log({
                "running mean score (last 100)": running_mean_scores,
                "highest score": record,
                "epoch": agent.n_games,
                "epsilon": agent.epsilon
                })

        if n_turns_no_improvement > 10000:
            break



if __name__ == '__main__':
    log = True
    display = True
    train(log, display)