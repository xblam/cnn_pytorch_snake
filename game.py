import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

#TODO: change this into number directions

"""
0 = UP
1 = DOWN
2 = RIGHT
3 = LEFT
"""


# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLACK = (0,0,0)
GREEN = (0, 255, 0)

BLOCK_SIZE = 100




get_reward = {
    "food" :   10,
    "death" : -10,
    "move" :  -0.1
}

class SnakeGameAI:    
    def __init__(self, show_display):
        # set the amount of rows and columns so that we can make the chart instead of displaying the snake
        self.nRows = 8
        self.nCols = 8
        self.show_display = show_display
        self.game_speed = 100000
        self.w = self.nCols * 100
        self.h = self.nRows * 100
        # init display
        if self.show_display:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    # resets the game everytime a snake dies
    def reset(self):
        # init game state, snake is headed up
        self.direction = 0

        self.head = (self.nCols/2, self.nRows/2)
        # this time lets just set the snake as a list
        # maybe change this
        self.snake = [self.head, (self.head[0], self.head[1]+1)]

        self.score = 0
        self.food = None
        self._place_food()

        self.update_game_matrix
        if self.show_display:
            self._update_ui
            # pygame.time.delay(10000)
        self.frame_iteration = 0

    @property
    def update_game_matrix(self):
        # reset the game matrix
        self.game_matrix = np.zeros((3, self.nCols, self.nRows))

        i,j = self.head
        self.game_matrix[1][int(j)][int(i)] = 1
        for pos in self.snake:
            # the first layer will only contain the white pixels of the snake
            self.game_matrix[0][int(pos[1])][int(pos[0])] = 1

        # the second layer will just contain red of the food
        self.game_matrix[2][self.food[1]][self.food[0]] = 1
        # the third layer will be where the head of the snake is


    # put down food and make sure it doesnt spawn inside the snake
    def _place_food(self):
        x = random.randint(0, self.nCols-1)
        y = random.randint(0, self.nRows-1)
        self.food = (x, y)
        if self.food in self.snake:
            self._place_food()

    # the snake does one step in the game
    def play_step(self, action):
        self.frame_iteration += 1
        # if we escape the game, quit the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        
        # dont let the snake idle for too long
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = get_reward["death"]
            return reward, game_over, self.score

        # if the snake hits the food
        if self.head == self.food:
            self.score += 1
            reward = get_reward["food"]
            self._place_food()

        else:
            # punish the snake whenever it moves
            reward = get_reward["move"]
            self.snake.pop()

        # update the game matrix
        self.update_game_matrix
        
        # 5. update ui and clock
        if self.show_display:
            self._update_ui
        self.clock.tick(self.game_speed)
        # 6. return game over and score
        return reward, game_over, self.score

    # check if the snake has collided with anything
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt[1] > self.nCols-1 or pt[1] < 0 or pt[0] > self.nRows-1 or pt[0] < 0:
            return True
        # if the snake hits itself
        if pt in self.snake[1:]:
            return True
        return False

    @property
    def _update_ui(self):
        self.display.fill(BLACK)
        # draw the head of the snake in different color and then draw the snake
        i, j = self.head
        pygame.draw.rect(self.display, GREEN, pygame.Rect(i*BLOCK_SIZE, j*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        for pos in self.snake[1:]:
            i,j = pos
            pygame.draw.rect(self.display, WHITE, pygame.Rect(i*BLOCK_SIZE, j*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) 



        # draw the food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) 

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [0, 3, 1, 2]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        # update the position of the snakes head depending on the input we recieve
        x = self.head[0]
        y = self.head[1]
        if self.direction == 0:
            y -= 1
        elif self.direction == 1:
            y += 1
        elif self.direction == 2:
            x += 1
        elif self.direction == 3:
            x -= 1

        self.head = (x, y)

if __name__ == "__main__":
    env = SnakeGameAI(True)
    for i in range(10):
        # env.play_step([1,0,0])
        env.play_step([1,0,0])
        env.play_step([0,1,0])