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

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLACK = (0,0,0)

BLOCK_SIZE = 100


# set the amount of rows and columns so that we can make the chart instead of displaying the snake
nRows = 8
nCols = 8

get_reward = {
    "food" :   50,
    "death" : -10,
    "move" :  -0.1
}

show_display = False
class SnakeGameAI:    
    def __init__(self):
        self.game_speed = 1000
        self.w = nCols * 100
        self.h = nRows * 100
        # init display
        if show_display:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    # resets the game everytime a snake dies
    def reset(self):
        # init game state
        self.direction = 0
        self.head = Point(nCols/2, nRows/2)
        # this time lets just set the snake as a list
        self.snake = [self.head, Point(self.head.x-1, self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()

        self.update_game_matrix
        if show_display:
            self._update_ui
        self.frame_iteration = 0

    @property
    def update_game_matrix(self):
        # reset the game matrix
        self.game_matrix = np.zeros((2, nCols, nRows))
        for pos in self.snake:
            # the first layer will only contain the white pixels of the snake
            self.game_matrix[0][int(pos[0])][int(pos[1])] = 1
        # the second layer will just contain red of the food
        self.game_matrix[1][self.food[0]][self.food[1]] = 1

    # put down food and make sure it doesnt spawn inside the snake
    def _place_food(self):
        x = random.randint(0, nCols-1)
        y = random.randint(0, nRows-1)
        self.food = Point(x, y)
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

        # 4. place new food or just move
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
        if show_display:
            self._update_ui
        self.clock.tick(self.game_speed)
        # 6. return game over and score
        return reward, game_over, self.score

    # check if the snake has collided with anything
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > nCols-1 or pt.x < 0 or pt.y > nRows-1 or pt.y < 0:
            return True
        # if the snake hits itself
        if pt in self.snake[1:]:
            return True
        return False

    @property
    def _update_ui(self):
        self.display.fill(BLACK)
        for i in range(nCols):
            for j in range(nRows):
                if self.game_matrix[0, i, j] == 1: pygame.draw.rect(self.display, WHITE, pygame.Rect(j*BLOCK_SIZE, i*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) 
                if self.game_matrix[1, i, j] == 1: pygame.draw.rect(self.display, RED, pygame.Rect(j*BLOCK_SIZE, i*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)) 

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
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        # update the position of the snakes head depending on the input we recieve
        x = self.head.x
        y = self.head.y
        if self.direction == 0:
            y -= 1
        elif self.direction == 1:
            y += 1
        elif self.direction == 2:
            x += 1
        elif self.direction == 3:
            x -= 1

    
        self.head = Point(x, y)

if __name__ == "__main__":
    env = SnakeGameAI()