"""
Play the 2048 game rendered by PyGame library.
Each game is logged - pairs of (game state, action) are saved.
"""

import pygame
import sys, os
from pygame.locals import *
from board_gym import GymBoard
from util import Step, Game, encode_action
import pickle
from pprint import pprint


def draw_text(surface, text, color, rect, font, aa=False, bkg=None):
    """
    Draw some text into an area of a surface.
    Automatically wraps words.
    Returns any text that didn't get blitted.
    """

    rect = Rect(rect)
    y = rect.top
    line_spacing = -2

    # get the height of the font
    font_height = font.size("Tg")[1]

    while text:
        i = 1

        # determine if the row of text will be outside our area
        if y + font_height > rect.bottom:
            break

        # determine maximum width of line
        while font.size(text[:i])[0] < rect.width and i < len(text):
            i += 1

        # if we've wrapped the text, then adjust the wrap to the last word
        if i < len(text):
            i = text.rfind(" ", 0, i) + 1

        # render the line and blit it to the surface
        if bkg:
            image = font.render(text[:i], 1, color, bkg)
            image.set_colorkey(bkg)
        else:
            image = font.render(text[:i], aa, color)

        surface.blit(image, (rect.left, y))
        y += font_height + line_spacing

        # remove the text we just blitted
        text = text[i:]

    return text


def render_board(board):
    screen.fill(COLOR_MAP["background"])

    for row in range(board.shape[0]):
        for column in range(board.shape[1]):
            tile_value = board.matrix[row, column]
            if tile_value in COLOR_MAP:
                color = COLOR_MAP[tile_value]
            #elif (row, column) == board.last_random_tile_index:
            #    color = YELLOW
            else:
                color = COLOR_MAP["super"]

            pygame.draw.rect(screen,
                             color,
                             [(T_MARGIN + T_WIDTH) * column + T_MARGIN,
                              (T_MARGIN + T_HEIGHT) * row + T_MARGIN,
                              T_WIDTH,
                              T_HEIGHT])

            if (row, column) == board.last_random_tile_index:
                text_color = RED
            else:
                text_color = COLOR_MAP["text"]

            if tile_value:
                text = str(tile_value)
            else:
                text = ""

            draw_text(screen,
                      text,
                      text_color,
                      [(T_MARGIN + T_WIDTH) * column + T_WIDTH / 2,
                       (T_MARGIN + T_HEIGHT) * row + T_HEIGHT / 2,
                       T_WIDTH,
                       T_HEIGHT],
                      FONT_TILE,
                      aa=True)

            draw_text(screen,
                      "SCORE: " + str(board.score),
                      BLACK,
                      [50,
                       HEIGHT - 50,
                       200,
                       50],
                      FONT_SCORE,
                      aa=True)


def play(board):
    steps = []
    render_board(board)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in POSSIBLE_ACTIONS:
                    matrix = board.matrix
                    action = POSSIBLE_ACTIONS[event.key]
                    moved = board.move(action)

                    if moved:
                        print()
                        print(board.matrix)
                        print("SCORE:", board.score, "\tSTEP:", board.n_steps_valid, "\tHIGHEST VALUE:", board.highest_value)
                        steps.append(Step(matrix=matrix, action=action, action_encoded=encode_action(action)))
                        render_board(board)

                        if board.is_gameover():
                            print("GAME OVER!")
                            return Game(steps=steps, score=board.score, random_seed=board.random_seed, is_gameover=True)
                    else:
                        print("\nCannot move to this direction!")
                elif event.key == pygame.K_q:
                    screen.fill(BLACK)
                    return Game(steps=steps, random_seed=board.random_seed, is_gameover=False)
                elif event.key == pygame.K_p:
                    screen.fill(BLACK)
                    return "quit"

        clock.tick(60)
        pygame.display.flip()


### PyGame options
os.environ["SDL_VIDEO_CENTERED"] = "1"
pygame.init()
WIDTH = 700
HEIGHT = 700
WINDOW_SIZE = (WIDTH, HEIGHT)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("2048")
FONT_TILE = pygame.font.SysFont("Arial", 40, bold=True)
FONT_SCORE = pygame.font.SysFont("Arial", 24, bold=True)
clock = pygame.time.Clock()
# Tile sizes
T_WIDTH = 150
T_HEIGHT = 150
T_MARGIN = 10

### Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

COLOR_MAP = {
    "text": (119, 110, 101),
    "background": (187, 173, 160),
    0: (238, 228, 218),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    "super": (60, 58, 50)
}

### Game logging
games_file = "data/games_01.pickle"
games = []
n_games = 3

# Map keys to 2048 board actions.
POSSIBLE_ACTIONS = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3
}

for i in range(n_games):
    board = GymBoard()
    print(board.matrix)
    game = play(board)
    if game == "quit":
        break
    #pprint(game)
    games.append(game)

with open(games_file, mode="bw") as f:
    pickle.dump(games, f)
