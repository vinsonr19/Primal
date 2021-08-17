from pygame.sprite import Sprite
from pygame import Surface
from constants import *


# Define a class for our pixels/grid spaces
class Pixel(Sprite):
    def __init__(self, x, y):
        super(Pixel, self).__init__()
        self.x = TOTAL*x + 1
        self.y = TOTAL*y + 1
        self.state = 0
        self._change_color()

        self.surf = Surface((GRID_PIXEL_SIZE, GRID_PIXEL_SIZE))
        self.surf.fill(self.color)
        self.rect = self.surf.get_rect()
        self.show()

    def _change_color(self):
        self.color = state_dict[self.state]

    def show(self):
        screen.blit(self.surf, (self.x, self.y))

    def change_state(self, state):
        if self.state == state:
            return

        self.state = state
        self._change_color()
        self.surf.fill(self.color)
        self.rect = self.surf.get_rect()
        self.show()
