import pygame
pygame.init()

WIDTH, HEIGHT = 800, 600
CAR_SIZE = (40, 20)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-Driving Car Simulation")
clock = pygame.time.Clock()

walls = [
    pygame.Rect(100, 100, 600, 20),  
    pygame.Rect(100, 480, 600, 20),  
    pygame.Rect(100, 100, 20, 400),  
    pygame.Rect(680, 100, 20, 400),
]

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.2
        self.friction = 0.05
        self.sensor_angles = [-60, -30, 0, 30, 60]
        self.sensors = []

    def move(self, keys):
        if keys[pygame.K_UP]:
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        elif keys[pygame.K_DOWN]:
            self.speed = max(self.speed - self.acceleration, -self.max_speed / 2)
