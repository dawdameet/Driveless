import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
CAR_SIZE = (40, 20)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Create window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-Driving Car Simulation")
clock = pygame.time.Clock()

# Walls (for sensors to detect)
walls = [
    pygame.Rect(100, 100, 600, 20),  # Top wall
    pygame.Rect(100, 480, 600, 20),  # Bottom wall
    pygame.Rect(100, 100, 20, 400),  # Left wall
    pygame.Rect(680, 100, 20, 400),  # Right wall
]

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0  # Car's rotation angle
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.2
        self.friction = 0.05
        self.sensor_angles = [-60, -30, 0, 30, 60]  # Sensor directions relative to car
        self.sensors = []

    def move(self, keys):
        # Handle acceleration & braking
        if keys[pygame.K_UP]:
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        elif keys[pygame.K_DOWN]:
            self.speed = max(self.speed - self.acceleration, -self.max_speed / 2)
        else:
            self.speed *= (1 - self.friction)  # Apply friction when no keys are pressed

        # Handle turning
        if keys[pygame.K_LEFT]:
            self.angle += 3  # Rotate counter-clockwise
        if keys[pygame.K_RIGHT]:
            self.angle -= 3  # Rotate clockwise

        # Move the car
        self.x += self.speed * np.cos(np.radians(self.angle))
        self.y += self.speed * np.sin(np.radians(self.angle))

        # Update sensors
        self.update_sensors()

    def update_sensors(self):
        """ Casts sensor rays and detects distances to obstacles """
        self.sensors = []
        for sensor_angle in self.sensor_angles:
            angle = self.angle + sensor_angle  # Adjusted relative to car direction
            distance = self.cast_ray(angle)
            self.sensors.append(distance)

    def cast_ray(self, angle):
        """ Casts a ray in the given direction and returns the distance to the nearest wall """
        ray_length = 150  # Max sensor distance
        x_end = self.x + ray_length * np.cos(np.radians(angle))
        y_end = self.y + ray_length * np.sin(np.radians(angle))

        # Check collision with walls
        for wall in walls:
            if wall.clipline((self.x, self.y), (x_end, y_end)):  
                return np.linalg.norm(np.array([self.x, self.y]) - np.array([x_end, y_end]))  

        return ray_length  # No collision, return max range

    def draw(self, screen):
        """ Draws the car and its sensors """
        car_surface = pygame.Surface(CAR_SIZE, pygame.SRCALPHA)
        car_surface.fill(RED)
        rotated_car = pygame.transform.rotate(car_surface, self.angle)
        car_rect = rotated_car.get_rect(center=(self.x, self.y))
        screen.blit(rotated_car, car_rect.topleft)

        # Draw sensors
        for i, sensor_angle in enumerate(self.sensor_angles):
            angle = self.angle + sensor_angle
            distance = self.sensors[i]
            x_end = self.x + distance * np.cos(np.radians(angle))
            y_end = self.y + distance * np.sin(np.radians(angle))
            pygame.draw.line(screen, BLUE, (self.x, self.y), (x_end, y_end), 2)

# Create car instance
car = Car(WIDTH//2, HEIGHT//2)

# Game loop
running = True
while running:
    screen.fill(WHITE)

    # Draw walls
    for wall in walls:
        pygame.draw.rect(screen, BLACK, wall)

    # Car movement & sensors
    keys = pygame.key.get_pressed()
    car.move(keys)
    car.draw(screen)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
