import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CAR_SIZE = 20
SENSOR_MAX_DISTANCE = 200
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.1

# Neural Network Definition
class DriverNN(nn.Module):
    def __init__(self):
        super(DriverNN, self).__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Car Class
class Car:
    def __init__(self, brain=None):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT - 100
        self.angle = 0
        self.speed = 0
        self.fitness = 0
        self.sensors = []
        self.alive = True
        
        if brain:
            self.brain = brain
        else:
            self.brain = DriverNN()
        
    def get_sensor_data(self, track):
        sensor_angles = [-90, -45, 0, 45, 90]
        sensor_data = []
        
        for angle in sensor_angles:
            rad_angle = math.radians(self.angle + angle)
            dx = math.cos(rad_angle)
            dy = math.sin(rad_angle)
            
            distance = 0
            collision = False
            while distance < SENSOR_MAX_DISTANCE and not collision:
                distance += 1
                check_x = int(self.x + dx * distance)
                check_y = int(self.y + dy * distance)
                
                if check_x < 0 or check_x >= SCREEN_WIDTH or \
                   check_y < 0 or check_y >= SCREEN_HEIGHT:
                    collision = True
                    
                for wall in track.walls:
                    if wall.clipline((check_x, check_y), (check_x, check_y)):
                        collision = True
                        break
            
            sensor_data.append(distance / SENSOR_MAX_DISTANCE)
        
        return torch.FloatTensor(sensor_data)
    
    def update(self, track):
        if not self.alive:
            return
            
        # Get sensor data
        sensor_input = self.get_sensor_data(track)
        
        # Neural network decision
        with torch.no_grad():
            output = self.brain(sensor_input)
        
        # Control the car
        self.angle += (output[0].item() - 0.5) * 5
        self.speed = output[1].item() * 5
        
        # Update position
        rad_angle = math.radians(self.angle)
        self.x += math.cos(rad_angle) * self.speed
        self.y += math.sin(rad_angle) * self.speed
        
        # Check collision
        car_rect = pygame.Rect(self.x - CAR_SIZE//2, self.y - CAR_SIZE//2, 
                             CAR_SIZE, CAR_SIZE)
        self.alive = not any(wall.colliderect(car_rect) for wall in track.walls)
        
        self.fitness += self.speed

# Track Class
class Track:
    def __init__(self):
        self.walls = [
            pygame.Rect(0, 0, SCREEN_WIDTH, 20),
            pygame.Rect(0, SCREEN_HEIGHT-20, SCREEN_WIDTH, 20),
            pygame.Rect(0, 0, 20, SCREEN_HEIGHT),
            pygame.Rect(SCREEN_WIDTH-20, 0, 20, SCREEN_HEIGHT)
        ]

# Genetic Algorithm Functions
def create_new_generation(previous_population):
    sorted_pop = sorted(previous_population, key=lambda x: x.fitness, reverse=True)
    next_pop = []
    
    # Keep top performers
    for i in range(2):
        next_pop.append(Car(brain=sorted_pop[i].brain))
    
    # Breed and mutate
    while len(next_pop) < POPULATION_SIZE:
        parent = random.choice(sorted_pop[:5])
        child = Car(brain=parent.brain)
        
        # Mutate weights
        with torch.no_grad():
            for param in child.brain.parameters():
                param.add_(torch.randn(param.size()) * MUTATION_RATE)
        
        next_pop.append(child)
    
    return next_pop

# Main Simulation
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    track = Track()
    
    population = [Car() for _ in range(POPULATION_SIZE)]
    
    for generation in range(GENERATIONS):
        running = True
        while running:
            screen.fill((0, 0, 0))
            
            # Draw track
            for wall in track.walls:
                pygame.draw.rect(screen, (255, 255, 255), wall)
            
            # Update and draw cars
            alive_count = 0
            for car in population:
                car.update(track)
                
                if car.alive:
                    alive_count += 1
                    # Draw car
                    car_rect = pygame.Rect(car.x - CAR_SIZE//2, car.y - CAR_SIZE//2,
                                         CAR_SIZE, CAR_SIZE)
                    pygame.draw.rect(screen, (0, 255, 0), car_rect)
                    
                    # Draw sensors
                    sensor_angles = [-90, -45, 0, 45, 90]
                    for i, angle in enumerate(sensor_angles):
                        rad_angle = math.radians(car.angle + angle)
                        end_x = car.x + math.cos(rad_angle) * SENSOR_MAX_DISTANCE
                        end_y = car.y + math.sin(rad_angle) * SENSOR_MAX_DISTANCE
                        pygame.draw.line(screen, (255, 0, 0), 
                                       (car.x, car.y), (end_x, end_y), 1)
            
            pygame.display.flip()
            clock.tick(60)
            
            if alive_count == 0:
                running = False
        
        # Create next generation
        population = create_new_generation(population)
    
    pygame.quit()

if __name__ == "__main__":
    main()