import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from collections import deque

# Constants
SCREEN_SIZE = (1200, 800)
CAR_SIZE = (30, 15)
MAX_SPEED = 8
SENSOR_COUNT = 16
SENSOR_SPREAD = 180
SENSOR_MAX_DISTANCE = 400
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128

# Neural Network with LSTM
class DriverNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(SENSOR_COUNT + 2, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(1) if x.dim() == 2 else x  # Ensure batch dimension
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(F.relu(self.fc2(x[:, -1, :])))
        x = torch.tanh(self.fc3(x))
        return x, hidden

# Track with Obstacles
class Track:
    def __init__(self):
        self.surface = pygame.Surface(SCREEN_SIZE)
        self.surface.fill((30, 30, 30))
        self.walls = [
            pygame.Rect(0, 0, SCREEN_SIZE[0], 50),
            pygame.Rect(0, SCREEN_SIZE[1]-50, SCREEN_SIZE[0], 50),
            pygame.Rect(0, 0, 50, SCREEN_SIZE[1]),
            pygame.Rect(SCREEN_SIZE[0]-50, 0, 50, SCREEN_SIZE[1]),
            pygame.Rect(200, 200, 400, 50),
            pygame.Rect(600, 500, 50, 200),
            pygame.Rect(100, 600, 300, 50),
            pygame.Rect(800, 300, 50, 400),
        ]
        for wall in self.walls:
            pygame.draw.rect(self.surface, (200, 200, 200), wall)
        self.mask = pygame.mask.from_surface(self.surface)

# Car Class
class Car:
    def __init__(self, brain=None):
        self.brain = brain or DriverNN()
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.reset()

    def reset(self):
        self.pos = pygame.Vector2(SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] - 100)
        self.velocity = pygame.Vector2(1, 0)  # Give initial movement
        self.angle = 0
        self.angular_velocity = 0
        self.alive = True
        self.fitness = 0
        self.hidden = None

    def get_sensor_readings(self, track):
        readings = []
        for angle in np.linspace(-SENSOR_SPREAD/2, SENSOR_SPREAD/2, SENSOR_COUNT):
            sensor_dir = pygame.Vector2(1, 0).rotate(self.angle + angle)
            hit_distance = SENSOR_MAX_DISTANCE
            for distance in range(1, SENSOR_MAX_DISTANCE, 5):
                check_pos = self.pos + sensor_dir * distance
                if 0 <= check_pos.x < SCREEN_SIZE[0] and 0 <= check_pos.y < SCREEN_SIZE[1]:
                    if track.mask.get_at((int(check_pos.x), int(check_pos.y))):
                        hit_distance = distance
                        break
            readings.append(hit_distance / SENSOR_MAX_DISTANCE)
        return torch.FloatTensor(readings)

    def get_state(self, track):
        sensors = self.get_sensor_readings(track)
        state = torch.cat([sensors, torch.FloatTensor([self.velocity.length() / MAX_SPEED, self.angular_velocity / 5])])
        return state.unsqueeze(0)

    def apply_physics(self, track):
        if not self.alive:
            return
        self.velocity *= 0.98  # Friction
        self.pos += self.velocity
        self.angle += self.angular_velocity
        self.angular_velocity *= 0.92
        offset = (int(self.pos.x - CAR_SIZE[0] // 2), int(self.pos.y - CAR_SIZE[1] // 2))
        if 0 <= offset[0] < SCREEN_SIZE[0] and 0 <= offset[1] < SCREEN_SIZE[1]:
            if track.mask.overlap(pygame.mask.from_surface(self.get_surface()), offset):
                self.alive = False
                self.fitness -= 10  

    def get_surface(self):
        surf = pygame.Surface(CAR_SIZE, pygame.SRCALPHA)
        pygame.draw.rect(surf, (0, 200, 100) if self.alive else (200, 0, 0), (0, 0, *CAR_SIZE), border_radius=3)
        return pygame.transform.rotate(surf, -self.angle)

    def update(self, track, action):
        if not self.alive:
            return
        throttle, steering, brake = action
        acceleration = pygame.Vector2(throttle * math.cos(math.radians(self.angle)),
                                      throttle * math.sin(math.radians(self.angle)))
        self.velocity += acceleration * 0.5
        if self.velocity.length() > MAX_SPEED:
            self.velocity.scale_to_length(MAX_SPEED)
        self.angular_velocity += steering * 4 * (1 - self.velocity.length() / MAX_SPEED)
        if brake > 0.5:
            self.velocity *= 0.9
        self.fitness += max(self.velocity.dot(pygame.Vector2(1, 0).rotate(self.angle)), 0) * 0.1
        self.apply_physics(track)

# Save & Load Model
def save_model(model, filename="car_model.pth"):
    torch.save(model.state_dict(), filename)
    print("✅ Model saved!")

def load_model(model, filename="car_model.pth"):
    try:
        model.load_state_dict(torch.load(filename))
        model.eval()
        print("✅ Model loaded!")
    except FileNotFoundError:
        print("⚠️ No model found. Training from scratch.")

# Training
def train():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    clock = pygame.time.Clock()
    track = Track()
    car = Car()
    load_model(car.brain)
    rewards = []
    running = True

    while running:
        state = car.get_state(track)
        with torch.no_grad():
            action, car.hidden = car.brain(state, car.hidden)
            action = action.squeeze().numpy()

        car.update(track, action)
        rewards.append(car.fitness)
        if not car.alive:
            car.reset()

        # Display
        screen.blit(track.surface, (0, 0))
        screen.blit(car.get_surface(), car.pos - pygame.Vector2(CAR_SIZE) / 2)
        pygame.display.flip()
        clock.tick(60)

        # Quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    save_model(car.brain)
    pygame.quit()

    # Plot Fitness Over Time
    plt.plot(rewards)
    plt.xlabel('Steps')
    plt.ylabel('Fitness')
    plt.title('Training Progress')
    plt.show()

if __name__ == "__main__":
    train()
