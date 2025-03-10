import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random
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

# Advanced Neural Network with LSTM
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
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(F.relu(self.fc2(x[:, -1, :])))
        x = torch.tanh(self.fc3(x))
        return x, hidden


# Track Class
class Track:
    def __init__(self):
        self.surface = pygame.Surface(SCREEN_SIZE)
        self.surface.fill((30, 30, 30))
        
        # Complex track design with multiple turns
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

# Physics-based Car Class
class Car:
    def __init__(self, brain=None):
        self.pos = pygame.Vector2(SCREEN_SIZE[0]//2, SCREEN_SIZE[1]-100)
        self.velocity = pygame.Vector2(0, 0)
        self.angle = 0
        self.angular_velocity = 0
        self.speed = 0
        self.fitness = 0
        self.alive = True
        self.sensor_data = []
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        
        self.brain = brain or DriverNN()
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)
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
        state = torch.cat([
            sensors,
            torch.FloatTensor([self.velocity.length() / MAX_SPEED, self.angular_velocity / 5])
        ])
        
        # Ensure state is 2D (batch_size=1, features)
        state = state.unsqueeze(0)  # Converts (features,) → (1, features)
        
        return state

    
    def apply_physics(self, track):
        if not self.alive:
            return
            
        self.velocity *= 0.98  # Friction
        self.pos += self.velocity
        self.angle += self.angular_velocity
        self.angular_velocity *= 0.92  # Angular damping
        
        # Corrected collision detection
        offset = (int(self.pos.x - CAR_SIZE[0] // 2), int(self.pos.y - CAR_SIZE[1] // 2))
        if 0 <= offset[0] < SCREEN_SIZE[0] and 0 <= offset[1] < SCREEN_SIZE[1]:
            if track.mask.overlap(pygame.mask.from_surface(self.get_surface()), offset):
                self.alive = False
                self.fitness -= 10  # Penalty for crashing
            
    def get_surface(self):
        surf = pygame.Surface(CAR_SIZE, pygame.SRCALPHA)
        pygame.draw.rect(surf, (0, 200, 100) if self.alive else (200, 0, 0), 
                         (0, 0, *CAR_SIZE), border_radius=3)
        return pygame.transform.rotate(surf, -self.angle)
    
    def update(self, track, action):
        if not self.alive:
            return
            
        throttle, steering, brake = action
        
        acceleration = pygame.Vector2(
            throttle * math.cos(math.radians(self.angle)),
            throttle * math.sin(math.radians(self.angle))
        )
        
        self.velocity += acceleration * 0.5
        if self.velocity.length() > MAX_SPEED:
            self.velocity.scale_to_length(MAX_SPEED)
            
        self.angular_velocity += steering * 4 * (1 - self.velocity.length() / MAX_SPEED)
        
        if brake > 0.5:
            self.velocity *= 0.9
            
        self.fitness += max(self.velocity.dot(pygame.Vector2(1, 0).rotate(self.angle)), 0) * 0.1
        self.apply_physics(track)

# Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, model):
        self.model = model
        self.target_model = DriverNN()
        self.target_model.load_state_dict(model.state_dict())
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.gamma = 0.95
        self.update_target_every = 1000
        self.steps = 0
        
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        current_q = self.model(states)[0].gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states)[0].max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            
        self.steps += 1
        return loss.item()

# Training Loop
def train():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    clock = pygame.time.Clock()
    
    track = Track()
    car = Car()
    agent = DQNAgent(car.brain)
    
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    
    while True:
        state = car.get_state(track)
        
        if np.random.random() < epsilon:
            action = np.random.uniform(-1, 1, size=3)
        else:
            with torch.no_grad():
                action, car.hidden = car.brain(state, car.hidden)
                action = action.squeeze().numpy()
        
        car.update(track, action)
        next_state = car.get_state(track)
        reward = car.fitness
        done = not car.alive
        
        car.memory.append((state, torch.FloatTensor(action), torch.FloatTensor([reward]), next_state, torch.FloatTensor([done])))
        
        if len(car.memory) >= BATCH_SIZE:
            batch = random.sample(car.memory, BATCH_SIZE)
            batch = [torch.stack(t) for t in zip(*batch)]
            agent.update(batch)
        
        if not car.alive:
            car = Car(car.brain)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        screen.blit(track.surface, (0, 0))
        screen.blit(car.get_surface(), car.pos - pygame.Vector2(CAR_SIZE)/2)
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()

if __name__ == "__main__":
    train()
