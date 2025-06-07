import torch
import numpy as np
import pygame
from flappy_bird_env import FlappyBirdEnv
from model import create_model

# Create and load the PyTorch model
model = create_model()
checkpoint = torch.load("final/flappy_model.pth")
model.load_state_dict(checkpoint["model_state"])
model.eval()

env = FlappyBirdEnv()
state = env.reset()
done = False

FRAME_SKIP_RENDER = 1
frame_count = 0

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # Convert state to tensor and get action from model
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())

    next_state, reward, done, _ = env.step(action)

    if frame_count % FRAME_SKIP_RENDER == 0:
        env.render()
    frame_count += 1

    state = next_state

env.close()
print("Game over.")
