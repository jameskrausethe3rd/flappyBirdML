import os
from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent
from model import create_model
import torch
import time

env = FlappyBirdEnv()
agent = DQNAgent(state_dim=6, action_dim=2)

episodes = 1000
target_update = 10

render = True

# Load pre-trained weights if available
if os.path.exists("flappy_model.pth"):
    checkpoint = torch.load("flappy_model.pth")
    agent.q_net.load_state_dict(checkpoint["model_state"])
    agent.target_net.load_state_dict(agent.q_net.state_dict())
    agent.epsilon = checkpoint.get("epsilon", 1.0)
    # agent.epsilon = .1
    print(f"Loaded model and epsilon ({agent.epsilon:.2f}) from flappy_model.pth")

else:
    print("No pre-trained weights found. Starting fresh.")

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.train()
        
        if render and ep % 10 == 0:
            env.render()

    if ep % target_update == 0:
        agent.target_net.load_state_dict(agent.q_net.state_dict())

    print(f"Episode {ep+1}: Total reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}")

    agent.decay_epsilon()

torch.save({
    "model_state": agent.q_net.state_dict(),
    "epsilon": agent.epsilon
}, "flappy_model.pth")

env.close()
