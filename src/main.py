from datetime import datetime
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from pytz import timezone
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

ENV_NAME = "CartPole-v1"
TRAINING_EPISODES = 1000
SAVE_EVERY = 100


class Agent(nn.Module):
    def __init__(self, gamma: float = 0.9) -> None:
        super().__init__()

        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=0),
        ).to(self.device)
        self.optimizer = Adam(self.policy.parameters())

        self.log_probabilities = []
        self.rewards = []

    def get_action(self, observation: np.ndarray) -> int:
        observation = torch.from_numpy(observation).to(self.device)
        probabilities = self.policy(observation)
        multinomial = Categorical(probabilities)
        action = multinomial.sample()
        self.log_probabilities.append(multinomial.log_prob(action))
        return action.item()

    def run_optimization(self) -> float:
        self.optimizer.zero_grad()
        loss = self._get_loss()
        loss.backward()
        self.optimizer.step()
        del self.log_probabilities[:]
        del self.rewards[:]
        return loss.item()

    def _get_loss(self) -> torch.Tensor:
        running_reward = 0
        returns = []
        for reward in self.rewards[::-1]:
            running_reward = reward + self.gamma * running_reward
            returns.append(running_reward)
        returns = list(reversed(returns))
        losses = []
        for log_probability, return_ in zip(self.log_probabilities, returns, strict=True):
            # the negation is necessary because by default gradient DEscent in computed
            losses.append(-log_probability * return_)
        return torch.vstack(losses).sum()

    def save_policy(self, file_name: Optional[str] = None) -> None:
        if file_name is None:
            local_tz = timezone("Europe/Luxembourg")
            local_datetime = datetime.now(local_tz)
            file_name = local_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"../checkpoints/{file_name}.pth"
        torch.save(self.policy.state_dict(), save_path)


def train() -> None:
    torch.set_flush_denormal(True)

    writer = SummaryWriter()
    agent = Agent()
    env = gym.make(ENV_NAME)

    for episode in trange(TRAINING_EPISODES):
        observation, _ = env.reset()
        episode_reward = 0
        for step in range(1, 1000):
            action = agent.get_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            agent.rewards.append(reward)
            if terminated or truncated:
                break
        loss = agent.run_optimization()
        if (episode + 1) % SAVE_EVERY == 0:
            agent.save_policy(f"episode_{episode + 1}")

        writer.add_scalar("loss", loss, episode)
        writer.add_scalar("reward", episode_reward, episode)
        writer.add_scalar("episode length", step, episode)


def predict(file_path: str) -> None:
    agent = Agent()
    agent.policy.load_state_dict(torch.load(file_path))
    env = gym.make(ENV_NAME, render_mode="human")

    observation, _ = env.reset()
    while True:
        with torch.inference_mode():
            action = agent.get_action(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            env.render()
            if terminated or truncated:
                observation, _ = env.reset()


if __name__ == "__main__":
    predict("../checkpoints/episode_1000.pth")
