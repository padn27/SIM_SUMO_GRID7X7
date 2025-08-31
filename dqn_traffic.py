import traci
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Configuração SUMO

sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-n", "grid_tls.net.xml", "-r", "trips.trips.xml"]

# Replay Buffer

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

    def __len__(self):
        return len(self.buffer)

# Rede Neural do Agente

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Função de escolha de ação (epsilon-greedy)

def select_action(agent, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, agent.fc3.out_features - 1)
    else:
        with torch.no_grad():
            q_values = agent(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(q_values).item()

# Função de treino

def train(agent, target_agent, buffer, optimizer, batch_size=64, gamma=0.99):
    if len(buffer) < batch_size:
        return

    states, actions, rewards, next_states = buffer.sample(batch_size)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_agent(next_states).max(1)[0]
    target = rewards + gamma * next_q_values

    loss = nn.MSELoss()(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Loop de simulação RL

def run_simulation_dqn_phases(steps=3600, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, target_update=50):
    traci.start(sumoCmd)
    tls_list = traci.trafficlight.getIDList()

    agents = {}
    target_agents = {}
    buffers = {}
    prev_states = {}
    prev_actions = {}

    for tls_id in tls_list:
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        n_phases = traci.trafficlight.getPhaseNumber(tls_id)
        state_size = len(lanes) + n_phases
        agents[tls_id] = DQNAgent(state_size, n_phases)
        target_agents[tls_id] = DQNAgent(state_size, n_phases)
        target_agents[tls_id].load_state_dict(agents[tls_id].state_dict())
        buffers[tls_id] = ReplayBuffer()

    optimizers = {tls_id: optim.Adam(agents[tls_id].parameters(), lr=0.001) for tls_id in tls_list}
    epsilon = epsilon_start

    for step in range(steps):
        traci.simulationStep()
        total_stopped = 0
        total_reward = 0

        for tls_id in tls_list:
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            n_phases = traci.trafficlight.getPhaseNumber(tls_id)

            lane_state = np.array([traci.lane.getLastStepHaltingNumber(l) for l in lanes], dtype=np.float32)
            current_phase = traci.trafficlight.getPhase(tls_id)
            phase_onehot = np.zeros(n_phases)
            phase_onehot[current_phase] = 1
            state = np.concatenate([lane_state, phase_onehot])

            total_stopped += sum(lane_state)

            agent = agents[tls_id]
            target_agent = target_agents[tls_id]
            action = select_action(agent, state, epsilon)
            traci.trafficlight.setPhase(tls_id, action)

            if tls_id in prev_states:
                reward = -sum(lane_state) - 0.1 * traci.lane.getWaitingTime(lanes[0])
                total_reward += reward
                buffers[tls_id].push(prev_states[tls_id], prev_actions[tls_id], reward, state)

            prev_states[tls_id] = state
            prev_actions[tls_id] = action

            train(agent, target_agent, buffers[tls_id], optimizers[tls_id])

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if step % target_update == 0:
            for tls_id in tls_list:
                target_agents[tls_id].load_state_dict(agents[tls_id].state_dict())

        if step % 100 == 0:
            print(f"Step {step}: {total_stopped} veículos parados, Reward={total_reward:.2f}, Epsilon={epsilon:.2f}")

    traci.close()

if __name__ == "__main__":
    run_simulation_dqn_phases()