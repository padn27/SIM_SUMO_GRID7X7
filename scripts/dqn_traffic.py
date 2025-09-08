import traci
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------
# Configuração GUI / sem GUI
# ------------------------
USE_GUI = True  # Pra lembrar  >  True para GUI, False para treino rápido sem GUI
if USE_GUI:
    SUMO_BINARY = "/usr/bin/sumo-gui"
else:
    SUMO_BINARY = "/usr/bin/sumo"

SUMO_CMD = [SUMO_BINARY, "-n", "net.net.xml", "-r", "routes.rou.xml", "--step-length", "1"]
OUTPUT_DIR = "results_dqn_3fases"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# Configurações gerais
# ------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

STEPS = 500_000              
EPISODE_STEPS = 2_000        
NUM_EPISODES = int(np.ceil(STEPS / EPISODE_STEPS))  

MOVING_AVG_WINDOW = 150      

BATCH_SIZE = 128             
GAMMA = 0.99
LR = 1e-4                    
EPS_START = 1.0
EPS_END = 0.01               
EPS_DECAY = 0.99995          

MIN_GREEN = 20               
YELLOW_TIME = 5
RED_TIME = 5
REPLAY_CAPACITY = 200_000    

UPDATES_PER_STEP = 8          
SOFT_TAU = 0.005
GRAD_CLIP_MAX_NORM = 1.0

SENSOR_RANGE = 15.0           

# ------------------------
# Buffer de replay
# ------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states)

    def __len__(self):
        return len(self.buffer)

# ------------------------
# Rede neural DQN
# ------------------------
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

# ------------------------
# Política epsilon-greedy
# ------------------------
def select_action(agent, state, epsilon, device):
    if random.random() < epsilon:
        return random.randint(0, agent.fc3.out_features - 1)
    else:
        agent.eval()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = agent(s)
            action = torch.argmax(q_values, dim=1).item()
        agent.train()
        return action

# ------------------------
# Soft update helper
# ------------------------
def soft_update(target, source, tau):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_((1.0 - tau) * t_param.data + tau * s_param.data)

# ------------------------
# Treinamento do DQN
# ------------------------
def train_step(agent, target_agent, buffer, optimizer, batch_size=BATCH_SIZE, gamma=GAMMA, device='cpu'):
    if len(buffer) < batch_size:
        return None

    states, actions, rewards, next_states = buffer.sample(batch_size)
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)

    q_values_all = agent(states)
    q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_actions = torch.argmax(agent(next_states), dim=1)
        next_q_values = target_agent(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + gamma * next_q_values

    loss_fn = nn.MSELoss()
    loss = loss_fn(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP_MAX_NORM)
    optimizer.step()
    soft_update(target_agent, agent, SOFT_TAU)

    return loss.item()

# ------------------------
# Funções de plotagem
# ------------------------
def plot_final_metrics(metrics):
    plt.figure(figsize=(14, 10))
    plt.subplot(2,2,1)
    plt.plot(metrics['episode_rewards'], label='Recompensa por episódio', color='blue')
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Total")
    plt.title("Evolução da Recompensa")
    plt.legend()
    plt.subplot(2,2,2)
    plt.plot(metrics['avg_waiting'], label='Tempo médio de espera', color='red')
    plt.xlabel("Episódio")
    plt.ylabel("Segundos")
    plt.title("Tempo médio de espera por episódio")
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(metrics['avg_throughput'], label='Throughput médio', color='green')
    plt.xlabel("Episódio")
    plt.ylabel("Veículos")
    plt.title("Throughput médio por episódio")
    plt.legend()
    plt.subplot(2,2,4)
    plt.plot(metrics['avg_queue'], label='Fila média', color='purple')
    plt.xlabel("Episódio")
    plt.ylabel("Veículos")
    plt.title("Tamanho médio da fila por episódio")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------
# Loop principal
# ------------------------
def run_simulation_dqn_episodes(step_print_interval=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    episode_rewards, episode_losses, avg_waiting, avg_throughput, avg_queue = [], [], [], [], []
    all_steps_rewards = []
    best_episode_reward = -np.inf

    # --- Start SUMO ---
    traci.start(SUMO_CMD)
    traci.simulationStep()
    tls_list = traci.trafficlight.getIDList()
    if not tls_list:
        raise RuntimeError("Nenhum semáforo encontrado na rede.")
    max_lanes = max(len(traci.trafficlight.getControlledLanes(tls)) for tls in tls_list)

    agents, target_agents, buffers, optimizers, epsilons = {}, {}, {}, {}, {}
    for tls_id in tls_list:
        state_size = max_lanes * 4
        action_size = 3
        agent = DQNAgent(state_size, action_size).to(device)
        target_agent = DQNAgent(state_size, action_size).to(device)
        target_agent.load_state_dict(agent.state_dict())
        agents[tls_id] = agent
        target_agents[tls_id] = target_agent
        buffers[tls_id] = ReplayBuffer(REPLAY_CAPACITY)
        optimizers[tls_id] = optim.Adam(agent.parameters(), lr=LR)
        epsilons[tls_id] = EPS_START

    # --- Episódios ---
    for ep in range(NUM_EPISODES):
        prev_states = {tls_id: np.zeros(max_lanes*4, dtype=np.float32) for tls_id in tls_list}
        prev_actions = {tls_id: 0 for tls_id in tls_list}
        phase_time_count = {tls_id: 0 for tls_id in tls_list}
        waiting_history = {tls: [] for tls in tls_list}
        throughput_history = {tls: [] for tls in tls_list}
        queue_history = {tls: [] for tls in tls_list}
        rewards_total_steps = []
        losses_ep = []

        for step in range(EPISODE_STEPS):
            traci.simulationStep()
            step_reward_sum = 0

            for tls_id in tls_list:
                lanes = traci.trafficlight.getControlledLanes(tls_id)
                lane_state, lane_speed, waiting, vehicles_count = [], [], [], []
                for l in lanes:
                    vehicles_in_lane = traci.lane.getLastStepVehicleIDs(l)
                    vehicles_in_range = [v for v in vehicles_in_lane if traci.vehicle.getLanePosition(v) <= SENSOR_RANGE]
                    lane_state.append(len([v for v in vehicles_in_range if traci.vehicle.getSpeed(v) < 0.1]))
                    lane_speed.append(np.mean([traci.vehicle.getSpeed(v) for v in vehicles_in_range]) if vehicles_in_range else 0.0)
                    waiting.append(np.sum([traci.vehicle.getWaitingTime(v) for v in vehicles_in_range]) if vehicles_in_range else 0.0)
                    vehicles_count.append(len(vehicles_in_range))

                lane_state_pad = np.pad(lane_state, (0, max_lanes - len(lane_state)))
                lane_speed_pad = np.pad(lane_speed, (0, max_lanes - len(lane_speed)))
                waiting_pad = np.pad(waiting, (0, max_lanes - len(waiting)))
                vehicles_pad = np.pad(vehicles_count, (0, max_lanes - len(vehicles_count)))

                state = np.concatenate([
                    lane_state_pad / max(lane_state_pad.max(), 1.0),
                    lane_speed_pad / max(lane_speed_pad.max(), 1e-3),
                    waiting_pad / max(waiting_pad.max(), 1.0),
                    vehicles_pad / 10.0
                ])

                agent = agents[tls_id]
                target_agent = target_agents[tls_id]
                action = select_action(agent, state, epsilons[tls_id], device=device)

                current_phase = traci.trafficlight.getPhase(tls_id)
                phase_time_count[tls_id] += 1
                defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
                num_phases = len(defs[0].phases)
                phase_duration_tls = [MIN_GREEN if i==0 else YELLOW_TIME if i==1 else RED_TIME for i in range(num_phases)]

                if action == 1 and phase_time_count[tls_id] >= phase_duration_tls[current_phase]:
                    traci.trafficlight.setPhase(tls_id, (current_phase+1) % num_phases)
                    phase_time_count[tls_id] = 0
                elif action == 2 and lane_state_pad.sum() > 5:
                    phase_time_count[tls_id] = max(phase_time_count[tls_id]-1, 0)

                reward = 2.0 * vehicles_pad.sum() - 1.0 * waiting_pad.sum() - 1.5 * lane_state_pad.sum()
                reward += 0.5 * max(0, prev_states[tls_id][:len(lanes)].sum() - lane_state_pad.sum())
                reward = np.clip(reward / max(vehicles_pad.sum(), 1.0), -200, 200)

                step_reward_sum += reward
                buffers[tls_id].push(prev_states[tls_id], prev_actions[tls_id], reward, state)

                for _ in range(UPDATES_PER_STEP):
                    loss = train_step(agent, target_agent, buffers[tls_id], optimizers[tls_id], device=device)
                    if loss is not None:
                        losses_ep.append(loss)

                prev_states[tls_id] = state
                prev_actions[tls_id] = action
                epsilons[tls_id] = max(EPS_END, epsilons[tls_id] * EPS_DECAY)

                waiting_history[tls_id].append(waiting_pad.sum())
                throughput_history[tls_id].append(vehicles_pad.sum())
                queue_history[tls_id].append(lane_state_pad.sum())

            rewards_total_steps.append(step_reward_sum)
            if (step+1) % step_print_interval == 0 or step == EPISODE_STEPS-1:
                print(f"[Ep {ep+1}/{NUM_EPISODES}] Step {step+1} - Recompensa últimos {step_print_interval} steps: "
                      f"{sum(rewards_total_steps[-step_print_interval:]):.2f}")

        all_steps_rewards.extend(rewards_total_steps)
        episode_rewards.append(sum(rewards_total_steps))
        episode_losses.append(np.mean(losses_ep) if len(losses_ep) > 0 else 0)
        avg_waiting.append(np.mean([np.mean(waiting_history[tls]) for tls in tls_list]))
        avg_throughput.append(np.mean([np.mean(throughput_history[tls]) for tls in tls_list]))
        avg_queue.append(np.mean([np.mean(queue_history[tls]) for tls in tls_list]))

        print(f"=== Episódio {ep+1}/{NUM_EPISODES} concluído: Recompensa total: {episode_rewards[-1]:.2f}, Loss média: {episode_losses[-1]:.3f} ===")

        if episode_rewards[-1] > best_episode_reward:
            best_episode_reward = episode_rewards[-1]
            for tls_id, agent in agents.items():
                path = os.path.join(OUTPUT_DIR, f"best_agent_{tls_id}.pt")
                torch.save(agent.state_dict(), path)

    traci.close()

    df_metrics = pd.DataFrame({
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
        "avg_waiting": avg_waiting,
        "avg_throughput": avg_throughput,
        "avg_queue": avg_queue
    })
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

    plot_final_metrics({
        "episode_rewards": episode_rewards,
        "avg_waiting": avg_waiting,
        "avg_throughput": avg_throughput,
        "avg_queue": avg_queue
    })

    plt.figure(figsize=(14,6))
    plt.plot(episode_rewards, color='lightblue', alpha=0.6, label='Recompensa por episódio')
    window = 10
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, color='red', linewidth=2, label=f'Média Móvel {window} episódios')
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Total")
    plt.title("Evolução da Recompensa por Episódio")
    plt.legend()
    plt.show()

    return all_steps_rewards

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    rewards = run_simulation_dqn_episodes()
    plt.figure(figsize=(14,6))
    plt.plot(rewards, color='lightblue', alpha=0.6, label='Recompensa por step')
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, color='red', linewidth=2, label=f'Média móvel {window} steps')
    plt.xlabel("Step")
    plt.ylabel("Recompensa acumulada")
    plt.title("Evolução da Recompensa por Step")
    plt.legend()
    plt.show()
