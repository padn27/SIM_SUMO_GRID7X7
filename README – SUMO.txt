README – Simulação de Tráfego com DQN e SUMO

Pré-requisitos

* Python 3.8+
* SUMO (com `sumo-gui` disponível no PATH)
* Bibliotecas Python:


pip install sumolib traci torch numpy


Arquivos necessários

* grid_tls.net.xml → rede 7x7 criada no SUMO
* trips.trips.xml → rotas dos veículos (geradas com `randomTrips.py`)
* dqn_traffic.py → script do agente DQN

Gerar trips 

python3 ~/SUMO_local/sumo-main/tools/randomTrips.py \
  -n grid_tls.net.xml \
  -o trips.trips.xml \
  -b 0 -e 3600 -p 0.5 -l --seed 42


* -b 0 → início da simulação
* -e 3600 → fim da simulação (em segundos)
* -p 0.5 → probabilidade de gerar veículo por passo
* -l → ignora rotas de veículos existentes
* --seed 42` → garante reproducibilidade

Rodar a simulação DQN


python3 dqn_traffic.py


* O agente escolhe fases de semáforo (verde, amarelo, vermelho) baseado no congestionamento
* É mostrado a cada 100 passos: número de veículos parados na rede
* DQN atualiza a rede neural de cada semáforo com replay buffer

Observações

* A simulação abre a interface gráfica `sumo-gui` por padrão; para rodar em modo console, altere no script:


sumoBinary = "sumo"  # ao invés de "sumo-gui"


* O loop de simulação considera congestionamento real, usando fração de veículos parados.
* Rede 7x7: cada semáforo é um agente DQN independente.
* Para ajustar número de passos ou epsilon-greedy: run_simulation_dqn_phases(steps=3600, epsilon=0.1)

