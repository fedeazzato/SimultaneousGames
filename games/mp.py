import numpy as np
from numpy import ndarray
from gymnasium.spaces import Discrete
#from pettingzoo.utils.env import ActionDict, ObsDict
from base.game import SimultaneousGame

ObsDict = dict #env.ObsDict
AgentID = dict #env.AgentID
ActionDict = dict #env.ActionDict

# implementaciòn del juego "Matching Penies"
# it is played between two players, Even and Odd. Each player has a penny and must secretly turn the penny to heads or tails.
# The players then reveal their choices simultaneously. If the pennies match (both heads or both tails), 
# then Even keeps both pennies, so wins one from Odd (+1 for Even, −1 for Odd). If the pennies do not match
#  (one heads and one tails) Odd keeps both pennies, so receives one from Even (−1 for Even, +1 for Odd). 
class MP(SimultaneousGame):

    def __init__(self):
        # payoff matrix desde el punto de vista del jugador par
        self._R = np.array([[1., -1.], [-1., 1.]])

        # agents
        self.agents = ["agent_" + str(r) for r in range(2)] # you get ["agent_0", "agent_1"]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents)))) # salida {"agent_0": 0, "agent_1": 1}


        # actions
        self._moves = ['H', 'T']
        self._num_actions = 2
        
        # diccionario a la salida de la siguiente linea:
        #{
        #    "agent_0": Discrete(2),
        #    "agent_1": Discrete(2)
        #    }       
        self.action_spaces = {
            agent: Discrete(self._num_actions) for agent in self.agents
        }

        # observations
        self.observation_spaces = {
            agent: ActionDict for agent in self.agents
        }

    def step(self, actions: ActionDict) -> tuple[ObsDict, dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        # rewards
        (a0, a1) = tuple(map(lambda agent: actions[agent], self.agents))
        r = self._R[a0][a1]
        self.rewards[self.agents[0]] = r
        self.rewards[self.agents[1]] = -r

        # observations
        self.observations = dict(map(lambda agent: (agent, actions), self.agents))

        # etcetera
        self.terminations = dict(map(lambda agent: (agent, True), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[ObsDict, dict[str, dict]]:
        self.observations = dict(map(lambda agent: (agent, None), self.agents))
        self.rewards = dict(map(lambda agent: (agent, None), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

        return self.observations, None

    def render(self) -> ndarray | str | list | None:
        for agent in self.agents:
            print(agent, self._moves[self.agent_name_mapping[agent]], self.rewards[agent])
