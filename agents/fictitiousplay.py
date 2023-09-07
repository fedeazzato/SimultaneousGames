from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class FictitiousPlay(Agent):
    
    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)

        
        self.count: dict[AgentID, ndarray] = {}
        #
        # Inicializar count con initial si no es None o, caso contrario, con valores random 
        if initial is None:   
            for agent in self.game.agents:
                # Generar valores aleatorios entre 1 y 10 para cada ndarray de cada counter de cada agente
                random_values = np.random.randint(1,10, self.game.num_actions(agent))
                self.count[agent] = random_values
        else:
            self.count = initial #.deepcopy()
            
        self.learned_policy: dict[AgentID, ndarray] = {}
        for agent in self.game.agents:
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def get_rewards(self) -> dict:
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}
        
        #Calcular los rewards de agente para cada acción conjunta 
        # Ayuda: usar product(*agents_actions) de itertools para iterar sobre agents_actions
        
        for act in product(*agents_actions):
            # action es un diccionario de agente - acción
            action = {}
            for agent in g.agents:
                action[agent] = act[g.agent_name_mapping[agent]]
            g.step(action)
            # el observe guarda
            rewards[act] = g.reward(self.agent)    
        return rewards
    
    def get_utility(self):
        rewards = self.get_rewards()
        # Inicializa con tantos ceros como cantidad de acciones posibles existan
        utility = np.zeros(self.game.num_actions(self.agent))
        
        for act in rewards.keys():
            # acción jugada por el agente actual
            a = act[self.game.agent_name_mapping[self.agent]]

            proba = 1
            for agent in self.game.agents:
                if agent != self.agent:
                    proba *= self.learned_policy[agent]

            utility[a]+= rewards[act]*proba

        return utility
    
    def bestresponse(self):
        a = None
        # retornar la acción de mayor utilidad
        a = np.argmax(self.get_utility)
        return a
     
    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
            return
        for agent in self.game.agents:
            self.count[agent][actions[agent]] += 1
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def action(self):
        self.update()
        return self.bestresponse()
    
    def policy(self):
       return self.learned_policy[self.agent]
    