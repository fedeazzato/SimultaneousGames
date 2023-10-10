import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict
from itertools import product

class RegretMatching(Agent):

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        if (initial is None):
          self.curr_policy = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent))
        else:
          self.curr_policy = initial.copy()
        self.cum_regrets = np.zeros(self.game.num_actions(self.agent))
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1
        np.random.seed(seed=seed)
    
    def regrets(self, played_actions: ActionDict) -> dict[AgentID, float]:
        # The played_actions parameter contains the actions chosen by all agents
        #  in the current iteration.
        actions = played_actions.copy()
        # a = actions[self.agent]
        g = self.game.clone()
        # u parece que está destinado a almacenar el regret para cada una de las tres 
        # acciones,que se asumen en orden R - P - S (array [valor valor valor])
        u = np.zeros(g.num_actions(self.agent), dtype=float)
        # 
        # TODO: calcular regrets
        # Como recibe "played actions" se asume que la verdadera jugada ya sucediò
        # entonces, ahora habrá que hacer jugadas en el juego clonado y evaluar
        # las recompensas de las jugadas que no se hicieron por parte del jugador actual
        # dejando fija la jugada del oponente 
        
        # recorro los agentes del juego verdadero para guardar las acciones
        # y las rewards de cada uno. Asumo que el reward que guarda el juego siempre es el último
        for agent in self.game.agents:
            if agent == self.agent:
                self_action = actions[agent]
                self_reward = self.game.reward(agent)
            else:
                opponent_action = actions[agent]
                # opponent_reward = self.game.reward(agent) (reward del oponente sólo para testing)
                opponent_agent = agent

        # Ahora, sobre el juego clonado "g" (que en realidad está vacío porque el clone
        # también implementa un reset) voy a jugar con las acciones alternativas para el
        # jugador actual              
        
        # Voy a iterar sobre todas las acciones posibles dejando afuera a la que ya jugó
        # Armo un diccionario que deja fija la acción del oponente y que varía 
        # las acciones del jugador actual, y se lo paso al step
        # possible_actions = [0,1,2] (esta era la versiòn con las acciones hardcodeadas) 
        
        possible_actions = g.action_iter(self.agent)

        # diccionario tipo {agente: acción}
        alternative_actions = {}
        alternative_actions[opponent_agent] = opponent_action

        # diccionario tipo {acción: regret}
        regrets={}
        regrets[self_action] = 0 # El regret sobre la jugada real del jugador actual es cero

        # print(actions)
        # print("acción y reward jugador actual", self_action, "/", self_reward, " - ", 
        #      "acción y reward oponente", opponent_action, "/", opponent_reward)    

        for action in possible_actions:
            if action != self_action:
                alternative_actions[self.agent] = action                
                _, alternative_rewards, _, _, _ = g.step(alternative_actions)
                regrets[action] = alternative_rewards[self.agent] - self_reward
                    
        #print(regrets)

        # Se pide un retorno con esta forma: dict[AgentID, float], pero mirando el resto del 
        # código, mas bien se espera un array ORDENADO que contenga el regret para cada acción
        # Dejo afuera los valores negativos

        for action in regrets:
            if regrets[action] > 0:
                u[action] = regrets[action]
            else:
                u[action] = 0

        # print (u)
        return u
    
    def regret_matching(self):
        #
        # TODO: calcular curr_policy y actualizar sum_policy
        #

        # Por defecto la curr_policy viene cargada con 1/3 para cada acción
        # Si en los regrets acumulados (sumados) tengo al menos un valor positivo
        # distinto de cero, normalizo y actualizo la sum_policy. 
        # Si no hay ningún valor positivo, dejo la policy inicial que distribuye con igual 
        # probabilidad cada una de las tres acciones 

        sum = self.cum_regrets.sum()
        if sum > 0:
            self.curr_policy = self.cum_regrets / sum
        else:
            self.curr_policy = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent))
        self.sum_policy = self.curr_policy * self.niter
        # print("regret matching  - ", self.agent, "policy - ",  self.curr_policy)               

    def update(self) -> None:
        # niter: nombre comùn para una variable que se usa como conador de loops
        actions = self.game.observe(self.agent)
        if actions is None:
           return
        regrets = self.regrets(actions)
        self.cum_regrets += regrets
        self.regret_matching()
        self.niter += 1
        self.learned_policy = self.sum_policy / self.niter
        # print("update - ", self.agent, "policy - ",  self.curr_policy)  

    def action(self):
        self.update()
        return np.argmax(np.random.multinomial(1, self.curr_policy, size=1))
    
    def policy(self):
        return self.learned_policy
