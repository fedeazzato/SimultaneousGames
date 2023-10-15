import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict

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
        actions = played_actions.copy()
        a = actions[self.agent]
        g = self.game.clone()
        u = np.zeros(g.num_actions(self.agent), dtype=float)
        # TODO: calcular regrets
        # Como recibe "played actions" se asume que la verdadera jugada ya sucediò
        # entonces, ahora habrá que hacer jugadas en el juego clonado y evaluar
        # las recompensas de las jugadas que no se hicieron por parte del jugador actual
        # dejando fija la jugada del oponente 

        for action in g.action_iter(self.agent):
            actions[self.agent] = action
            # Reset del ambiente para aplicar la acción alternativa y evaluar utilidad
            g.reset()
            g.step(actions)
            u[action] = g.reward(self.agent)

        # Calcular los regrets positivos de acciones alternativas
        r = np.zeros(g.num_actions(self.agent), dtype=float)
        for action in g.action_iter(self.agent):
            r[action] = max(0.0, u[action] - u[a])

        return r
    
    def regret_matching(self):
        #
        # TODO: calcular curr_policy y actualizar sum_policy
        #

        # Actualizar sum_policy con la cur_policy actual
        self.sum_policy += self.curr_policy

        # Calcular el nuevo valor de cur_policy en base a cum_regrets
        sum = self.cum_regrets.sum()
        if sum > 0:
            for action in self.game.action_iter(self.agent):
                self.curr_policy[action] = max(0, self.cum_regrets[action]) / sum
        else:
            self.curr_policy = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent))

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
