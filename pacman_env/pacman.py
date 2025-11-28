import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Ball, Wall
from minigrid.minigrid_env import MiniGridEnv

#hereda de ball para que sepa dibujarse solo
class Pellet(Ball):
    def __init__(self):
        super().__init__(color='yellow')

    def can_overlap(self):
        #permite pasar por encima
        return True

class PacmanEnv(MiniGridEnv):
    def __init__(self, size=13, num_ghosts=2, **kwargs):
        self.num_ghosts = num_ghosts
        self.ghosts = []
        self.pellets = []
        
        #mision requerida por minigrid
        mission_space = MissionSpace(mission_func=lambda: "survive and eat pellets")
        
        #vision local 7x7
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=200, 
            agent_view_size=3,
            **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        #distribucion uniforme en grid 13x13
        
        #--- FILA SUPERIOR ---
        
        #1. izquierda: pieza L
        self.grid.vert_wall(2, 2, length=3)
        self.grid.set(3, 4, Wall())
        
        #2. centro: pieza barra pequeña
        self.grid.horz_wall(6, 2, length=2)
        
        #3. derecha: pieza cuadrado
        self.grid.wall_rect(9, 2, 2, 2)
        
        #--- FILA MEDIA ---
        
        #4. izquierda: pieza I vertical
        self.grid.vert_wall(2, 6, length=3)
        
        #5. centro: pieza T
        self.grid.horz_wall(5, 6, length=3) 
        self.grid.set(6, 7, Wall())

        #6. derecha: pieza I vertical pegada al borde
        self.grid.vert_wall(10, 6, length=3)

        #--- FILA INFERIOR ---

        #7. izquierda: pieza J
        self.grid.vert_wall(3, 9, length=2)
        self.grid.set(2, 10, Wall())

        #8. centro: pieza guion
        self.grid.horz_wall(6, 10, length=2)

        #9. derecha: pieza Z
        self.grid.set(8, 9, Wall())
        self.grid.set(9, 9, Wall())
        self.grid.set(9, 10, Wall())
        self.grid.set(10, 10, Wall())

        self.place_agent()

        self.pellets = []
        for _ in range(10):
            pellet = Pellet()
            self.place_obj(pellet, max_tries=100)
            self.pellets.append(pellet)

        self.ghosts = []
        for _ in range(self.num_ghosts):
            ghost = Ball(color='red')
            #variable para recordar si pisa comida
            ghost.covered_pellet = None
            self.place_obj(ghost, max_tries=100)
            self.ghosts.append(ghost)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        #logica de comer pellets manual
        agent_x = self.agent_pos[0]
        agent_y = self.agent_pos[1]
        
        current_cell = self.grid.get(agent_x, agent_y)
        
        if isinstance(current_cell, Pellet):
            reward += 1.0
            self.grid.set(agent_x, agent_y, None)
            #borrar de la lista para saber cuando ganar
            if current_cell in self.pellets:
                self.pellets.remove(current_cell)
        
        #condicion de victoria: si no quedan pellets
        if len(self.pellets) == 0:
            terminated = True
            #damos un bonus extra por ganar
            reward += 10.0

        #movimiento de fantasmas manual
        for ghost in self.ghosts:
            if ghost.cur_pos is not None:
                ghost_x = ghost.cur_pos[0]
                ghost_y = ghost.cur_pos[1]
                
                possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                valid_moves = []
                
                for move in possible_moves:
                    next_x = ghost_x + move[0]
                    next_y = ghost_y + move[1]
                    
                    if 0 <= next_x < self.grid.width and 0 <= next_y < self.grid.height:
                        cell = self.grid.get(next_x, next_y)
                        
                        is_obstacle = False
                        
                        #obstaculo si es pared o bola (salvo que sea pellet)
                        if isinstance(cell, Wall) or isinstance(cell, Ball):
                            if not isinstance(cell, Pellet):
                                is_obstacle = True
                        
                        #el agente no es obstaculo para el fantasma
                        if (next_x == agent_x and next_y == agent_y):
                            is_obstacle = False
                        
                        if not is_obstacle:
                            valid_moves.append((next_x, next_y))
                
                if len(valid_moves) > 0:
                    idx = np.random.choice(len(valid_moves))
                    new_pos = valid_moves[idx]
                    
                    #1. restaurar lo que habia debajo
                    if ghost.covered_pellet is not None:
                        self.grid.set(ghost_x, ghost_y, ghost.covered_pellet)
                        ghost.covered_pellet = None
                    else:
                        self.grid.set(ghost_x, ghost_y, None)
                    
                    #2. ver que hay en la nueva casilla y guardarlo si es pellet
                    target_obj = self.grid.get(new_pos[0], new_pos[1])
                    if isinstance(target_obj, Pellet):
                        ghost.covered_pellet = target_obj
                    
                    #3. mover fantasma
                    self.grid.set(new_pos[0], new_pos[1], ghost)
                    ghost.cur_pos = new_pos

        #colision con fantasmas manual
        for ghost in self.ghosts:
            if self.agent_pos[0] == ghost.cur_pos[0] and self.agent_pos[1] == ghost.cur_pos[1]:
                reward = -10.0
                terminated = True

        return obs, reward, terminated, truncated, info