import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Ball, Wall
from minigrid.minigrid_env import MiniGridEnv

class Pellet(Ball):
    def __init__(self):
        super().__init__(color='yellow')
    
    def can_overlap(self):
        #permite pasar por encima
        return True

class PacmanEnv(MiniGridEnv):
    def __init__(self, size=13, num_ghosts=2, agent_view_size=7, **kwargs):
        self.num_ghosts = num_ghosts
        self.ghosts = []
        self.pellets = []
        
        #mision requerida por minigrid
        mission_space = MissionSpace(mission_func=lambda: "survive and eat pellets")
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=300,
            agent_view_size=agent_view_size,
            **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        #diseño simetrico 13x13
        
        #--- esquinas superiores ---
        self.grid.vert_wall(2, 2, length=3)
        self.grid.vert_wall(10, 2, length=3)
        
        #bloque horizontal arriba
        self.grid.horz_wall(4, 2, length=5)

        #--- centro ---
        #obstaculo central solido
        self.grid.wall_rect(5, 5, 3, 3)

        #--- zona media lateral ---
        self.grid.horz_wall(2, 6, length=2)
        self.grid.horz_wall(9, 6, length=2)

        #--- parte inferior ---
        #coberturas en l invertida
        self.grid.vert_wall(4, 9, length=2)
        self.grid.vert_wall(8, 9, length=2)
        
        #linea final abajo
        self.grid.horz_wall(5, 10, length=3)

        self.place_agent()

        #generar pellets
        self.pellets = []
        #12 pellets distribuidos
        for _ in range(12): 
            pellet = Pellet()
            self.place_obj(pellet, max_tries=100)
            self.pellets.append(pellet)

        #generar fantasmas
        self.ghosts = []
        for _ in range(self.num_ghosts):
            ghost = Ball(color='red')
            self.place_obj(ghost, max_tries=100)
            self.ghosts.append(ghost)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        #penalizacion por tiempo
        reward -= 0.01 

        #logica comida
        current_cell = self.grid.get(self.agent_pos[0], self.agent_pos[1])

        if isinstance(current_cell, Pellet):
            reward += 1.0 
            self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
            if current_cell in self.pellets:
                self.pellets.remove(current_cell)

        #condicion victoria
        if len(self.pellets) == 0:
            terminated = True
            reward += 10.0 

        #logica fantasmas
        for ghost in self.ghosts:
            if ghost.cur_pos is None: continue 
            
            gx, gy = ghost.cur_pos
            possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            np.random.shuffle(possible_moves) 

            for dx, dy in possible_moves:
                nx, ny = gx + dx, gy + dy
                
                #check limites
                if not (0 <= nx < self.grid.width and 0 <= ny < self.grid.height):
                    continue
                
                target_cell = self.grid.get(nx, ny)
                
                #movimiento valido si es vacio o es el agente
                can_move = (target_cell is None) or (np.array_equal((nx, ny), self.agent_pos))
                
                if can_move:
                    self.grid.set(gx, gy, None) 
                    self.grid.set(nx, ny, ghost) 
                    ghost.cur_pos = (nx, ny)
                    break 

        #logica muerte
        for ghost in self.ghosts:
            if np.array_equal(self.agent_pos, ghost.cur_pos):
                reward -= 10.0
                terminated = True
                break

        return obs, reward, terminated, truncated, info