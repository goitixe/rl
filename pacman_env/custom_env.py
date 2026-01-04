import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Ball
from pacman_env.pacman import PacmanEnv

#diseño del mapa 19x19 simetrico
LABERINTO = [
    "WWWWWWWWWWWWWWWWWWW",
    "W........W........W",
    "W.WW.WWW.W.WWW.WW.W",
    "W.WW.WWW.W.WWW.WW.W",
    "W.................W",
    "W.WW.W.WWWWW.W.WW.W",
    "W....W...W...W....W",
    "WWWW.WWW.W.WWW.WWWW",
    "W.......GGG.......W",
    "WWWW.W.WWWWW.W.WWWW",
    "W....W...P...W....W",
    "W.WW.W.WWWWW.W.WW.W",
    "W.................W",
    "W.WW.WWW.W.WWW.WW.W",
    "W..W.....W.....W..W",
    "WW.W.W.WWWWW.W.W.WW",
    "W....W.......W....W",
    "W.WWWWWWWWWWWWWWW.W",
    "WWWWWWWWWWWWWWWWWWW"
]

class CustomPellet(Ball):
    def __init__(self):
        super().__init__(color='yellow')
    def can_overlap(self):
        return True

class ProPacmanEnv(PacmanEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        ghost_spawns = []
        self.pellets = [] 

        for y, row in enumerate(LABERINTO):
            for x, char in enumerate(row):
                if char == 'W':
                    self.grid.set(x, y, Wall())
                elif char == 'G':
                    ghost_spawns.append((x, y))
                    self.grid.set(x, y, None) 
                elif char == 'P':
                    #correccion critica: definir pos actual y start
                    self.agent_start_pos = (x, y)
                    self.agent_start_dir = 0
                    self.agent_pos = (x, y)
                    self.agent_dir = 0
                    self.grid.set(x, y, None)
                else:
                    pellet = CustomPellet()
                    self.grid.set(x, y, pellet)
                    self.pellets.append(pellet)

        self.ghosts = []
        for i in range(self.num_ghosts):
            gx, gy = ghost_spawns[i % len(ghost_spawns)]
            ghost = Ball(color='red')
            self.grid.set(gx, gy, ghost)
            ghost.cur_pos = (gx, gy)
            self.ghosts.append(ghost)

        self.mission = "eat all pellets"

    def step(self, action):
        obs, reward, terminated, truncated, info = super(PacmanEnv, self).step(action)
        
        #reward shaping
        reward -= 0.01 
        
        ax, ay = self.agent_pos
        current_cell = self.grid.get(ax, ay)

        if isinstance(current_cell, CustomPellet):
            reward += 1.0 
            self.grid.set(ax, ay, None)
            if current_cell in self.pellets:
                self.pellets.remove(current_cell)

        if len(self.pellets) == 0:
            terminated = True
            reward += 10.0 

        for ghost in self.ghosts:
            if ghost.cur_pos is None: continue
            gx, gy = ghost.cur_pos
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            np.random.shuffle(moves)
            for dx, dy in moves:
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    target = self.grid.get(nx, ny)
                    if target is None or isinstance(target, CustomPellet) or np.array_equal((nx, ny), self.agent_pos):
                        self.grid.set(gx, gy, None)
                        self.grid.set(nx, ny, ghost)
                        ghost.cur_pos = (nx, ny)
                        break

        for ghost in self.ghosts:
            if np.array_equal(self.agent_pos, ghost.cur_pos):
                reward -= 10.0
                terminated = True
                break

        return obs, reward, terminated, truncated, info