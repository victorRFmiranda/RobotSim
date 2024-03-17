import math
import pygame
from ROBOT import Environment,Robot,LiDAR


# Max  x == y (m), need to be equals
MAP_DIMENSIONS = 50

# the environment graphics
env = Environment(MAP_DIMENSIONS,'world/mapa_2.png')

dt=0
last_time = pygame.time.get_ticks()

running = True

First_run = True
# simulation loop
while running:
    dt=(pygame.time.get_ticks()-last_time)/1000
    last_time = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                env.reset(dt)

    if First_run:
        env.reset(dt)
        First_run = False

    # Robot kinematics [v (m/s), w(rad/s), dt]
    running, obs = env.step(0,0.5,dt)
        

    pygame.display.update()