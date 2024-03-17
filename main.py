import math
import pygame
from ROBOT import Environment,Robot,LiDAR

import cv2
import matplotlib.pyplot as plt

# Max  x == y (m), need to be equals
MAP_DIMENSIONS = 50

# the environment graphics
env = Environment(MAP_DIMENSIONS,'world/mapa_2.png')

dt=0
last_time = pygame.time.get_ticks()

done = False

First_run = True
# simulation loop
vel = 0.0
w_vel = 0.0
while not done:
    dt=(pygame.time.get_ticks()-last_time)/1000
    last_time = pygame.time.get_ticks()

    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                vel = 2.0
                w_vel = 0.0
            if event.key == pygame.K_DOWN:
                vel = 0.0
                w_vel = 0.0
            if event.key == pygame.K_LEFT:
                vel = 0.0
                w_vel = 0.5
            if event.key == pygame.K_RIGHT:
                vel = 0.0
                w_vel = -0.5
            if event.key == pygame.K_SPACE:
                # cv2.imshow('Mapa',mapa_)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                plt.imshow(mapa_, cmap='gray')
                plt.colorbar()  # Adicionar uma barra de cores para visualização
                plt.show()

    if First_run:
        env.reset(dt)
        First_run = False

    # Robot kinematics [v (m/s), w(rad/s), dt]
    reward, done, obs, mapa_ = env.step(vel,w_vel,dt,1)
    
    # print(done)

    pygame.display.update()