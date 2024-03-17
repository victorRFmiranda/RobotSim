import math
import pygame
from ROBOT import Graphics,Robot,LiDAR


# MAP_DIMENSIONS = (600, 1200)
MAP_DIMENSIONS = (500, 500)
MAP_SCALE = 10


# the environment graphics
gfx = Graphics(MAP_SCALE, MAP_DIMENSIONS,'DDR.png','world/mapa.png')

# the robot
start = (10,10)
robot=Robot(start,10)


# the sensor   (range [m], deg)
sensor_range = 5, math.radians(135)
laser = LiDAR(sensor_range, gfx.map)

dt=0
last_time = pygame.time.get_ticks()

running = True

obstacles, freeSpace = gfx.get_mapObsFree()

# simulation loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    dt=(pygame.time.get_ticks()-last_time)/1000
    last_time = pygame.time.get_ticks()

    # start world
    gfx.map.blit(gfx.map_img,(0,0))

    # Robot kinematics [v (m/s), w(rad/s), dt]
    robot.kinematics(0,0.5,dt)

    # extract point_cloud and lidar scan
    point_cloud, scan = laser.sense_obstacles(robot.x, robot.y, robot.heading)

    # draw liDAR
    gfx.draw_sensor_data(point_cloud)

    # draw Robot
    gfx.draw_robot(robot.x, robot.y, robot.heading)

    # Stop sim if collision is True
    running = robot.check_collision(point_cloud, dt)


    pygame.display.update()






































