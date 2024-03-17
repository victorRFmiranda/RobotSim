
import pygame
import math
import numpy as np

MAP_SCALE=10

def distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1-point2)

# Função para converter de metros para pixels
def meters_to_pixels(meters):
    return int(meters * MAP_SCALE)

def pixels_to_meter(pixels):
    return pixels/MAP_SCALE

class Robot:
    def __init__(self, startpos, width):

        # self.m2p = 3779.52  # from meters to pixels
        self.m2p = 10  # from meters to pixels
        # robot dims
        self.w = width

        self.x = startpos[0]*self.m2p
        self.y = startpos[1]*self.m2p
        self.heading = 0

        self.min_obs_dist = 1
        self.count_down = 5 # seconds

    def get_pose(self):
        pose = [pixels_to_meter(self.x),pixels_to_meter(self.y),self.heading]

        return pose

    def check_collision(self, point_cloud, dt):
        closest_obs = None
        dist = np.inf

        if len(point_cloud) > 1:
            for point in point_cloud:
                if dist > distance([self.x,self.y], point):
                    dist = distance([self.x,self.y], point)
                    closest_obs = (point,dist)

            if closest_obs[1] < self.min_obs_dist*self.m2p:
                return False
            else:
                return True

        else:
            return True

    def kinematics(self, v, w, dt):
        b = 0.5

        self.vr = v*self.m2p + (w*self.m2p*b / 2)
        self.vl = v*self.m2p - (w*self.m2p*b / 2)

        self.x += ((self.vl+self.vr)/2) * math.cos(self.heading) * dt
        self.y -= ((self.vl+self.vr)/2) * math.sin(self.heading) * dt
        self.heading+= (self.vr - self.vl) / self.w * dt

        if self.heading>2*math.pi or  self.heading<-2*math.pi:
            self.heading = 0


class Graphics:
    def __init__(self, map_scale, dimentions, robot_img_path, map_img_path):
        pygame.init()
        # COLORS
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.yel = (255, 255, 0)

        # ---------MAP---------

        # load imgs
        self.robot = pygame.image.load(robot_img_path)
        self.map_img = pygame.image.load(map_img_path)

        # dimentions
        self.height, self.width = dimentions
        self.map_scale = map_scale

        # window settings
        pygame.display.set_caption("Reinforcement Learning")
        self.map = pygame.display.set_mode((self.width,self.height))
        self.map.blit(self.map_img, (0,0))

    def draw_robot(self, x, y, heading):
        rotated = pygame.transform.rotozoom(self.robot, math.degrees(heading), 1)
        rect = rotated.get_rect(center=(x, y))
        self.map.blit(rotated, rect)

    def draw_sensor_data(self, point_cloud):
        for point in point_cloud:
            pygame.draw.circle(self.map, self.red, point, 3, 0)


    def get_mapObsFree(self):
        obstacles = []
        free_areas = []
        # Obtém os pixels da imagem como uma matriz numpy
        pixels = np.array(pygame.surfarray.array2d(self.map_img))
        # Percorra os pixels da imagem para criar obstáculos e áreas livres
        for x in range(self.width):
            for y in range(self.height):
                if pixels[x][y] == -1:  # Verifique se o pixel é preto (0)
                    free_areas.append((x, y))
                    # obstacles.append((x, y))
                else:
                # elif pixels[x][y] == 255:  # Verifique se o pixel é branco (255)
                    # free_areas.append((x, y))
                    obstacles.append((x, y))

        return  obstacles, free_areas
            

class LiDAR:

    def __init__(self, sensor_range, map):
        self.sensor_range = (meters_to_pixels(sensor_range[0]), sensor_range[1])
        # self.sensor_range = sensor_range
        # print(self.sensor_range)
        self.map_width, self.map_height= pygame.display.get_surface().get_size()
        self.map = map
        self.numberLines = 100
        self.lidar_scan = sensor_range[0]*np.ones((self.numberLines,2))

    def sense_obstacles(self, x, y, heading):
        obstacles = []
        x1, y1 = x, y
        start_angle = heading - self.sensor_range[1]
        finish_angle = heading + self.sensor_range[1]
        j=0
        for angle in np.linspace(start_angle, finish_angle , self.numberLines, True):
            x2 = x1 + self.sensor_range[0] * math.cos(angle)
            y2 = y1 - self.sensor_range[0] * math.sin(angle)
            self.lidar_scan[j,1] = angle
            for i in range(0, 100):
                u = i / 100
                x = int(x2 * u + x1 * (1 - u))
                y = int(y2 * u + y1 * (1 - u))
                if 0 < x < self.map_width and 0 < y < self.map_height:
                    color = self.map.get_at((x, y))
                    self.map.set_at((x, y), (0, 208, 255)) ### Preciso add as distancias em um vetor tambem
                    if (color[0], color[1], color[2]) == (0, 0, 0):
                        obstacles.append([x,y])
                        self.lidar_scan[j] = [pixels_to_meter(distance([x,y],[x1,y1])),angle]
                        break

            j+=1
        return obstacles, self.lidar_scan



















































