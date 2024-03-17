import pygame
import math
import numpy as np
import random
from collections import deque


import cv2

from grid_map import *
from map_utils import *

MAP_SCALE=10

def distance(point1:float , point2:float) -> float:
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1-point2)

def _difference_two_angles(angle1: float, angle2: float) -> float:
    diff = (angle1 - angle2) % (math.pi * 2)
    if diff >= math.pi:
        diff -= math.pi * 2
    return diff

def _orientation_robot_to_end(point1: float, point2: float) -> float:
    x = point1[0] - point2[0]
    y = point1[1] - point2[1]
    return math.atan2(y, x)

# Função para converter de metros para pixels
def meters_to_pixels(meters):
    return int(meters * MAP_SCALE)

def pixels_to_meter(pixels):
    return pixels/MAP_SCALE

class Robot:
    def __init__(self, startpos):

        # self.m2p = 3779.52  # from meters to pixels
        self.m2p = 10  # from meters to pixels
        # robot axis dist
        self.w = 1

        self.x = startpos[0]*self.m2p
        self.y = startpos[1]*self.m2p
        self.heading = 0

        self.min_obs_dist = 1
        self.count_down = 5 # seconds

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

    def get_pose(self):
        pose = [pixels_to_meter(self.x),pixels_to_meter(self.y),self.heading]

        return pose

    def get_vel(self):

        linear_velocity = self.linear_velocity + random.uniform(-0.02, 0.02)
        angular_velocity = self.angular_velocity + random.uniform(-0.02, 0.02)

        return linear_velocity, angular_velocity

    def check_collision(self, point_cloud):
        closest_obs = None
        dist = np.inf

        if len(point_cloud) > 1:
            for point in point_cloud:
                if dist > distance([self.x,self.y], point):
                    dist = distance([self.x,self.y], point)
                    closest_obs = (point,dist)

            if closest_obs[1] < self.min_obs_dist*self.m2p:
                return True
            else:
                return False

        else:
            return False


    def check_collision2(self, point_cloud, obs_dist):
        closest_obs = None
        dist = np.inf

        if len(point_cloud) > 1:
            for point in point_cloud:
                if dist > distance([self.x,self.y], point):
                    dist = distance([self.x,self.y], point)
                    closest_obs = (point,dist)

            if closest_obs[1] < obs_dist*self.m2p:
                return True
            else:
                return False

        else:
            return False

    def step(self, v, w, dt):
        b = 0.5

        self.vr = v*self.m2p + (w*self.m2p*b / 2)
        self.vl = v*self.m2p - (w*self.m2p*b / 2)

        self.x += ((self.vl+self.vr)/2) * math.cos(self.heading) * dt
        self.y -= ((self.vl+self.vr)/2) * math.sin(self.heading) * dt
        self.heading+= (self.vr - self.vl) / self.w * dt

        if self.heading>2*math.pi or  self.heading<-2*math.pi:
            self.heading = 0

        self.linear_velocity = v
        self.angular_velocity = w


class Graphics:
    def __init__(self, dimentions, robot_img_path, map_img_path):
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
        map_img2 = pygame.image.load(map_img_path)
        self.map_img = pygame.transform.scale(map_img2, (500, 500))

        # dimentions
        self.height = self.map_img.get_width()
        self.width = self.map_img.get_height()
        self.map_scale = self.width/dimentions

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

    def draw_target(self, pos):
        n_pos = (meters_to_pixels(pos[0]),meters_to_pixels(pos[1]))
        pygame.draw.circle(self.map, self.green, n_pos, 5)


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
                else:
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

        # for map
        self.P_prior = 0.5
        self.P_occ = 0.9
        self.P_free = 0.3
        self.gridMap = GridMap(X_lim = [0, self.map_height], 
                  Y_lim = [0, self.map_width], 
                  resolution = MAP_SCALE, 
                  p = self.P_prior)
        

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



    ########################################
    '''         Compute Map         '''
    ########################################
    def get_map(self, lidar_scan, robot_pose):

        map_increase = 0.0

        lidar = lidar_scan
        pose = robot_pose
        # pmap,map_increase  = self.Map.update_map(lidar,pose)

        ang_step = 0.04759988999999998
        angle_min = -2.3561944902
        angle_max = 2.3561944902
        range_min = 0.1
        range_max = 5
        angles = np.arange(angle_min,angle_max,ang_step)

        alpha = 1.0  # delta for max rang noise

        # Lidar measurements in X-Y plane
        distances_x, distances_y = lidar_scan_xy(lidar, angles, pose[0], pose[1], pose[2])

        # x1 and y1 for Bresenham's algorithm
        x1, y1 = self.gridMap.discretize(pose[0], pose[1])

        # for BGR image of the grid map
        X2 = []
        Y2 = []

        for (dist_x, dist_y, dist) in zip(distances_x, distances_y, lidar):

            # x2 and y2 for Bresenham's algorithm
            x2, y2 = self.gridMap.discretize(dist_x, dist_y)

            # draw a discrete line of free pixels, [robot position -> laser hit spot)
            for (x_bres, y_bres) in bresenham(self.gridMap, x1, y1, x2, y2):

                self.gridMap.update(x = x_bres, y = y_bres, p = self.P_free)

            # mark laser hit spot as ocuppied (if exists)
            if dist < range_max - alpha:
                
                self.gridMap.update(x = x2, y = y2, p = self.P_occ)

            # for BGR image of the grid map
            X2.append(x2)
            Y2.append(y2)

        # converting grip map to BGR image
        # bgr_image = self.gridMap.to_BGR_image()

        gray_image, map_increase = self.gridMap.to_grayscale_image()


        resized_image = cv2.resize(gray_image,(500, 500),interpolation = cv2.INTER_NEAREST)


        rotated_image = cv2.rotate(src = gray_image, 
                       rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)


        # cv2.imshow('Mapa',gray_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        pmap = rotated_image

        return pmap, map_increase



class Environment:

    def __init__(self, dimentions, map_img_path):
        self.gfx = Graphics(dimentions,'DDR.png','world/mapa_2.png')
        start = (15,10)
        self.robot = Robot(start)
        sensor_range = 5, math.radians(135)
        self.lidar = LiDAR(sensor_range, self.gfx.map)
        self.point_cloud = []
        self.scan = []
        self.dt = 0.01
        self.target_pos = (0,0)

        obs, self.free_areas = self.gfx.get_mapObsFree()

        self.last_map_info = 0.0
        self.delta_map = deque(maxlen=50)
        self.map_penality = 0

    def reset(self, dt):
        done = True
        while done:
            new_pixel = random.choice(self.free_areas)
            new_pos = (pixels_to_meter(new_pixel[0]),pixels_to_meter(new_pixel[1]))
            self.robot = Robot(new_pos)
            _,done,_,_ = self.step(0,0,dt,0)
            done = self.robot.check_collision2(self.point_cloud, 3)
            t_pos = random.choice(self.free_areas)
            self.target_pos = (pixels_to_meter(t_pos[0]),pixels_to_meter(t_pos[1]))


    def step(self, liner_vel, ang_vel,dt, num_steps):

        # update map
        self.gfx.map.blit(self.gfx.map_img,(0,0))

        # robot step
        self.robot.step(liner_vel,ang_vel,dt)

        # extract point_cloud and lidar scan
        self.point_cloud, self.scan = self.lidar.sense_obstacles(self.robot.x, self.robot.y, self.robot.heading)

        # draw liDAR
        self.gfx.draw_sensor_data(self.point_cloud)

        # draw Robot
        self.gfx.draw_robot(self.robot.x, self.robot.y, self.robot.heading)
        done = self.robot.check_collision(self.point_cloud)

        # draw target
        self.gfx.draw_target(self.target_pos)

        # define observation
        robot_pose = self.robot.get_pose()
        dist_to_goal = distance(robot_pose[0:2],self.target_pos)
        angle_to_goal = _orientation_robot_to_end(self.target_pos,robot_pose[0:2])
        diff_angle = _difference_two_angles(robot_pose[2], angle_to_goal)
        lin_vel, ang_vel = self.robot.get_vel()

        observation = self.scan[:,0].tolist()
        observation.append(dist_to_goal)
        observation.append(diff_angle)
        observation.append(lin_vel)
        observation.append(ang_vel)

        # print(self.point_cloud)

        mapa_, map_info = self.lidar.get_map(self.scan[:,0].tolist(), robot_pose)

        reward, done = self.calculate_reward(robot_pose, done, lin_vel, ang_vel, self.scan, map_info, num_steps)

        return reward, done, observation, mapa_
        


    def calculate_reward(self, robot_pose, env_done, linear_velocity, angular_velocity,
                         Lidar_scan, map_info, num_steps):
        done = False
        reward = 0.0

        # """
        #     Distancy
        # """
        distance_robot_to_end = distance(robot_pose[0:2], self.target_pos)
 
        # """
        #     Map Data Not Increasing - Prevent Local Minima
        # """
        self.delta_map.append(map_info - self.last_map_info)
        self.last_map_info = map_info
        # if(np.mean(self.delta_map) == 0):
        #     reward = -self.map_penality
        #     self.map_penality += 1

        
        # """
        #     Collision, Conclusion, Always pen
        # """
        if env_done:
            reward = -200
            done = True
        elif distance_robot_to_end < 0.5:
            reward = 100
            done = True
        elif num_steps > 500:
            reward = -200
            done = True
        else: 
            G = np.mean(self.delta_map) + 0.1
            D = distance_robot_to_end + 0.0001  # prevent division by 0
            KG = 0.2
            # KG = 1
            L = Lidar_scan[int(round(len(Lidar_scan)/2.0)),0] # Lidar measure in robot orientation (middle of the measures)
            K3 = 1

            # OUR Method (PAPER) - test 1
            # r = KG*(G/D) + K3*L + (linear_velocity - abs(angular_velocity)) 

            # New version - test2
            r = -KG*(D/G)

            # print("GD = ", min(KG*(D/G),10))
            # print("L = ", (1/(L+0.0001)))
            # print("vel = ", 1/(0.0001+(linear_velocity - abs(angular_velocity))))
            # print("reward = ", r)

            reward = r
            # reward = min(0, -D + r)
            # reward = -1/r

            # txtfile = open(txt_file,"a")
            # txtfile.write("D = " + str(-D) + "; r = " + str(r) + "; reward = " + str(reward) + "\n" )
            # txtfile.close()


        # self._robot_x_last = robot_x
        # self._robot_y_last = robot_y
        # self._robot_orientation_last = robot_orientation

        return reward, done









































