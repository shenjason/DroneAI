import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
import pygame
import pymunk
import pymunk.pygame_util
import math
import random


WIDTH, HEIGHT = 800, 600

GRAVITY = (0, 981)
FPS = 60
DT = 1/FPS

BLUE = (0, 0, 255, 0)
BLACK = (0, 0, 0, 0)
RED = (255, 0, 0, 0)
YELLOW = (255, 255, 0, 0)
FIRERED = (169, 66, 63, 0)

PICKUPDELAY = [1, 1, 1, 0.5, 0.5, 0.3, 0.2, 0.1, 0.1]

def distance(pos1, pos2):
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

def to2pi(x):
    if (x<0):
        return 2*math.pi-(-x % (2*math.pi))
    return (x % (2*math.pi))

def lint(a, b, t):
    return a + (b-a) * t 

def blendcolor(color1, color2, t):
    return (lint(color1[0], color2[0], t), lint(color1[1], color2[1], t), lint(color1[2], color2[2], t), 0)


def anglediff(a1, a2):
    a1 = to2pi(a1)
    a2 = to2pi(a2)

    dif1 = a2 - a1
    dif2 = (2*math.pi - a1) + a2
    dif3 = -(2*math.pi - a2) - a1
    if (abs(dif1) < abs(dif2)):
        if (abs(dif1) < abs(dif3)):
            return dif1
        else:
            return dif3
    else:   
        if (abs(dif2) < abs(dif3)):
            return dif2
        else:
            return dif3



class GameEnv(Env):

    def __init__(self, visualization=True, interactable=False, cantimeout=True):
        super().__init__()
        #             AI
        #           Actions
        # LeftMotorPower  RightMotorPower  LeftMotorTurnPower  RightMotorTurnPower
        #           Inputs
        # PosX PosY MainBodyVelX MainBodyVelY  GlobalLeftMotorDir GlobalRightMotorDir GlobalShaftDir GlobalShaftAngularAccel DXtoTarget DYtoTarget
        self.action_space = Box(low=np.array([0, 0, -1, -1]), 
                                high=np.array([1, 1, 1, 1]), dtype=np.float64)
        self.observation_space = Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, 
                                                   -1, -1,
                                                   -np.inf, -np.inf, -np.inf]), 
                                     high=np.array([np.inf, np.inf, np.inf, np.inf, 
                                                    1, 1,
                                                    np.inf, np.inf, np.inf]), dtype=np.float64)
        
        self.Target = (WIDTH/2 + random.randint(-100, 100), HEIGHT/4)
        self.targetspicked = 0
        self.cantimeout = cantimeout

        #Game
        self.space = pymunk.Space()
        self.screen = None
        self.render_mode = None
        if visualization: self.render_mode = "human"
        self.Clock = pygame.time.Clock()
        self.space.gravity = GRAVITY
        self.space.damping = 0.8
        self.interactable = interactable

     
        self.drone = Drone(self.space, WIDTH/2, HEIGHT/2)

        self.TargetTicks = 0
        self.timepasted = 0

    def get_state(self):
        dx = self.Target[0] - self.drone.MainBody.x()
        dy = self.Target[1] - self.drone.MainBody.y()


        state = np.array([self.drone.MainBody.x(),
                          self.drone.MainBody.y(), 
                          self.drone.MainBody.vx(),
                          self.drone.MainBody.vy(),
                          math.sin(self.drone.Shaft.get_rot()),
                          math.cos(self.drone.Shaft.get_rot()),
                          self.drone.Shaft.body.angular_velocity,
                          dx, dy])

        return state
        
    
    def InBounds(self, x, y):
        if x < -0.5*WIDTH: return False
        if y < -0.5*HEIGHT: return False
        if x > 1.5*WIDTH: return False
        if y > 1.5*HEIGHT: return False
        return True
        

    def step(self, action):
        finished = False
        self.timepasted += DT

        if self.interactable: self.Target = pygame.mouse.get_pos()

        self.drone.ThrustLeft(action[0], (self.screen != None))
        self.drone.ThrustRight(action[1], (self.screen != None))
        self.drone.SetTargetLeftMotor(action[2])
        self.drone.SetTargetRightMotor(action[3])

        pd = distance(self.drone.MainBody.get_pos(), self.Target)

        self.drone.step()
        self.space.step(DT)

        state = self.get_state()

        cd = distance(self.drone.MainBody.get_pos(), self.Target)

        reward = 0
        reward += max(-1, min((pd - cd) * 0.1, 1))

        if self.targetspicked >= len(PICKUPDELAY): pickupdelay = 0.05
        else: pickupdelay = PICKUPDELAY[self.targetspicked]

        if cd <= 20:
            reward += (0.5) / (pickupdelay)
            self.TargetTicks += DT

        if cd <= 10:
            reward += (1) / (pickupdelay)

        if self.TargetTicks >= pickupdelay:
            if not self.interactable: self.PickTarget()
            self.TargetTicks = 0
            self.targetspicked += 1
        

        tilt = abs(anglediff(self.drone.Shaft.get_rot(), 0))

        #or tilt > (2*math.pi)/3
        if not self.InBounds(self.drone.MainBody.x(), self.drone.MainBody.y()) : finished = True

        truncated = False
        if self.timepasted > 30 and not self.interactable and self.cantimeout: truncated = True
        
        if tilt > math.pi/3: reward -= 0.8

        reward += 0.1

        if (self.render_mode == "human"): self.render()

        return state, reward, finished, truncated, {}
        


    def render(self, render_mode="None"):
        if self.screen == None: return
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.close()
                return
        
        self.screen.fill(BLACK)
        pygame.draw.circle(self.screen, YELLOW, self.Target, 10)
        self.drone.drawEffects(self.screen)
        self.space.debug_draw(pymunk.pygame_util.DrawOptions(self.screen))
        self.Clock.tick(FPS)

        pygame.display.update()
        
    def PickTarget(self):
        targetX = random.randint(50, WIDTH-50)
        targetY = random.randint(50, HEIGHT-50)
        for i in range(20):
            if distance((targetX, targetY), self.drone.MainBody.get_pos()) > 100: break
            targetX = random.randint(50, WIDTH-50)
            targetY = random.randint(50, HEIGHT-50)
        
        self.Target = (targetX, targetY)


    def reset(self, seed=None, options=None):
        if seed != None: random.seed(seed)
        self.timepasted = 0
        self.targetspicked = 0


        self.drone.remove()
        self.drone = Drone(self.space, WIDTH/2, HEIGHT/2)

        self.TargetTicks = 0
        self.Target = (WIDTH/2 + random.randint(-300, 300), HEIGHT/2 + random.randint(-300, 300))

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("DroneAI 2.0")
            
        return self.get_state(), {}
    
    def close(self):
        pygame.quit()

    


class Part():
    MAXJOINTFORCE = math.inf
    def ShapeKeyToShape(key, body, args):
        if (key == "circle"): return pymunk.Circle(body, args[0])
        if (key == "rect"): return pymunk.Poly.create_box(body, args[0])
        return None
    
    def __init__(self, shape, shapeargs, x=WIDTH/2, y=HEIGHT/2, mass=1, color=BLUE):
        self.joints = []
        self.body = pymunk.Body()
        self.body.position = (x, y)

        self.shape = Part.ShapeKeyToShape(shape, self.body, shapeargs)
        self.shape.color = color
        self.shape.mass = mass
        self.shape.elasticity = 0.4
        self.shape.friction = 0.4
    
    def ScrewPart(self, otherpart, anchorself=(0, 0), anchorother=(0 , 0)):
        joint1 = pymunk.PinJoint(self.body, otherpart.body, anchorself, anchorother)
        joint2 = pymunk.RotaryLimitJoint(self.body, otherpart.body, 0, 0)

        joint1.max_force = Part.MAXJOINTFORCE
        joint2.max_force = Part.MAXJOINTFORCE
        joint1.distance = 0
        joint1.collide_bodies = False
        joint2.collide_bodies = False
        joint1.error_bias = pow(1 - 0.9, 60)
        joint2.error_bias = pow(1 - 0.9, 60)
        self.joints.append(joint2)
        self.joints.append(joint1)
    
    def PinPart(self, otherpart, anchorself=(0, 0), anchorother=(0 , 0)):
        joint1 = pymunk.PivotJoint(self.body, otherpart.body, anchorself, anchorother)

        joint1.max_force = Part.MAXJOINTFORCE
        joint1.collide_bodies = False
        joint1.error_bias = pow(1 - 0.9, 60)
        self.joints.append(joint1)

    def AttachMotorControl(self, otherpart):
        motor = pymunk.SimpleMotor(self.body, otherpart.body, 0)
        motor.error_bias = pow(1 - 0.9, 60)
        motor.max_force = Part.MAXJOINTFORCE
        motor.collide_bodies = False
        motor.rate = 0
        self.joints.append(motor)
        return motor

    def init(self, space):
        space.add(self.body, self.shape)
        for j in self.joints: space.add(j)

    def remove(self, space):
        for j in self.joints: space.remove(j)
        space.remove(self.shape, self.body)


    def get_pos(self):
        return self.body.position
    
    def get_vel(self):
        return self.body.velocity
    
    def vx(self):
        return self.body.velocity[0]
    def vy(self):
        return self.body.velocity[1]
    
    def x(self):
        return self.body.position[0]

    def y(self):
        return self.body.position[1]
    
    def get_rot(self):
        return self.body.angle
    
    def add_force(self, force):
        self.body.apply_force_at_world_point(force, self.get_pos())

class DroneFire():
    def __init__(self, x, y, dir, vel, r):
        self.ir = r
        self.r = r
        self.x = x
        self.y = y
        self.vx = math.cos(dir) * vel
        self.vy = math.sin(dir) * vel

    def draw(self, screen):
        if (self.r<=0): return
        color = blendcolor(blendcolor(YELLOW, RED, math.sqrt(1-(self.r/self.ir))), BLACK, (1-(self.r/self.ir))**2)
        pygame.draw.circle(screen, color, (self.x, self.y), self.r)
        self.x += self.vx
        self.y += self.vy
        self.r -= 0.1
    
    def is_dead(self):
        return (self.r<=1)


class Drone():
    MAXTHRUST = 800
    TURNPOWER = 0.4

    def __init__(self, space, x, y):
        self.space = space
        
        self.MainBody = Part("circle", [12], x, y, 0.1, BLUE)
        self.Shaft = Part("rect", [[60, 10]], x, y, 0.1, RED)

        self.LeftMotor = Part("rect", [[10, 18]], x-30, y, 0.1, BLUE)
        self.RightMotor = Part("rect", [[10, 18]], x+30, y, 0.1, BLUE)

        self.MainBody.PinPart(self.Shaft, (0, 0), (0, 0))
        self.Shaft.PinPart(self.LeftMotor, (-30, 0), (0, 0))
        self.Shaft.PinPart(self.RightMotor, (30, 0), (0, 0))

        self.LeftMotorRot = 0
        self.RightMotorRot = 0

        self.Shaft.init(self.space)
        self.MainBody.init(self.space)
        self.LeftMotor.init(self.space)
        self.RightMotor.init(self.space)

        self.TargetLeftMotorRot = 0
        self.TargetRightMotorRot = 0

        self.Particles = []
        
        
    
    def drawEffects(self, screen):
        for i, par in enumerate(self.Particles):
            if par.is_dead(): self.Particles.pop(i)
            par.draw(screen)

    def step(self):
        # self.Shaft.body.angular_velocity *= 0.8
        self.RightMotorRot += (self.TargetRightMotorRot - self.RightMotorRot) * Drone.TURNPOWER
        self.LeftMotorRot += (self.TargetLeftMotorRot - self.LeftMotorRot) * Drone.TURNPOWER
        self.RightMotorRot = max(-math.pi/3, min(math.pi/3, self.RightMotorRot))
        self.LeftMotorRot = max(-math.pi/3, min(math.pi/3, self.LeftMotorRot))
        # self.Shaft.body.angular_velocity = min(max(self.Shaft.body.angular_velocity, -20), 20)
        self.LeftMotor.body.angle = self.LeftMotorRot + self.Shaft.get_rot()
        self.RightMotor.body.angle = self.RightMotorRot + self.Shaft.get_rot()
        

    def ThrustLeft(self, power, par=True):
        power = max(0, min(1, power))
        
        ThrustDir = self.LeftMotor.get_rot() - math.pi/2
        fx = Drone.MAXTHRUST * power * math.cos(ThrustDir)
        fy = Drone.MAXTHRUST * power * math.sin(ThrustDir)

        self.LeftMotor.add_force((fx, fy))
        if not par: return
        for i in range(math.ceil(10*power)):
            angleoffset = ((random.random()*2)-1) * math.pi/12 * power + math.pi
            self.Particles.append(DroneFire(self.LeftMotor.get_pos()[0], self.LeftMotor.get_pos()[1], ThrustDir + angleoffset, power*12+random.randint(1, 3), power*3+random.random()*2))
            

    def ThrustRight(self, power, par=True):
        power = max(0, min(1, power))
        
        ThrustDir = self.RightMotor.get_rot() - math.pi/2
        fx = Drone.MAXTHRUST * power * math.cos(ThrustDir)
        fy = Drone.MAXTHRUST * power * math.sin(ThrustDir)

        self.RightMotor.add_force((fx, fy))
        if not par: return
        for i in range(math.ceil(10*power)):
            angleoffset = ((random.random()*2)-1) * math.pi/12 * power + math.pi
            self.Particles.append(DroneFire(self.RightMotor.get_pos()[0], self.RightMotor.get_pos()[1], ThrustDir + angleoffset, power*12+random.randint(1, 3), power*3+random.random()*2))


    def SetTargetLeftMotor(self, dir):
        self.TargetLeftMotorRot = dir * math.pi/3

    def SetTargetRightMotor(self, dir):
        self.TargetRightMotorRot = dir * math.pi/3

    def remove(self):
        self.MainBody.remove(self.space)
        self.Shaft.remove(self.space)
        self.LeftMotor.remove(self.space)
        self.RightMotor.remove(self.space)