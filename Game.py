"""
Game.py - DroneAI Environment

Defines the Gymnasium reinforcement learning environment for a 2D drone simulation.
Contains the drone physics model (using pymunk), the observation/action spaces,
the reward function, and rendering logic (using pygame).

Do not run this file directly — it is imported by Train.py and Load.py.
"""

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
import pygame
import pymunk
import pymunk.pygame_util
import math
import random


# ── Simulation constants ──────────────────────────────────────────────────────

WIDTH, HEIGHT = 800, 600

GRAVITY = (0, 981)       # Downward gravity (pymunk uses pixel units)
FPS = 60
DT = 1/FPS               # Fixed timestep per frame

# Colors (R, G, B, A)
BLUE = (0, 0, 255, 0)
BLACK = (0, 0, 0, 0)
RED = (255, 0, 0, 0)
YELLOW = (255, 255, 0, 0)
FIRERED = (169, 66, 63, 0)

# Seconds the drone must hover on target before it counts as "picked up".
# Decreases as more targets are collected, making later targets harder.
PICKUPDELAY = [1, 1, 1, 0.5, 0.5, 0.3, 0.2, 0.1, 0.1]


# ── Utility functions ─────────────────────────────────────────────────────────

def distance(pos1, pos2):
    """Euclidean distance between two 2D points."""
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

def to2pi(x):
    """Normalize an angle to the range [0, 2π)."""
    if (x<0):
        return 2*math.pi-(-x % (2*math.pi))
    return (x % (2*math.pi))

def lint(a, b, t):
    """Linear interpolation from a to b by factor t."""
    return a + (b-a) * t

def blendcolor(color1, color2, t):
    """Linearly blend two RGBA colors by factor t."""
    return (lint(color1[0], color2[0], t), lint(color1[1], color2[1], t), lint(color1[2], color2[2], t), 0)


def anglediff(a1, a2):
    """Compute the shortest signed angular difference between two angles."""
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


# ── Gymnasium Environment ─────────────────────────────────────────────────────

class GameEnv(Env):
    """
    A Gymnasium environment that simulates a 2D drone navigating to targets.

    Action space (continuous, 4 values):
        [left_motor_power, right_motor_power, left_motor_tilt, right_motor_tilt]
        - Power: 0 to 1 (proportion of max thrust)
        - Tilt: -1 to 1 (mapped to ±60° relative to the drone body)

    Observation space (continuous, 9 values):
        [pos_x, pos_y, vel_x, vel_y, sin(rotation), cos(rotation),
         angular_velocity, dx_to_target, dy_to_target]
    """

    def __init__(self, visualization=True, interactable=False, cantimeout=True):
        super().__init__()

        # Action space: [left_power, right_power, left_tilt, right_tilt]
        self.action_space = Box(low=np.array([0, 0, -1, -1]),
                                high=np.array([1, 1, 1, 1]), dtype=np.float64)

        # Observation space: [x, y, vx, vy, sin(rot), cos(rot), ang_vel, dx, dy]
        self.observation_space = Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf,
                                                   -1, -1,
                                                   -np.inf, -np.inf, -np.inf]),
                                     high=np.array([np.inf, np.inf, np.inf, np.inf,
                                                    1, 1,
                                                    np.inf, np.inf, np.inf]), dtype=np.float64)

        # Target position and collection tracking
        self.Target = (WIDTH/2 + random.randint(-100, 100), HEIGHT/4)
        self.targetspicked = 0
        self.cantimeout = cantimeout

        # Physics world setup
        self.space = pymunk.Space()
        self.screen = None
        self.render_mode = None
        if visualization: self.render_mode = "human"
        self.Clock = pygame.time.Clock()
        self.space.gravity = GRAVITY
        self.space.damping = 0.8       # Velocity damping (not true air resistance)
        self.interactable = interactable  # If True, target follows the mouse cursor

        # Create the drone in the center of the screen
        self.drone = Drone(self.space, WIDTH/2, HEIGHT/2)

        self.TargetTicks = 0   # How long the drone has hovered on the current target
        self.timepasted = 0    # Total elapsed simulation time

    def get_state(self):
        """Build the 9-element observation vector for the agent."""
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
        """Check if (x, y) is within the extended play area (50% margin around screen)."""
        if x < -0.5*WIDTH: return False
        if y < -0.5*HEIGHT: return False
        if x > 1.5*WIDTH: return False
        if y > 1.5*HEIGHT: return False
        return True


    def step(self, action):
        """
        Advance the simulation by one timestep.

        Returns: (observation, reward, terminated, truncated, info)
        """
        finished = False
        self.timepasted += DT

        # In interactive mode, the target follows the mouse cursor
        if self.interactable: self.Target = pygame.mouse.get_pos()

        # Apply the agent's actions to the drone
        self.drone.ThrustLeft(action[0], (self.screen != None))
        self.drone.ThrustRight(action[1], (self.screen != None))
        self.drone.SetTargetLeftMotor(action[2])
        self.drone.SetTargetRightMotor(action[3])

        # Record distance before physics step for reward shaping
        pd = distance(self.drone.MainBody.get_pos(), self.Target)

        # Step the drone motor interpolation and the physics world
        self.drone.step()
        self.space.step(DT)

        state = self.get_state()

        # Distance after physics step
        cd = distance(self.drone.MainBody.get_pos(), self.Target)

        # ── Reward calculation ────────────────────────────────────────────
        reward = 0

        # Reward for getting closer to the target (clamped to [-1, 1])
        reward += max(-1, min((pd - cd) * 0.1, 1))

        # Determine required hover time for current difficulty level
        if self.targetspicked >= len(PICKUPDELAY): pickupdelay = 0.05
        else: pickupdelay = PICKUPDELAY[self.targetspicked]

        # Medium reward: within 20 px of target
        if cd <= 20:
            reward += (0.5) / (pickupdelay)
            self.TargetTicks += DT

        # Large reward: within 10 px of target (on target)
        if cd <= 10:
            reward += (1) / (pickupdelay)

        # Target collected — relocate and increase difficulty
        if self.TargetTicks >= pickupdelay:
            if not self.interactable: self.PickTarget()
            self.TargetTicks = 0
            self.targetspicked += 1

        tilt = abs(anglediff(self.drone.Shaft.get_rot(), 0))

        # Terminate if drone flies out of bounds
        if not self.InBounds(self.drone.MainBody.x(), self.drone.MainBody.y()) : finished = True

        # Truncate episode after 30 seconds (during training only)
        truncated = False
        if self.timepasted > 30 and not self.interactable and self.cantimeout: truncated = True

        # Penalty for excessive tilt (beyond ±60°)
        if tilt > math.pi/3: reward -= 0.8

        # Small survival reward for each step
        reward += 0.1

        if (self.render_mode == "human"): self.render()

        return state, reward, finished, truncated, {}



    def render(self, render_mode="None"):
        """Draw the current frame: target, drone, and thrust particle effects."""
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
        """Randomly place a new target, ensuring it's at least 100 px from the drone."""
        targetX = random.randint(50, WIDTH-50)
        targetY = random.randint(50, HEIGHT-50)
        for i in range(20):
            if distance((targetX, targetY), self.drone.MainBody.get_pos()) > 100: break
            targetX = random.randint(50, WIDTH-50)
            targetY = random.randint(50, HEIGHT-50)

        self.Target = (targetX, targetY)


    def reset(self, seed=None, options=None):
        """Reset the environment: rebuild the drone and pick a new target."""
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
        """Shut down the pygame window."""
        pygame.quit()


# ── Physics Part (rigid body wrapper) ─────────────────────────────────────────

class Part():
    """
    A wrapper around a pymunk Body + Shape, with helpers for joining parts together.
    Used to build the drone's main body, shaft, and motor components.
    """
    MAXJOINTFORCE = math.inf

    def ShapeKeyToShape(key, body, args):
        """Factory: create a pymunk shape from a string key ('circle' or 'rect')."""
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
        """Rigidly attach another part (pin + rotation lock, like a screw)."""
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
        """Attach another part with a pivot joint (allows rotation)."""
        joint1 = pymunk.PivotJoint(self.body, otherpart.body, anchorself, anchorother)

        joint1.max_force = Part.MAXJOINTFORCE
        joint1.collide_bodies = False
        joint1.error_bias = pow(1 - 0.9, 60)
        self.joints.append(joint1)

    def AttachMotorControl(self, otherpart):
        """Attach a simple motor constraint between this part and another."""
        motor = pymunk.SimpleMotor(self.body, otherpart.body, 0)
        motor.error_bias = pow(1 - 0.9, 60)
        motor.max_force = Part.MAXJOINTFORCE
        motor.collide_bodies = False
        motor.rate = 0
        self.joints.append(motor)
        return motor

    def init(self, space):
        """Add this part's body, shape, and joints to the physics space."""
        space.add(self.body, self.shape)
        for j in self.joints: space.add(j)

    def remove(self, space):
        """Remove this part's joints, shape, and body from the physics space."""
        for j in self.joints: space.remove(j)
        space.remove(self.shape, self.body)

    # ── Convenience accessors ─────────────────────────────────────────────

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
        """Apply a force at this part's center of mass (world coordinates)."""
        self.body.apply_force_at_world_point(force, self.get_pos())


# ── Thrust Particle Effect ────────────────────────────────────────────────────

class DroneFire():
    """A single particle emitted from a propeller for visual thrust effects."""

    def __init__(self, x, y, dir, vel, r):
        self.ir = r          # Initial radius (used for color fade calculation)
        self.r = r           # Current radius
        self.x = x
        self.y = y
        self.vx = math.cos(dir) * vel
        self.vy = math.sin(dir) * vel

    def draw(self, screen):
        """Draw the particle and advance its position; shrinks each frame."""
        if (self.r<=0): return
        # Fade from yellow → red → black as the particle shrinks
        color = blendcolor(blendcolor(YELLOW, RED, math.sqrt(1-(self.r/self.ir))), BLACK, (1-(self.r/self.ir))**2)
        pygame.draw.circle(screen, color, (self.x, self.y), self.r)
        self.x += self.vx
        self.y += self.vy
        self.r -= 0.1

    def is_dead(self):
        return (self.r<=1)


# ── Drone (assembled from Parts) ─────────────────────────────────────────────

class Drone():
    """
    The 2D drone, assembled from four Part objects:
      - MainBody: central circle (pivot point)
      - Shaft: horizontal bar connecting the motors
      - LeftMotor / RightMotor: tiltable propeller mounts on each end of the shaft

    The agent controls motor thrust power and tilt angle; the Drone class
    handles force application, motor interpolation, and particle effects.
    """
    MAXTHRUST = 800        # Maximum thrust force per motor (in pymunk units)
    TURNPOWER = 0.4        # Motor tilt interpolation speed (0–1, higher = faster response)

    def __init__(self, space, x, y):
        self.space = space

        # Central body (circle, r=12)
        self.MainBody = Part("circle", [12], x, y, 0.1, BLUE)
        # Horizontal shaft (60×10 rectangle)
        self.Shaft = Part("rect", [[60, 10]], x, y, 0.1, RED)

        # Left and right motors (10×18 rectangles) mounted at ends of shaft
        self.LeftMotor = Part("rect", [[10, 18]], x-30, y, 0.1, BLUE)
        self.RightMotor = Part("rect", [[10, 18]], x+30, y, 0.1, BLUE)

        # Connect parts: body↔shaft (pivot), shaft↔motors (pivot)
        self.MainBody.PinPart(self.Shaft, (0, 0), (0, 0))
        self.Shaft.PinPart(self.LeftMotor, (-30, 0), (0, 0))
        self.Shaft.PinPart(self.RightMotor, (30, 0), (0, 0))

        # Current motor tilt angles (interpolated toward targets each step)
        self.LeftMotorRot = 0
        self.RightMotorRot = 0

        # Register all parts in the physics space
        self.Shaft.init(self.space)
        self.MainBody.init(self.space)
        self.LeftMotor.init(self.space)
        self.RightMotor.init(self.space)

        # Target motor tilt angles (set by the agent each step)
        self.TargetLeftMotorRot = 0
        self.TargetRightMotorRot = 0

        # Visual thrust particles
        self.Particles = []

    def drawEffects(self, screen):
        """Draw and update all thrust particles, removing dead ones."""
        for i, par in enumerate(self.Particles):
            if par.is_dead(): self.Particles.pop(i)
            par.draw(screen)

    def step(self):
        """Interpolate motor angles toward their targets and apply tilt limits."""
        self.RightMotorRot += (self.TargetRightMotorRot - self.RightMotorRot) * Drone.TURNPOWER
        self.LeftMotorRot += (self.TargetLeftMotorRot - self.LeftMotorRot) * Drone.TURNPOWER
        # Clamp tilt to ±60°
        self.RightMotorRot = max(-math.pi/3, min(math.pi/3, self.RightMotorRot))
        self.LeftMotorRot = max(-math.pi/3, min(math.pi/3, self.LeftMotorRot))
        # Set motor angles relative to the shaft's current rotation
        self.LeftMotor.body.angle = self.LeftMotorRot + self.Shaft.get_rot()
        self.RightMotor.body.angle = self.RightMotorRot + self.Shaft.get_rot()


    def ThrustLeft(self, power, par=True):
        """Apply thrust to the left motor and spawn particle effects."""
        power = max(0, min(1, power))

        # Thrust direction is perpendicular to motor orientation (pointing "up" from motor)
        ThrustDir = self.LeftMotor.get_rot() - math.pi/2
        fx = Drone.MAXTHRUST * power * math.cos(ThrustDir)
        fy = Drone.MAXTHRUST * power * math.sin(ThrustDir)

        self.LeftMotor.add_force((fx, fy))
        if not par: return
        # Spawn fire particles in the opposite direction of thrust
        for i in range(math.ceil(10*power)):
            angleoffset = ((random.random()*2)-1) * math.pi/12 * power + math.pi
            self.Particles.append(DroneFire(self.LeftMotor.get_pos()[0], self.LeftMotor.get_pos()[1], ThrustDir + angleoffset, power*12+random.randint(1, 3), power*3+random.random()*2))


    def ThrustRight(self, power, par=True):
        """Apply thrust to the right motor and spawn particle effects."""
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
        """Set the target tilt for the left motor. dir ∈ [-1, 1] → angle ∈ [-60°, 60°]."""
        self.TargetLeftMotorRot = dir * math.pi/3

    def SetTargetRightMotor(self, dir):
        """Set the target tilt for the right motor. dir ∈ [-1, 1] → angle ∈ [-60°, 60°]."""
        self.TargetRightMotorRot = dir * math.pi/3

    def remove(self):
        """Remove all drone parts from the physics space."""
        self.MainBody.remove(self.space)
        self.Shaft.remove(self.space)
        self.LeftMotor.remove(self.space)
        self.RightMotor.remove(self.space)
