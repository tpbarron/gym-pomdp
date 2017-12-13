"""
Adapted from: https://raw.githubusercontent.com/bulletphysics/bullet3/master/examples/pybullet/gym/pybullet_envs/bullet/racecarGymEnv.py
"""

import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import math
import time
import random

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

import pybullet
import pybullet_data
import pybullet_envs.bullet.bullet_client as bullet_client
import pybullet_envs.bullet.racecar as racecar

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class TMazeRacecarGymEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=50,
                 isEnableSelfCollision=True,
                 isDiscrete=False,
                 renders=False,
                 length=1):
        self._timeStep = 0.01
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._ballUniqueId = -1
        self._envStepCounter = 0
        self._renders = renders
        self._isDiscrete = isDiscrete
        if self._renders:
            self._p = bullet_client.BulletClient(
            connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self._seed()
        #self.reset()
        observationDim = 2
        observation_high = np.ones(observationDim) * 1000 #np.inf

        if (isDiscrete):
            self.action_space = spaces.Discrete(9)
        else:
            action_dim = 2
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)

        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None
        self.length = length
        self.switch = None

    def _build_tmaze(self):
        self._p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), "assets/")) #used by loadURDF
        self.cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[-1, 0, 0.5])

        for i in range(-1, self.length):
            self.cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[i, 1, 0.5])
            self.cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[i, -1, 0.5])

        self.cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[self.length, -2, 0.5])
        self.cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[self.length, 2, 0.5])

        self.cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[self.length-1, -2, 0.5])
        self.cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[self.length-1, 2, 0.5])

        for i in range(5):
            self.cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[self.length+1, i-2, 0.5])

    def _reset(self):
        self._p.resetSimulation()
        #p.setPhysicsEngineParameter(numSolverIterations=300)
        self._p.setTimeStep(self._timeStep)
        #self._p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"))
        stadiumobjects = self._p.loadSDF(os.path.join(self._urdfRoot,"stadium.sdf"))
        #move the stadium objects slightly above 0
        for i in stadiumobjects:
            pos,orn = self._p.getBasePositionAndOrientation(i)
            newpos = [pos[0],pos[1],pos[2]-0.1]
            self._p.resetBasePositionAndOrientation(i,newpos,orn)

        # TODO: add walls to Tmaze
        self._build_tmaze()

        self._p.setGravity(0,0,-10)
        self._racecar = racecar.Racecar(self._p,urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        for i in range(100):
            self._p.stepSimulation()

        # self.switch = 1 if np.random.random() < 0.5 else -1

        # temporariliy just drive foward
        self.goal = np.array([self.length, 0]) #self.switch])
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def __del__(self):
        self._p = 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        self._observation = [] #self._racecar.getObservation()
        carpos,carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        # ballpos,ballorn = self._p.getBasePositionAndOrientation(self._ballUniqueId)
        # invCarPos,invCarOrn = self._p.invertTransform(carpos,carorn)
        # ballPosInCar,ballOrnInCar = self._p.multiplyTransforms(invCarPos,invCarOrn,ballpos,ballorn)

        self._observation.extend([carpos[0]/(self.length+1), carpos[1]])
        # self._observation.extend([ballPosInCar[0],ballPosInCar[1]])
        return self._observation

    def _step(self, action):
        if (self._renders):
            basePos,orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
            #self._p.resetDebugVisualizerCamera(1, 30, -40, basePos)

        if (self._isDiscrete):
	        fwd = [-1,-1,-1,0,0,0,1,1,1]
	        steerings = [-0.6,0,0.6,-0.6,0,0.6,-0.6,0,0.6]
	        forward = fwd[action]
	        steer = steerings[action]
	        realaction = [forward,steer]
        else:
            realaction = action

        self._racecar.applyAction(realaction)
        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            self._observation = self.getExtendedObservation()

            if self._termination():
                break
        self._envStepCounter += 1
        reward = self._reward()
        done = self._termination()

        return np.array(self._observation), reward, done, {}

    def _render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos,orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        return False

    def _reward(self):
        """
        negative dist to goal
        """
        carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        carxy = np.array(carpos[0:2])
        dist = np.linalg.norm(carxy - self.goal)
        reward = -dist
        return reward


from itertools import count
if __name__ == '__main__':
    env = TMazeRacecarGymEnv(isDiscrete=True, renders=True, length=1)

    while True:
        env.reset()
        env.render()
        done = False

        for i in range(2):
            env.render()
            obs, rew, done, info = env.step(7) #env.action_space.sample())
            import time
            time.sleep(0.1)

        for i in count(1):
        # while not done:
            env.render()
            obs, rew, done, info = env.step(6) #env.action_space.sample())
            import time
            time.sleep(0.1)

            input("")
