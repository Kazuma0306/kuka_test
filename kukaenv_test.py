import os
if not os.path.exists('bullet3'):
  os.system('git clone https://github.com/bulletphysics/bullet3.git')


import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from base64 import b64encode
from IPython.display import HTML
import math
import numpy as np
import imageio

imageio.plugins.ffmpeg.download()

physicsClient = p.connect(p.GUI)  # ローカルで実行するときは、p.GUI を指定してください


# 床を出現させます
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
timestep = 1. / 240.
p.setTimeStep(timestep)
planeId = p.loadURDF("plane.urdf")

# kukaを読み込みます
kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])

# kukaのEnd EffectorのIndexを取得
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
if (numJoints != 7):
    exit()

# kukaのパラメータに関する設定
#lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
#upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
#joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
#restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
#joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# 状態をリセット
for j_idx in range(numJoints):
    p.resetJointState(kukaId, j_idx, rp[j_idx])

frames = []
t = 0.

for i in range(2000):
    t = t + 0.01
    p.stepSimulation()

    # End Effectorの位置 x-y平面で(-0.6, 0)を中心に半径0.2の円を描くように動く
    pos = [ -0.6 + 0.2 * math.cos(t), 0.2 * math.sin(t), 0.4]

    # End Effectorの方向
    # 今回は固定
    orn = p.getQuaternionFromEuler([0, -math.pi, 0])
    # IKの計算
    jointPoses = p.calculateInverseKinematics(kukaId,
                                              kukaEndEffectorIndex,
                                              pos,
                                              orn,
                                              lowerLimits=ll,
                                              upperLimits=ul,
                                              jointRanges=jr,
                                              restPoses=rp)
    # IKの計算結果通りに各関節を動かす
    for j_idx in range(numJoints):
        p.setJointMotorControl2(bodyIndex=kukaId,
                            jointIndex=j_idx,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=jointPoses[j_idx],
                            targetVelocity=0,
                            force=500,
                            positionGain=0.03,
                            velocityGain=1)

    ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
    if i % 8 == 0:
      width, height, rgbImg, depthImg, segImg = p.getCameraImage(360,240)
      frames.append(rgbImg)
