# ur3Move.py
# created by pongwsl on mar 7, 2025
# to use postions from .txt file to move UR3 robot in CoppeliaSim
# based on p'nine's work. thanks to P'Nine Ninth2234 >> see his work in GitHub.

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from tools.ur3 import UR3

input("""
Before starting, open CoppeliaSim:
  File → Open scene → coppeliasim/ur3.ttt
Press ENTER to continue...
""")

client = RemoteAPIClient()
sim = client.getObject('sim')

ur3 = UR3(sim,"/UR3")

ur3.reset_target()
sim.startSimulation()

# -----move-----
quart = [0.5,0.5,0.5,0.5]
path_pos = [[-0.4,-0.1,0.5],
            [-0.4,0.1,0.5],
            [-0.4,0.1,0.3],
            [-0.4,-0.1,0.3]]

for _ in range(3):
    for pos in path_pos:
        pose_to_move = pos+quart
        ur3.move_pose(pose_to_move)
        print(pose_to_move)

# --------------

sim.stopSimulation()
