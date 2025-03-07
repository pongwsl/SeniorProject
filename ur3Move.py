# ur3Move.py
# created by pongwsl on mar 7, 2025
# to use postions from .txt file to move UR3 robot in CoppeliaSim
# based on p'nine's work. thanks to P'Nine Ninth2234 >> see his work in GitHub.

import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from tools.ur3 import UR3
from tools.getPosition import getPosition

input("""
Before starting, open CoppeliaSim:
  File → Open scene → coppeliasim/ur3.ttt
Press ENTER to continue...
""")

client = RemoteAPIClient()
sim = client.getObject('sim')

ur3 = UR3(sim, "/UR3")
ur3.reset_target()
sim.startSimulation()

# Define the quaternion (orientation) values
quart = [0.5, 0.5, 0.5, 0.5]

# Get the position generator (handControl() internally displays the annotated frame)
pos_gen = getPosition()

while True:
    try:
        pos = next(pos_gen)
    except StopIteration:
        print("Hand control exited. Exiting simulation.")
        break

    # Combine the position with the quaternion
    pose_to_move = list(pos) + quart
    

    # Command the UR3 robot to move to the new pose
    ur3.move_pose(pose_to_move)
    print("Moving to pose:", pose_to_move)

    # Small delay to avoid overloading the simulation
    time.sleep(0.05)

sim.stopSimulation()