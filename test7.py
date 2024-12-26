# test7.py
# created by pongwsl on dec 25, 2024
# last updated dec 26, 2024
# trying to connect python with coppeliasim to control UR3 arm

# from tools.sim import sim
# import sim
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import sys
import time
import numpy as np

client = RemoteAPIClient()
sim = client.require('sim')

def connect_to_coppeliasim():
    sim.simxFinish(-1)  # Close any existing connections
    client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if client_id == -1:
        print("Failed to connect to CoppeliaSim")
        sys.exit()
    else:
        print("Connected to CoppeliaSim with client ID:", client_id)
    return client_id

def get_joint_handles(client_id, joint_names):
    joint_handles = []
    for name in joint_names:
        error_code, handle = sim.simxGetObjectHandle(client_id, name, sim.simx_opmode_blocking)
        if error_code != sim.simx_return_ok:
            print(f"Failed to get handle for joint: {name}")
            sys.exit()
        joint_handles.append(handle)
    return joint_handles

def set_joint_positions(client_id, joint_handles, positions):
    for handle, pos in zip(joint_handles, positions):
        sim.simxSetJointTargetPosition(client_id, handle, pos, sim.simx_opmode_oneshot)

def disconnect(client_id):
    sim.simxFinish(client_id)


def main():
    # Define joint names as per your CoppeliaSim scene
    joint_names = ['UR3_joint1', 'UR3_joint2', 'UR3_joint3',
                  'UR3_joint4', 'UR3_joint5', 'UR3_joint6']

    client_id = connect_to_coppeliasim()
    joint_handles = get_joint_handles(client_id, joint_names)

    # Define target joint positions in radians
    target_positions = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]

    print("Moving UR3 to target positions:", target_positions)
    set_joint_positions(client_id, joint_handles, target_positions)

    # Allow some time for the movement to complete
    time.sleep(5)

    disconnect(client_id)
    print("Disconnected from CoppeliaSim")

if __name__ == "__main__":
    main()