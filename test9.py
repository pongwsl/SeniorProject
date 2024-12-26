# test10.py
# created by pongwsl on Dec 26, 2024
# last updated Dec 26, 2024
# Controlling UR3 arm in CoppeliaSim using official Remote API

import sim  # Ensure sim.py is in the same directory or in PYTHONPATH
import sys
import time
import math

def connect_to_coppeliasim():
    sim.simxFinish(-1)  # Close any existing connections
    client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if client_id == -1:
        print("Failed to connect to CoppeliaSim")
        sys.exit()
    else:
        print(f"Connected to CoppeliaSim with client ID: {client_id}")
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
        # Alternatively, use sim.simx_opmode_blocking for synchronous calls

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

    print(f"Moving UR3 to target positions: {target_positions}")
    set_joint_positions(client_id, joint_handles, target_positions)

    # Allow some time for the movement to complete
    time.sleep(5)

    disconnect(client_id)
    print("Disconnected from CoppeliaSim")

if __name__ == "__main__":
    main()