# test14.py
# created Feb 19, 2025
# edited Feb 19, 2025

#!/usr/bin/env python3
"""
This script connects to CoppeliaSim via the ZeroMQ Remote API and gradually moves the
UR3 robot arm's end-effector to a specified position (x, y, z) with a desired orientation
(alpha, beta, gamma). The UR3's base is fixed at (0, 0, 0), and a dummy target (named "/UR3")
is used by the scene's inverse kinematics to drive the end-effector. The scene file used is "UR3.ttt".

Usage:
    Run the script and enter coordinates when prompted, e.g.,
        Enter target position and orientation as x y z alpha beta gamma (or 'q' to quit): 0.5 0.0 0.2 0 0 1.57
"""

import time
import sys
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def lerp(a, b, t):
    """Linearly interpolate between values a and b using factor t (0 <= t <= 1)."""
    return a + (b - a) * t

def move_target_slowly(sim, target_handle, new_pos, new_ori, steps=50, step_duration=0.1):
    """
    Gradually moves the dummy target from its current position and orientation to new_pos and new_ori.

    Parameters:
      sim           : The CoppeliaSim remote API object.
      target_handle : Handle of the dummy target.
      new_pos       : List of three floats [x, y, z] indicating the new target position.
      new_ori       : List of three floats [alpha, beta, gamma] for the new orientation in Euler angles.
      steps         : Number of interpolation steps (default 50).
      step_duration : Time (in seconds) between steps (default 0.1).
    """
    # Get current position and orientation in the world frame (-1)
    current_pos = sim.getObjectPosition(target_handle, -1)
    current_ori = sim.getObjectOrientation(target_handle, -1)
    
    for i in range(steps + 1):
        t = i / steps
        interp_pos = [lerp(current_pos[j], new_pos[j], t) for j in range(3)]
        interp_ori = [lerp(current_ori[j], new_ori[j], t) for j in range(3)]
        sim.setObjectPosition(target_handle, -1, interp_pos)
        sim.setObjectOrientation(target_handle, -1, interp_ori)
        time.sleep(step_duration)

def main():
    print("Connecting to CoppeliaSim...")
    client = RemoteAPIClient()  # Create the remote API client
    sim = client.require('sim')

    # Stop any running simulation and wait a moment
    try:
        sim.stopSimulation()
    except Exception:
        pass
    time.sleep(0.5)

    # Load the scene file "UR3.ttt"
    scene_path = sim.getStringParam(sim.stringparam_scenedefaultdir) + '/UR3.ttt'
    sim.loadScene(scene_path)

    # Start the simulation and allow some time for initialization
    sim.startSimulation()
    time.sleep(0.5)

    # Retrieve the dummy target handle.
    # Ensure your scene "UR3.ttt" includes a dummy or target object named "/UR3" for the IK target.
    try:
        target_handle = sim.getObject('/UR3')
    except Exception as e:
        print("Error: Could not get target handle. Please ensure an object named '/UR3' exists in the scene.")
        sim.stopSimulation()
        sys.exit(1)

    print("UR3 target found. Enter target position and orientation for the end-effector.")
    print("The UR3's base remains fixed at (0, 0, 0).")

    try:
        while True:
            user_input = input("Enter target position and orientation as x y z alpha beta gamma (or 'q' to quit): ").strip()
            if user_input.lower() == 'q':
                break

            parts = user_input.split()
            if len(parts) != 6:
                print("Please enter exactly six numerical values separated by spaces.")
                continue

            try:
                x, y, z, alpha, beta, gamma = map(float, parts)
            except ValueError:
                print("Invalid input. Please ensure you enter numeric values.")
                continue

            new_position = [x, y, z]
            new_orientation = [alpha, beta, gamma]
            # Gradually move the dummy target to the new position and orientation.
            move_target_slowly(sim, target_handle, new_position, new_orientation, steps=50, step_duration=0.1)
            print(f"End-effector target moved slowly to position x={x}, y={y}, z={z} with orientation alpha={alpha}, beta={beta}, gamma={gamma}")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received; exiting.")

    # Stop the simulation before quitting
    sim.stopSimulation()
    print("Simulation stopped. Program ended.")

if __name__ == '__main__':
    main()