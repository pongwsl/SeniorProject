# test11.py
# created by pongwsl on Jan 14, 2024
# latest edited on Jan 14, 2024
#
# This script demonstrates how to retrieve (x, y, z) values from hand control
# (via getPosition.py) and then move a sphere in CoppeliaSim in real-time.
#
# Requirements:
#  - CoppeliaSim with the "ZMQ remote API" add-on running
#  - A sphere in the scene named "Sphere" (or rename as needed)
#  - getPosition.py in "tools/" folder

import time
import sys
import threading
from typing import Tuple

# Import the Remote API Client
# (pip install coppeliasim-zmqremoteapi-client if needed)
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Import the position generator from your local tools folder
# Adjust the import path if needed
from tools.getPosition import getPosition


def main():
    print("Program started")

    # Connect to CoppeliaSim using the ZeroMQ Remote API
    client = RemoteAPIClient()
    sim = client.require('sim')

    # (Optional) If you want to load a specific scene instead of an already-running instance:
    sim.loadScene(sim.getStringParam(sim.stringparam_scenedefaultdir) + '/sphere.ttt')

    # Start the simulation (if not already started)
    sim.startSimulation()

    # Retrieve a handle to your sphere in CoppeliaSim
    # Make sure the sphere object in CoppeliaSim is named "/Sphere"
    try:
        sphereHandle = sim.getObject('/Sphere')
    except Exception as e:
        print("Could not get handle to /Sphere. Please check the object name in CoppeliaSim.")
        sim.stopSimulation()
        sys.exit(1)

    # Initialize the generator that yields (x, y, z) from hand movements
    positionGen = getPosition()

    print("Moving sphere in real-time based on hand movements...")

    try:
        while True:
            # Get next (x, y, z) from the generator
            x, y, z = next(positionGen)

            # Update the sphere's position in CoppeliaSim
            # -1 indicates absolute/world frame reference
            sim.setObjectPosition(sphereHandle, -1, [x, y, z])

            # A small sleep to avoid flooding CoppeliaSim with updates
            time.sleep(0.05)

    except StopIteration:
        # If the generator is exhausted (not typically expected in a continuous scenario)
        print("Position generator has finished producing values.")
    except KeyboardInterrupt:
        print("Interrupted by user.")

    # Stop simulation and wrap up
    sim.stopSimulation()
    print("Program ended")


if __name__ == "__main__":
    main()