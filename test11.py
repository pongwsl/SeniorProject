# test11.py
# created by pongwsl on Jan 14, 2025
# This script creates a sphere in CoppeliaSim and moves it based on hand movements using getPosition()

import threading
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Import the getPosition generator from the tools module
from tools.getPosition import getPosition

def moveSphere():
    client = RemoteAPIClient()
    sim = client.require('sim')
    sim.setStepping(True)
    
    # Create a sphere shape in the simulation
    shapeType = 0                # 0 indicates a sphere shape
    options = 16                 # default options
    sizes = [0.05, 0.05, 0.05]   # radii of the sphere along x, y, z
    mass = 0.01
    sphereHandle = sim.createPureShape(shapeType, options, sizes, mass)
    
    # Initialize the position generator
    positionGen = getPosition()
    
    try:
        while True:
            # Get the next position from the generator
            pos = next(positionGen)
            # Update the sphere's position relative to the world frame (-1)
            sim.setObjectPosition(sphereHandle, -1, pos)
            # Wait a short while before the next update
            time.sleep(0.05)
    except StopIteration:
        # In case the generator finishes (unlikely with continuous hand control), exit loop
        pass
    finally:
        sim.setStepping(False)

print('Program started')

# Initialize client and simulation
client = RemoteAPIClient()
sim = client.require('sim')

# Load a scene; adjust the path to your scene file if necessary
scenePath = sim.getStringParam(sim.stringparam_scenedefaultdir) + '/messaging/movementViaRemoteApi.ttt'
sim.loadScene(scenePath)

# Start the sphere movement in a separate thread
sphereThread = threading.Thread(target=moveSphere)
sphereThread.start()

# Start the simulation
sim.startSimulation()

# Wait for the sphere thread to complete (it runs indefinitely until interrupted)
sphereThread.join()

# Stop the simulation when the thread ends or on interrupt
sim.stopSimulation()

print('Program ended')