# playground.py
# created by pongwsl on dec 20, 2024
# This is only a playground, not save something to be permanent here.

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

sim.setStepping(True)

sim.startSimulation()
while (t := sim.getSimulationTime()) < 3:
    print(f'Simulation time: {t:.2f} [s]')
    sim.step()
sim.stopSimulation()