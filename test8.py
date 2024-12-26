# test8.py
# created by pongwsl on dec 26, 2024
# last updated dec 26, 2024
# connect python with coppeliasim
# THIS IS THE ONLY ONE THAT WORK ON dec 26.

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

sim.setStepping(True)

sim.startSimulation()
while (t := sim.getSimulationTime()) < 3:
    print(f'Simulation time: {t:.2f} [s]')
    sim.step()
sim.stopSimulation()