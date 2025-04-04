# ur3Move.py
# created by pongwsl on mar 7, 2025
# to use postions from .txt file to move UR3 robot in CoppeliaSim
# based on p'nine's work. thanks to P'Nine Ninth2234 >> see his work in GitHub.

import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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

latencies = []

while True:
    try:
        start_time = time.time()
        pos = next(pos_gen)
    except StopIteration:
        print("Hand control exited. Exiting simulation.")
        break

    # Combine the position with the quaternion
    pose_to_move = list(pos) + quart

    # Command the UR3 robot to move to the new pose
    ur3.move_pose_nonblocking(pose_to_move)
    print("Moving to pose:", pose_to_move)

    latency = (time.time() - start_time) * 1000  # Convert to milliseconds
    latencies.append(round(latency))

    # Small delay to avoid overloading the simulation
    time.sleep(0.05)

sim.stopSimulation()

total_frames = len(latencies)
avg_latency = np.mean(latencies)
min_latency = np.min(latencies)
max_latency = np.max(latencies)
range_latency = max_latency - min_latency
median_latency = np.median(latencies)
mode_latency = stats.mode(latencies, keepdims=True)[0][0]
std_dev_latency = np.std(latencies)

print(f"Total Frames Processed: {total_frames}")
print(f"Average Latency: {avg_latency:.2f} ms")
print(f"Min Latency: {min_latency:.2f} ms")
print(f"Max Latency: {max_latency:.2f} ms")
print(f"Latency Range: {range_latency:.2f} ms")
print(f"Median Latency: {median_latency:.2f} ms")
print(f"Mode Latency: {mode_latency:.2f} ms")
print(f"Standard Deviation: {std_dev_latency:.2f} ms")

# Define bin width (adjust as needed)
bin_width = 10
latency_bins = np.arange(int(min(latencies)), int(max(latencies)) + bin_width, bin_width)
latency_counts, _ = np.histogram(latencies, bins=latency_bins)

plt.figure(figsize=(10, 5))
plt.bar(latency_bins[:-1], latency_counts, width=bin_width, align='edge', color='blue', edgecolor='black')
plt.xlabel('Latency (ms)')
plt.ylabel('Number of Frames')
plt.title('getPostion.py Latency Distribution')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
