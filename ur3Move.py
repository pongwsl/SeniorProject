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
  File → Open scene → coppeliasim/ur3withSpray.ttt
Press ENTER to continue...
""")

client = RemoteAPIClient()
sim = client.getObject('sim')

ur3 = UR3(sim, "/UR3")
ur3.reset_target()
sim.startSimulation()

# --- Spray control signals ---
spray_signal = 'sprayOn'
def set_spray(on: bool):
    """
    Turn the spray on or off by signaling CoppeliaSim.
    """
    # 1 for on, 0 for off
    sim.setIntegerSignal(spray_signal, 1 if on else 0)

import math

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) in radians to a quaternion.
    Returns:
        [x, y, z, w]
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [x, y, z, w]

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

    # Unpack hand position and orientation: x, y, z, roll, pitch, yaw
    x, y, z, roll, pitch, yaw = pos

    # Convert Euler angles from degrees to radians (if getPosition returns degrees)
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)

    # Convert Euler angles to quaternion
    quat = euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)

    # Build the pose to move: position and orientation from the hand
    pose_to_move = [x, y, z] + quat

    # Command the UR3 robot to move to the new pose
    ur3.move_pose_nonblocking(pose_to_move)
    print("Moving to pose:", pose_to_move)
    # Example: spray when close to target height
    if z < 0.2:
        set_spray(True)
    else:
        set_spray(False)

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
