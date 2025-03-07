# plotPosition.py
# by pongwsl on feb 28, 2025
# use position data .txt to plot a matlab graph

#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast  # for safely evaluating tuple strings

def load_positions(file_path):
    """
    Reads the file and parses each line as a (x, y, z) tuple.
    """
    positions = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pos = ast.literal_eval(line)  # Convert string to tuple
                    if isinstance(pos, tuple) and len(pos) == 3:
                        positions.append(pos)
                    else:
                        print(f"Invalid format in line: {line}")
                except Exception as e:
                    print(f"Error parsing line '{line}': {e}")
    return positions

def main():
    # Use tkinter to pop up a file selection dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select Position File",
        filetypes=[("Text Files", "*.txt")]
    )
    
    if not file_path:
        print("No file selected. Exiting...")
        return
    
    # Load positions from the selected file
    positions = load_positions(file_path)
    if not positions:
        print("No valid positions found in the file.")
        return

    # Separate the positions into x, y, z coordinates
    xs = [pos[0] for pos in positions]
    ys = [pos[1] for pos in positions]
    zs = [pos[2] for pos in positions]

    # Set up the Matplotlib figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Trajectory from Position File")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")

    # Plot the trajectory with a solid line and markers for each position
    ax.plot(xs, ys, zs, linestyle='-', color='blue', marker='o', markersize=4, label="Trajectory")
    ax.legend()

    # Optionally adjust plot limits dynamically with a small buffer
    buffer = 0.5
    ax.set_xlim(min(xs)-buffer, max(xs)+buffer)
    ax.set_ylim(min(ys)-buffer, max(ys)+buffer)
    ax.set_zlim(min(zs)-buffer, max(zs)+buffer)

    plt.show()

if __name__ == "__main__":
    main()