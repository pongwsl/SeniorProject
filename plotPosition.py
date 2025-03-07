# plotPosition.py
# by pongwsl on feb 28, 2025
# edited on mar 7, 2025
# use position data .txt to plot a matlab graph

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Button

def read_positions(file_path):
    """
    Reads positions from a text file line by line.
    Each line contains a tuple of (x, y, z) coordinates.

    Args:
        file_path (str): Path to the positions.txt file.

    Returns:
        list of tuples: List of (x, y, z) positions.
    """
    positions = []
    with open(file_path, 'r') as file:
        for line in file:
            positions.append(eval(line.strip()))  # Convert string tuple to actual tuple
    return positions

def main():
    """
    Main function to read positions from a file and visualize them in real-time using Matplotlib.
    """
    file_path = "positions_20250228_183419.txt"  # Update with actual file name
    positions = read_positions(file_path)

    # Set up the Matplotlib figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Object Position Tracking')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    # Initialize a point in the plot to represent the object
    point, = ax.plot([0], [0], [0], marker='o', markersize=10, color='red')

    # Store the trajectory
    trajectory_x, trajectory_y, trajectory_z = [], [], []
    trajLine, = ax.plot([], [], [], linestyle='-', color='blue')

    def update_plot(frame):
        """
        Update function for Matplotlib animation.
        """
        if frame >= len(positions):
            ani.event_source.stop()
            return point, trajLine

        x, y, z = positions[frame]
        
        # Update the point's position
        point.set_data([x], [y])
        point.set_3d_properties([z])

        # Update the trajectory
        trajectory_x.append(x)
        trajectory_y.append(y)
        trajectory_z.append(z)
        trajLine.set_data(trajectory_x, trajectory_y)
        trajLine.set_3d_properties(trajectory_z)

        return point, trajLine

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update_plot, frames=len(positions), interval=50, blit=False
    )

    # Add a "Clear" button to reset the trajectory
    ax_clear = plt.axes([0.8, 0.05, 0.1, 0.075])
    clear_button = Button(ax_clear, 'Clear')

    def clear_graph(event):
        """
        Callback function for the Clear button.
        """
        trajectory_x.clear()
        trajectory_y.clear()
        trajectory_z.clear()
        trajLine.set_data([], [])
        trajLine.set_3d_properties([])
        plt.draw()

    clear_button.on_clicked(clear_graph)

    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()
