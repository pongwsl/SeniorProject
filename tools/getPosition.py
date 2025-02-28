# tools/getPosition.py
# created by pongwsl on Dec 27, 2024
# latest edited on Feb 28, 2024
# Controls the position of an object based on hand movements using MediaPipe
# Displays the object's position in real-time using Matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Button
import time
from typing import Generator, Tuple
# from tools.handControl import handControl

if __name__ == "__main__" and __package__ is None:
    import sys
    import os
    # Add the parent directory to sys.path so that 'tools' is recognized.
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "tools"
from .handControl import handControl

def getPosition() -> Generator[Tuple[float, float, float], None, None]:
    """
    Controls the position of an object based on hand movement deltas.

    Initializes the object's position at (0, 0, 0) and updates it based on the deltas
    (dx, dy, dz) received from the handControl generator.

    Yields:
        Tuple[float, float, float]: The updated (x, y, z) position of the object.
    """
    # Initialize the starting position
    x, y, z = 0.0, 0.0, 0.0

    # Initialize the handControl generator
    controlGen = handControl()

    for dx, dy, dz in controlGen:
        # Update the position based on the deltas
        x += dx
        y += dy
        z += dz

        yield (x, y, z)

def main():
    """
    Main function for debugging getPosition().
    Uses Matplotlib to display the current position of the object in real-time.
    """
    # Initialize the position generator
    positionGen = getPosition()

    # Set up the Matplotlib figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Object Position Tracking')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    # Set the initial limits of the plot
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Initialize a point in the plot to represent the object
    point, = ax.plot([0], [0], [0], marker='o', markersize=10, color='red')

    # Store the trajectory
    trajectory_x, trajectory_y, trajectory_z = [], [], []
    # Use a solid line for the trajectory
    trajLine, = ax.plot([], [], [], linestyle='-', color='blue')

    def update_plot(frame):
        """
        Update function for Matplotlib animation.

        Args:
            frame: Frame number (unused).
        """
        try:
            x, y, z = next(positionGen)
        except StopIteration:
            # If the generator is exhausted, stop the animation
            ani.event_source.stop()
            return point, trajLine

        # Update the point's position
        point.set_data([x], [y])
        point.set_3d_properties([z])

        # Update the trajectory
        trajectory_x.append(x)
        trajectory_y.append(y)
        trajectory_z.append(z)
        trajLine.set_data(trajectory_x, trajectory_y)
        trajLine.set_3d_properties(trajectory_z)

        # Optionally, adjust the plot limits dynamically
        buffer = 0.5
        ax.set_xlim(min(trajectory_x) - buffer, max(trajectory_x) + buffer)
        ax.set_ylim(min(trajectory_y) - buffer, max(trajectory_y) + buffer)
        ax.set_zlim(min(trajectory_z) - buffer, max(trajectory_z) + buffer)

        return point, trajLine

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update_plot, frames=None, interval=50, blit=False
    )

    # Add a "Clear" button to reset the trajectory
    ax_clear = plt.axes([0.8, 0.05, 0.1, 0.075])  # Adjust position as needed
    clear_button = Button(ax_clear, 'Clear')

    def clear_graph(event):
        """
        Callback function for the Clear button.
        Resets the trajectory data and the corresponding plot.
        """
        trajectory_x.clear()
        trajectory_y.clear()
        trajectory_z.clear()
        trajLine.set_data([], [])
        trajLine.set_3d_properties([])
        # Optionally, reset axis limits to defaults
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.draw()

    clear_button.on_clicked(clear_graph)

    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()