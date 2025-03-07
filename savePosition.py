# savePosition.py
# by pongwsl on feb 28, 2025
# save postion data (x,y,z) from tools/getPostion.py to a .txt file

#!/usr/bin/env python3
import time
import datetime
from tools.getPosition import getPosition

def main():
    # Initialize the position generator from getPosition()
    position_generator = getPosition()
    
    # List to store the recorded positions
    positions = []
    
    # Record positions for 10 seconds
    start_time = time.time()
    print("Recording positions for 10 seconds...")
    while time.time() - start_time < 10:
        try:
            # Get the next position from the generator
            pos = next(position_generator)
            positions.append(pos)
            # Sleep briefly to avoid a busy loop (adjust if needed)
            time.sleep(0.01)
        except StopIteration:
            # If the generator stops yielding, exit the loop
            break

    # Generate a filename with the current date and time (e.g., positions_20250228_153045.txt)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"positions_{timestamp}.txt"
    
    # Save the recorded positions to the file
    with open(filename, "w") as file:
        for pos in positions:
            file.write(f"{pos}\n")
    
    print(f"Saved positions to {filename}")

if __name__ == "__main__":
    main()