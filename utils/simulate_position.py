import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class RobotSimulation:
    """
    A class to simulate the position of a robot in 3D space.
    This class creates an animation of the robot's position using the `matplotlib` library.
    """

    def __init__(self, list1, list2, list3):
        """
        Initialize the RobotSimulation class.

        Args:
            list1 (list): The first list.
            list2 (list): The second list.
            list3 (list): The third list.
        """
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        self.base = [[(0., 0., 0)]] * len(list1)
        self.num_frames = len(list1)
        self.max_range = None

        # Create a 3D plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_chain(self, chain, color='b'):
        """
        Plots a chain of points in 3D space.

        Args:
            chain (list): List of tuples representing the coordinates of each point in the chain.
            color (str, optional): Color of the plotted chain. Defaults to 'b'.
        """
        x, y, z = zip(*chain)
        self.ax.plot(x, y, z, color=color, marker='o', linestyle='-', markersize=5)

    def update(self, frame):
        """
        Update the plot with the data for the given frame.

        Parameters:
        - frame (int): The frame number to update the plot with.

        Returns:
        None
        """
        self.ax.cla()  # Clear the previous plot
        self.plot_chain(self.list1[frame], color='b')
        self.plot_chain(self.list2[frame], color='g')
        self.plot_chain(self.list3[frame], color='r')
        self.plot_chain(self.base[frame], color='orange')
        self.ax.set_xlabel('X [mm]')
        self.ax.set_ylabel('Y [mm]')
        self.ax.set_zlabel('Z [mm]')
        self.ax.set_title(f'Frame {frame + 1}')

        # Find the maximum range of coordinates across all lists for the current frame
        all_lists = [self.list1[frame], self.list2[frame], self.list3[frame], self.base[frame]]
        self.max_range = max(
            max(max(p[i] for p in lst) for i in range(3)) for lst in all_lists)

        # Set equal scales for all axes by adjusting the limits
        self.ax.set_xlim(-self.max_range, self.max_range)
        self.ax.set_ylim(-self.max_range, self.max_range)
        self.ax.set_zlim(0, self.max_range)

    def animate(self):
        """
        Animates the position simulation.

        This function creates an animation of the position simulation using the `FuncAnimation` class from the `matplotlib.animation` module.
        It calls the `update` method for each frame and displays the animation using `plt.show()`.

        Parameters:
            None

        Returns:
            None
        """
        ani = FuncAnimation(self.fig, self.update, frames=self.num_frames, interval=1000)
        plt.show()
