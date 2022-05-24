import numpy as np
import matplotlib.pyplot as plt


class String2D:
    """
    Class containing methods to compute the minimum energy path between two
    points on an energy landscape $V$.

    Args:
        x: Array of shape (nx,) specifying x-axis coordinates.
        y: Array of shape (ny,) specifying y-axis coordinates
        V: Array of shape (nx, ny) specifying points along the x- and y- axes. Missing values should be set to np.inf.

    Attributes:
        x: Array of shape (nx,) specifying x-axis coordinates.
        y: Array of shape (ny,) specifying y-axis coordinates.
        V: Array of shape (nx, ny) specifying points along the x- and y- axes.
        X: Grid of shape (nx, ny) containing x-coordinates.
        Y: Grid of shape (nx, ny) containing y-coordinates.
        string_traj: Trajectory showing the evolution of the string (default=[]).
        mep: Converged minimum energy path (default=None, if not converged).
    """
    def __init__(self, x, y, V):
        self.x = x
        self.y = y
        self.V = V
        self.X, self.Y = np.meshgrid(x, y)
        self.string_traj = []
        self.mep = None

    def compute_mep(self, begin, end, mid=[], npts=100, integrator='forward_euler', dt=0.01, traj_every=50):
        """
        Computes the minimum free energy path connecting the points `begin`
        and `end`. Points passed through `mid` are used to generate
        an initial guess (which interpolates through all the points).

        Args:
            begin: Array of shape (2,) specifuing starting point of the string.
            end: Array of shape (2,) specifying end point of the string.
            mid: List of arrays of shape (2,) specifying points between `begin` and `end`
                to use for generating an initial guess of the minimum energy path (default=[]).
            npts: Number of points along the string (default=100).
            integrator: Integration scheme to use (default='forward_euler'). Options=['forward_euler']
            dt: Integration timestep (default=0.01)
            traj_every: Interval to store string trajectory (default=50).

        Returns:
            mep: Array of shape (npts, 2) specifying string images along the minimum energy path between `begin` and `end`.
        """
        pass

    def step_euler(self, string, dt):
        """
        Evolves string images in time in response to forces calculated from the energy landscape.

        Args:
            string: Array of shape (npts, 2) specifying string images at the previous timestep.
            dt: Timestep.

        Returns:
            newstring: Array of shape (ntps, 2) specifying stirng images after a timestep.
        """
        pass

    def plot_V(self, cmap='', contourvals=None, dpi=300):
        """
        Generates a filled contour plot of the energy landscape $V$.

        Args:
            cmap: Colormap for plot.
            contourvals: Values to plot contours at.
            dpi: DPI.
        """
        pass

    def plot_mep(self, **plot_V_kwargs):
        """
        Plots the minimum energy path on the energy landscape $V$.

        Args:
            **plot_V_kwargs: Keyword arguments for plotting the energy landscape V.
        """
        pass

    def plot_string_evolution(self, string_cmap='gray', **plot_V_args):
        """
        Plots the evolution of the string on the energy landscape $V$.

        Args:
            string_cmap: Colormap to use for plotting the evolution of the string.
            **plot_V_kwargs: Keyword arguments for plotting the energy landscape V.
        """
        pass
