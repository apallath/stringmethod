# @author Akash Pallath
# Licensed under the MIT license, see LICENSE for details.
#
# Reference: Weinan E, "Simplified and improved string method for computing the minimum energy paths in barrier-crossing events",
# J. Chem. Phys. 126, 164103 (2007), https://doi.org/10.1063/1.2720838

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


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
        gradX: Gradient in x.
        gradY: Gradient in y.
        string_traj: Trajectory showing the evolution of the string (default=[]).
        mep: Converged minimum energy path (default=None, if not converged).
    """
    def __init__(self, x, y, V):
        self.x = x
        self.y = y
        self.V = V

        # Generate grids
        self.X, self.Y = np.meshgrid(x, y)

        # Compute gradients
        self.gradX, self.gradY = np.gradient(V, self.x, self.y)

        # String method variables
        self.string_traj = []
        self.mep = None

    def compute_mep(self, begin, end, mid=[], npts=100, integrator='forward_euler', dt=0.01, tol=1e-3, maxsteps=1000, traj_every=50):
        """
        Computes the minimum free energy path connecting the points `begin`
        and `end`. Midpoints passed through `mid` are used to generate
        an initial guess (a quadratic spline which interpolates through all the points).
        If no midpoints are defined, then the initial guess is a line connecting `begin`
        and `end`.

        Args:
            begin: Array of shape (2,) specifying starting point of the string.
            end: Array of shape (2,) specifying end point of the string.
            mid: List of arrays of shape (2,) specifying points between `begin` and `end`
                to use for generating an initial guess of the minimum energy path (default=[]).
            npts: Number of points between any two valuesalong the string (default=100).
            integrator: Integration scheme to use (default='forward_euler'). Options=['forward_euler'].
            dt: Integration timestep (default=0.01).
            tol: Convergence criterion; stop stepping if string has an RMSD < tol between
                consecutive steps (default = 1e-3).
            maxsteps: Maximum number of steps to take (default=1000).
            traj_every: Interval to store string trajectory (default=50).

        Returns:
            mep: Array of shape (npts, 2) specifying string images along the minimum energy path between `begin` and `end`.
        """
        # Generate initial guess
        if len(mid) > 0:
            string_x = np.linspace(begin[0], end[0], npts)
            xpts = [begin[0]] + [mpt[0] for mpt in mid] + [end[0]]
            ypts = [begin[1]] + [mpt[1] for mpt in mid] + [end[1]]
            spline = UnivariateSpline(xpts, ypts, k=2)  # quadratic spline
            string_y = spline(string_x)
        else:
            string_x = np.linspace(begin[0], end[0], npts)
            string_y = np.linspace(begin[1], end[1], npts)

        string = np.vstack([string_x, string_y]).T

        # Store initial guess
        self.string_traj.append(string)

        # Loop
        for tstep in range(1, maxsteps + 1):
            # Integrator step
            if integrator == "forward_euler":
                old_string = string
                string = self.step_euler(string, dt)
            else:
                raise ValueError("Invalid integrator")

            # Reparameterize string
            # TODO

            # Store
            if tstep % traj_every == 0:
                self.string_traj.append(string)

            # Test for convergence
            if np.sqrt(np.mean((string - old_string) ** 2)) < tol:
                break

        # Store minimum energy path
        self.mep = string

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

    def plot_V(self, levels=None, cmap='RdYlBu', dpi=300):
        """
        Generates a filled contour plot of the energy landscape $V$.

        Args:
            cmap: Colormap for plot.
            levels: Levels to plot contours at (see matplotlib contour/contourf docs for details).
            dpi: DPI.
        """
        fig, ax = plt.subplots(dpi=dpi)
        ax.contourf(self.X, self.Y, self.V, levels=levels, cmap=cmap)
        ax.contour(self.X, self.Y, self.V, levels=levels, colors="black", alpha=0.2)
        return fig, ax

    def plot_mep(self, **plot_V_kwargs):
        """
        Plots the minimum energy path on the energy landscape $V$.

        Args:
            **plot_V_kwargs: Keyword arguments for plotting the energy landscape V.
        """
        fig, ax = self.plot_V(**plot_V_kwargs)
        ax.plot(self.mep[:, 0], self.mep[:, 1])
        return fig, ax

    def plot_string_evolution(self, string_cmap='gray', **plot_V_kwargs):
        """
        Plots the evolution of the string on the energy landscape $V$.

        Args:
            string_cmap: Colormap to use for plotting the evolution of the string.
            **plot_V_kwargs: Keyword arguments for plotting the energy landscape V.
        """
        fig, ax = self.plot_V(**plot_V_kwargs)
        for string in self.string_traj:
            ax.plot(string[:, 0], string[:, 1])
        return fig, ax
