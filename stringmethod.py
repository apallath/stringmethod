# @author Akash Pallath
# Licensed under the MIT license, see LICENSE for details.
#
# Reference: Weinan E, "Simplified and improved string method for computing the minimum energy paths in barrier-crossing events",
# J. Chem. Phys. 126, 164103 (2007), https://doi.org/10.1063/1.2720838

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata, interp1d, UnivariateSpline
from tqdm import tqdm


class String2D:
    """
    Class containing methods to compute the minimum energy path between two
    points on an energy landscape $V$.

    Args:
        x: Array of shape (nx,) specifying x-axis coordinates of grid.
        y: Array of shape (ny,) specifying y-axis coordinates of grid.
        V: Array of shape (ny, nx) or (nx, ny) specifying energy values at each point on the grid.
            Missing values should be set to np.inf.
        indexing: Indexing of V array ('xy' specifies (ny, nx), 'ij' specifies (nx, ny); default = 'xy').

    Attributes:
        x: Array of shape (nx,) specifying x-axis coordinates of grid.
        y: Array of shape (ny,) specifying y-axis coordinates of grid.
        V: Array of shape (ny, nx) or (nx, ny) specifying energy values at each point on the grid.
        X: Grid of shape (ny, nx) or (nx, ny) containing x-coordinates of each point on the grid.
        Y: Grid of shape (ny, nx) or (nx, ny) containing y-coordinates of each point on the grid.
        gradX: Gradient in x.
        gradY: Gradient in y.
        string_traj: Trajectory showing the evolution of the string (default=[]).
        mep: Converged minimum energy path (default=None, if not converged).
    """
    def __init__(self, x, y, V, indexing='xy'):
        self.x = x
        self.y = y
        self.V = V

        # Generate grids
        self.X, self.Y = np.meshgrid(x, y, indexing=indexing)
        self.grid = np.vstack([self.X.ravel(), self.Y.ravel()]).T

        # Compute gradients
        if indexing == 'xy':
            self.gradY, self.gradX = np.gradient(self.V, self.x, self.y)
        elif indexing == 'ij':
            self.gradX, self.gradY = np.gradient(self.V, self.x, self.y)
        else:
            raise ValueError("Indexing method not recognized.")

        # String method variables
        self.string_traj = []
        self.mep = None

    def compute_mep(self, begin, end, mid=[], spline_order=2, npts=100, integrator='forward_euler', dt=0.1, tol=None, maxsteps=100, traj_every=10):
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
            spline_order: Order of spline interpolating begin, mid, and end (default=2).
            npts: Number of points between any two valuesalong the string (default=100).
            integrator: Integration scheme to use (default='forward_euler'). Options=['forward_euler'].
            dt: Integration timestep (default=0.1).
            tol: Convergence criterion; stop stepping if string has an RMSD < tol between
                consecutive steps (default = max{npts^-4, 10^-10}).
            maxsteps: Maximum number of steps to take (default=100).
            traj_every: Interval to store string trajectory (default=10).

        Returns:
            mep: Array of shape (npts, 2) specifying string images along the minimum energy path between `begin` and `end`.
        """
        # Calculate params
        if tol is None:
            tol = max([npts ** -4, 1e-10])

        # Generate initial guess
        if len(mid) > 0:
            string_x = np.linspace(begin[0], end[0], npts)
            xpts = [begin[0]] + [mpt[0] for mpt in mid] + [end[0]]
            ypts = [begin[1]] + [mpt[1] for mpt in mid] + [end[1]]
            spline = UnivariateSpline(xpts, ypts, k=spline_order)
            string_y = spline(string_x)
        else:
            string_x = np.linspace(begin[0], end[0], npts)
            string_y = np.linspace(begin[1], end[1], npts)

        string = np.vstack([string_x, string_y]).T

        # Store initial guess
        self.string_traj = []
        self.string_traj.append(string)

        # Loop
        old_string = np.zeros_like(string)

        for tstep in tqdm(range(1, maxsteps + 1)):
            # Integrator step
            if integrator == "forward_euler":
                old_string[:] = string
                string = self.step_euler(string, dt)
            else:
                raise ValueError("Invalid integrator")

            # Reparameterize string (equal arc length reparameterization)
            arclength = np.hstack([0, np.cumsum(np.linalg.norm(string[1:] - string[:-1], axis=1))])
            arclength /= arclength[-1]
            reparam_x = interp1d(arclength, string[:, 0])
            reparam_y = interp1d(arclength, string[:, 1])
            gamma = np.linspace(0, 1, npts)
            string = np.vstack([reparam_x(gamma), reparam_y(gamma)]).T

            # Store
            if tstep % traj_every == 0:
                self.string_traj.append(string)
                # Print convergence
                print("Change in string: {:.10f}".format(np.sqrt(np.mean((string - old_string) ** 2))))

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
            newstring: Array of shape (npts, 2) specifying string images after a timestep.
        """
        # Compute gradients at string points
        string_grad_x = griddata(self.grid, self.gradX.ravel(), string, method='linear')
        string_grad_y = griddata(self.grid, self.gradY.ravel(), string, method='linear')
        h = np.max(np.sqrt(string_grad_x ** 2 + string_grad_y ** 2))

        # Euler step
        string = string - dt * np.vstack([string_grad_x, string_grad_y]).T / h

        return string

    def plot_V(self, clip_min=None, clip_max=None, levels=None, cmap='RdYlBu', dpi=300):
        """
        Generates a filled contour plot of the energy landscape $V$.

        Args:
            cmap: Colormap for plot.
            levels: Levels to plot contours at (see matplotlib contour/contourf docs for details).
            dpi: DPI.
        """
        fig, ax = plt.subplots(dpi=dpi)

        V = self.V
        if clip_min is not None:
            V = V.clip(min=clip_min)
        if clip_max is not None:
            V = V.clip(max=clip_max)

        ax.contourf(self.X, self.Y, V, levels=levels, cmap=cmap)
        ax.contour(self.X, self.Y, V, levels=levels, colors="black", alpha=0.2)
        return fig, ax

    def plot_mep(self, **plot_V_kwargs):
        """
        Plots the minimum energy path on the energy landscape $V$.

        Args:
            **plot_V_kwargs: Keyword arguments for plotting the energy landscape V.
        """
        fig, ax = self.plot_V(**plot_V_kwargs)
        ax.scatter(self.mep[0, 0], self.mep[0, 1], color='C0')
        ax.scatter(self.mep[-1, 0], self.mep[-1, 1], color='C0')
        ax.plot(self.mep[:, 0], self.mep[:, 1])
        return fig, ax

    def plot_string_evolution(self, cmap=cm.gray, **plot_V_kwargs):
        """
        Plots the evolution of the string on the energy landscape $V$.

        Args:
            string_cmap: Colormap to use for plotting the evolution of the string.
            **plot_V_kwargs: Keyword arguments for plotting the energy landscape V.
        """
        fig, ax = self.plot_V(**plot_V_kwargs)
        colors = cmap(np.linspace(0, 1, len(self.string_traj)))
        for sidx, string in enumerate(self.string_traj):
            ax.plot(string[:, 0], string[:, 1], '--', color=colors[sidx])
        return fig, ax
