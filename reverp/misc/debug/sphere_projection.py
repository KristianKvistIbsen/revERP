#!/usr/bin/env python3
# sphere_visualization.py

import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Optional
import matplotlib.pyplot as plt

def create_sphere_mesh(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    grid_values: np.ndarray,
    radius: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a spherical mesh from latitude, longitude, and grid values.

    Args:
        grid_lats: 2D array of latitudes
        grid_lons: 2D array of longitudes
        grid_values: 2D array of values to plot on sphere
        radius: Sphere radius (default: 1.0)

    Returns:
        Tuple containing x, y, z coordinates and normalized values
    """
    # Convert to radians
    lats_rad = np.radians(grid_lats)
    lons_rad = np.radians(grid_lons)

    # Create sphere coordinates
    x = radius * np.cos(lats_rad) * np.cos(lons_rad)
    y = radius * np.cos(lats_rad) * np.sin(lons_rad)
    z = radius * np.sin(lats_rad)

    # Normalize values for coloring
    values_norm = (grid_values - grid_values.min()) / (grid_values.max() - grid_values.min())

    return x, y, z, values_norm

def plot_sphere_interactive(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    grid_values: np.ndarray,
    radius: float = 1.0,
    title: str = "Spherical Mode Shape",
    colormap: str = "viridis",
    show_colorbar: bool = True
) -> None:
    """
    Create an interactive 3D plot of values on a sphere using plotly.

    Args:
        grid_lats: 2D array of latitudes
        grid_lons: 2D array of longitudes
        grid_values: 2D array of values to plot on sphere
        radius: Sphere radius
        title: Plot title
        colormap: Name of the colormap to use
        show_colorbar: Whether to show the colorbar
    """
    x, y, z, values = create_sphere_mesh(grid_lats, grid_lons, grid_values, radius)

    # Create figure
    fig = go.Figure()

    # Add surface plot
    fig.add_surface(
        x=x, y=y, z=z,
        surfacecolor=values,
        colorscale=colormap,
        showscale=show_colorbar,
        colorbar=dict(
            title="Normalized Value",
            titleside="right"
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        width=800,
        height=800
    )

    # Show figure
    fig.show()

def plot_sphere_static(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    grid_values: np.ndarray,
    radius: float = 1.0,
    title: str = "Spherical Mode Shape",
    colormap: str = "viridis",
    view_angles: Tuple[float, float] = (30, 45),
    save_path: Optional[str] = None
) -> None:
    """
    Create a static 3D plot of values on a sphere using matplotlib.

    Args:
        grid_lats: 2D array of latitudes
        grid_lons: 2D array of longitudes
        grid_values: 2D array of values to plot on sphere
        radius: Sphere radius
        title: Plot title
        colormap: Name of the colormap to use
        view_angles: Tuple of (elevation, azimuth) viewing angles in degrees
        save_path: Optional path to save the plot
    """
    x, y, z, values = create_sphere_mesh(grid_lats, grid_lons, grid_values, radius)

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create scatter plot
    scatter = ax.scatter(
        x, y, z,
        c=values,
        cmap=colormap,
        marker='o'
    )

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Set viewing angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])

    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Normalized Value')

    # Set aspect ratio to be equal
    ax.set_box_aspect([1,1,1])

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def main():
    # """Example usage of the sphere plotting functions."""
    # # Create example data
    # nlat, nlon = 180, 360
    # lats = np.linspace(-90, 90, nlat)
    # lons = np.linspace(0, 360, nlon)
    # grid_lats, grid_lons = np.meshgrid(lats, lons, indexing='ij')

    # # Create example mode shape (spherical harmonic Y_2^1)
    # theta = np.radians(90 - grid_lats)
    # phi = np.radians(grid_lons)
    # grid_values = np.sin(theta)**2 * np.cos(theta) * np.cos(phi)

    # # Create interactive plot
    # plot_sphere_interactive(
    #     grid_lats,
    #     grid_lons,
    #     grid_values,
    #     radius=1.0,
    #     title="Example Mode Shape (Y_2^1)",
    #     colormap="viridis"
    # )

    # Create static plot
    plot_sphere_static(
        grid_lats,
        grid_lons,
        grid_values,
        radius=1.0,
        title="Example Mode Shape (Y_2^1)",
        colormap="viridis",
        view_angles=(30, 45),
        save_path="sphere_plot.png"
    )

if __name__ == "__main__":
    main()
    
    
    
                            # aaaa = grid_values.real

                        # from ...misc.debug.sphere_projection import plot_sphere_interactive, plot_sphere_static
                        # # In your modal analysis code, after computing grid_values:
                        # plot_sphere_interactive(
                        #     grid_lats=grid_lats,
                        #     grid_lons=grid_lons,
                        #     grid_values=grid_values.real,
                        #     radius=sphere_data["radius"],
                        #     title=f"Mode {id_f + 1} Shape",
                        #     colormap="jet"
                        # )

                        # # # Or for static plotting:
                        # from ...misc.debug.sphere_projection import plot_sphere_interactive, plot_sphere_static
                        # plot_sphere_static(
                        #     grid_lats=grid_lats,
                        #     grid_lons=grid_lons,
                        #     grid_values=grid_values.imag,
                        #     radius=sphere_data["radius"],
                        #     title=f"Mode {id_f + 1} Shape",
                        #     colormap="jet",
                        #     view_angles=(30, 45),
                        #     save_path=f"mode_{id_f+1}_shape.png"
                        # )