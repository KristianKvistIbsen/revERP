import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple

def visualize_sphere_fit(
    coordinates: np.ndarray,
    center: np.ndarray,
    radius: float,
    nodal_areas: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Sphere Fitting Visualization",
    show_mesh: bool = True,
    show_sphere: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the mesh points and fitted sphere.

    Args:
        coordinates: Node coordinates (n_nodes × 3)
        center: Sphere center coordinates [x, y, z]
        radius: Sphere radius
        nodal_areas: Optional array of nodal areas for coloring points (n_nodes,)
        save_path: Optional path to save the figure
        title: Title for the plot
        show_mesh: Whether to show the mesh points
        show_sphere: Whether to show the fitted sphere

    Returns:
        Tuple containing:
        - matplotlib Figure object
        - matplotlib Axes object
    """
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot mesh points if requested
    if show_mesh:
        if nodal_areas is not None:
            # Normalize areas for coloring
            areas_normalized = nodal_areas / np.max(nodal_areas)
            scatter = ax.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 2],
                c=areas_normalized,
                cmap='viridis',
                alpha=0.6,
                label='Mesh Points'
            )
            plt.colorbar(scatter, ax=ax, label='Normalized Nodal Area')
        else:
            ax.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 2],
                color='blue',
                alpha=0.6,
                label='Mesh Points'
            )

    # Plot fitted sphere if requested
    if show_sphere:
        # Create sphere mesh
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

        # Plot sphere surface
        ax.plot_surface(
            x, y, z,
            color='red',
            alpha=0.1,
            label='Fitted Sphere'
        )

        # Plot sphere center
        ax.scatter(
            [center[0]],
            [center[1]],
            [center[2]],
            color='red',
            marker='*',
            s=200,
            label='Sphere Center'
        )

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set title
    ax.set_title(title)

    # Make the plot look nice
    ax.grid(True)
    ax.legend()

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Add info text
    info_text = f'Radius: {radius:.2f}\nCenter: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]'
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def visualize_sphere_fit_error(
    coordinates: np.ndarray,
    center: np.ndarray,
    radius: float,
    nodal_areas: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Sphere Fitting Error Distribution"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the error distribution of the sphere fitting.

    Args:
        coordinates: Node coordinates (n_nodes × 3)
        center: Sphere center coordinates [x, y, z]
        radius: Sphere radius
        nodal_areas: Optional array of nodal areas for weighting
        save_path: Optional path to save the figure
        title: Title for the plot

    Returns:
        Tuple containing:
        - matplotlib Figure object
        - matplotlib Axes object
    """
    # Calculate distances from points to center
    distances = np.linalg.norm(coordinates - center, axis=1)

    # Calculate errors (difference from radius)
    errors = distances - radius

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    if nodal_areas is not None:
        # Create weighted histogram
        ax.hist(errors, bins=50, weights=nodal_areas, alpha=0.7)
        plt.ylabel('Total Area')
    else:
        # Create regular histogram
        ax.hist(errors, bins=50, alpha=0.7)
        plt.ylabel('Count')

    # Add statistics
    weighted_mean = np.average(errors, weights=nodal_areas) if nodal_areas is not None else np.mean(errors)
    weighted_std = np.sqrt(np.average((errors - weighted_mean)**2, weights=nodal_areas)) if nodal_areas is not None else np.std(errors)

    stats_text = f'Mean Error: {weighted_mean:.2e}\nStd Dev: {weighted_std:.2e}'
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8))

    # Customize plot
    ax.set_xlabel('Distance Error (Actual - Radius)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_sphere_analysis(
    coordinates: np.ndarray,
    center: np.ndarray,
    radius: float,
    nodal_areas: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None
) -> None:
    """
    Create a complete analysis of the sphere fitting with multiple visualizations.

    Args:
        coordinates: Node coordinates (n_nodes × 3)
        center: Sphere center coordinates [x, y, z]
        radius: Sphere radius
        nodal_areas: Optional array of nodal areas
        save_dir: Optional directory to save the figures
    """
    # Create 3D visualization
    fig_3d, _ = visualize_sphere_fit(
        coordinates, center, radius, nodal_areas,
        save_path=f"{save_dir}/sphere_fit_3d.png" if save_dir else None,
        title="Sphere Fitting - 3D Visualization"
    )

    # Create error distribution plot
    fig_error, _ = visualize_sphere_fit_error(
        coordinates, center, radius, nodal_areas,
        save_path=f"{save_dir}/sphere_fit_error.png" if save_dir else None,
        title="Sphere Fitting - Error Distribution"
    )

    plt.show()



# # Basic usage
# plot_sphere_analysis(
#     coordinates=skin_mesh_coor,
#     center=optimal_center,
#     radius=optimal_radius,
#     nodal_areas=nodal_area_matrix,
#     save_dir="output_directory"  # optional
# )

# # Or for more control, use individual functions
# fig, ax = visualize_sphere_fit(
#     coordinates=coordinates,
#     center=optimal_center,
#     radius=optimal_radius,
#     nodal_areas=nodal_areas,
#     show_mesh=True,
#     show_sphere=True
# )
# plt.show()

# # View error distribution
# fig, ax = visualize_sphere_fit_error(
#     coordinates=skin_mesh_coor,
#     center=optimal_center,
#     radius=optimal_radius,
#     nodal_areas=nodal_area_matrix
# )
# plt.show()