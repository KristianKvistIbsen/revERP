# reverp/core/mesh_handler.py
from typing import Dict, Any, Optional
import numpy as np
import trimesh
import logging
from ansys.dpf import core as dpf
from ..utils.spherical import best_fit_sphere
from ..lib.pySDEM import pySDEM
from ..lib.hansen_dpf_functions import (
    get_skin_mesh_from_ns,
    get_scoping_from_mesh,
    get_normals,
    get_normal_velocitiy_fc_from_skin_mesh,
    get_modal_normal_displacement_fc_from_skin_mesh,
    compute_faces_area
)


class MeshHandler:
    """Handles all mesh-related operations."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize MeshHandler with optional configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

    def load_model(self, model_path: str) -> dpf.Model:
        """Load the DPF model."""
        try:
            self.logger.info(f"Loading model from {model_path}")
            return dpf.Model(model_path)
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def get_skin_mesh(self, model: dpf.Model, ns: str) -> Any:
        """Get skin mesh for named selection."""
        try:
            self.logger.info(f"Getting skin mesh for named selection: {ns}")
            return get_skin_mesh_from_ns(ns, model)
        except Exception as e:
            self.logger.error(f"Failed to get skin mesh: {str(e)}")
            raise

    def fit_sphere(self, coordinates: np.ndarray) -> Dict[str, Any]:
        """Fit sphere to mesh coordinates."""
        try:
            center, radius = best_fit_sphere(coordinates)
            return {
                "center": center,
                "radius": radius
            }
        except Exception as e:
            self.logger.error(f"Sphere fitting failed: {str(e)}")
            raise

    def stl_to_dpf_mesh(self, stl_path):
        mesh = trimesh.load(stl_path, file_type='stl')
        connectivity = mesh.faces
        coordinates = mesh.vertices
        meshed_region = dpf.MeshedRegion(
            num_nodes=coordinates.shape[0],
            num_elements=connectivity.shape[0]
        )
        print("Warning: scaling factor of 1000 used to convert from mm to m")
        for i, coord in enumerate(coordinates, start=0):
            meshed_region.nodes.add_node(i, coord / 1000)

        for i, face in enumerate(connectivity, start=0):
            meshed_region.elements.add_element(i, "shell", face.tolist())
        return meshed_region, mesh.area_faces

    def process_mesh(self, model: dpf.Model, skin_mesh: Any) -> Dict[str, Any]:
        try:
            self.logger.info("Processing mesh data")

            # Get basic information of the skin mesh (skin_mesh is the mesh of the actual structure)
            nodal_normals = get_normals(skin_mesh)
            skin_mesh_scoping = get_scoping_from_mesh(skin_mesh, "Nodal")
            coordinates = skin_mesh.nodes.coordinates_field.data
            face_areas = compute_faces_area(skin_mesh,get_scoping_from_mesh(skin_mesh, "Elemental")).data
            # Get time frequencies
            tfreq = model.metadata.time_freq_support.time_frequencies
            # Ready for implementation when pyANSYS updates to handle complex time-frequency support... :(
            # tfreq = bandpass_tfreq(model.metadata.time_freq_support,self.config.get('analysis')['minimum_frequency'],self.config.get('analysis')['maximum_frequency'])

            if self.config.get('analysis')['analysis_type'] in ["harmonic","spectral"]:

                if self.config.get('SDEM_projection').get('SDEM', False):
                    self.logger.info("Performing SDEM")

                    # Midside nodes are not supported in SDEM, so these are removed
                    skin_mesh_grid = skin_mesh.grid

                    v = skin_mesh_grid.points

                    f = skin_mesh_grid.cells_dict
                    f = f[list(f.keys())[0]]
                    v_used = np.unique(f)
                    v_unused = np.setdiff1d(np.arange(len(v)), v)
                    v = v[v_used]

                    # Create mapping from old to new vertex indices
                    v_map = np.zeros(len(v_used) + len(v_unused), dtype=int)
                    v_map[v_used] = np.arange(len(v_used))

                    # Update face indices using the mapping
                    f = v_map[f]

                    # Check whether the mesh without midside nodes is genus-0 ("water-tight") using Euler's formula
                    if len(v) - 3*len(f)/2 + len(f) != 2:
                        print('The provided mesh is not a genus-0 closed triangular mesh.')
                        if self.config.get('SDEM_projection').get('shrink_wrap_stl', None) is not None:
                            print('Attempting projection to shrink-wrapped STL')

                            wrap, face_areas = self.stl_to_dpf_mesh(self.config.get('SDEM_projection').get('shrink_wrap_stl'))
                            wrap_grid = wrap.grid
                            v = wrap_grid.points
                            f = wrap_grid.cells_dict
                            f = f[list(f.keys())[0]]
                            v_used = np.unique(f)

                        else:
                            raise ValueError('Skin mesh not genous-0, and no stl file provided for shrink-wrapping.')

                    S = pySDEM.pySDEM(v, f, face_areas,
                               float(self.config.get('SDEM_projection')['SDEM_dt']),
                               float(self.config.get('SDEM_projection')['SDEM_eps']),
                               int(self.config.get('SDEM_projection')['SDEM_itt']))

                    # Scale coordinates of sphere to preserve initial surface area of the skin mesh
                    scaling_factor_sphere_area_normalization = np.sqrt(skin_mesh_grid.area / (4 * np.pi))
                    coordinates = S * scaling_factor_sphere_area_normalization

                    if 'wrap' in locals():
                        print("\n\nReading ANSYS .rst file. Please be patient...")
                        op_mapping_workflow = dpf.operators.mapping.prepare_mapping_workflow()
                        op_mapping_workflow.inputs.input_support.connect(skin_mesh)
                        op_mapping_workflow.inputs.output_support.connect(wrap)
                        op_mapping_workflow.inputs.filter_radius.connect(float(self.config.get('SDEM_projection').get('shrink_wrap_map_filter_radius')))
                        mapping_workflow = op_mapping_workflow.outputs.mapping_workflow()
                        mapping_workflow.connect(
                            'source',
                            get_normal_velocitiy_fc_from_skin_mesh(model, skin_mesh, tfreq, skin_mesh_scoping, nodal_normals)
                        )
                        mapping_workflow.progress_bar=False
                        vel_disp_fc = mapping_workflow.get_output('target', output_type="fields_container")
                        field_mesh = vel_disp_fc[0].meshed_region
                        print("Done... phew...")
                    else:
                        print("Reading ANSYS .rst file. Please be patient...")
                        vel_disp_fc = get_normal_velocitiy_fc_from_skin_mesh(
                            model, skin_mesh, tfreq, skin_mesh_scoping, nodal_normals
                        ).eval()
                        field_mesh = vel_disp_fc[0].meshed_region
                        print("Done... phew...")
                    spherical_skin_mesh = field_mesh.deep_copy()
                    spherical_skin_mesh.nodes.coordinates_field.data[v_used] = coordinates
                else: #Not using SDEM
                # Get normal velocities on skin mesh
                    print("Reading ANSYS .rst file. Please be patient...")
                    vel_disp_fc = get_normal_velocitiy_fc_from_skin_mesh(
                        model, skin_mesh, tfreq, skin_mesh_scoping, nodal_normals
                    ).eval()
                    field_mesh = vel_disp_fc[0].meshed_region
                    print("Done... phew...")

            elif self.config.get('analysis')['analysis_type'] == "modal":
                if self.config.get('SDEM_projection').get('SDEM', False):
                    self.logger.info("Performing SDEM")

                    # Midside nodes are not supported in SDEM, so these are removed
                    skin_mesh_grid = skin_mesh.grid
                    v = skin_mesh_grid.points
                    f = skin_mesh_grid.cells_dict
                    f = f[list(f.keys())[0]]
                    v_used = np.unique(f)
                    v_unused = np.setdiff1d(np.arange(len(v)), v)
                    v = v[v_used]

                    # Create mapping from old to new vertex indices
                    v_map = np.zeros(len(v_used) + len(v_unused), dtype=int)
                    v_map[v_used] = np.arange(len(v_used))

                    # Update face indices using the mapping
                    f = v_map[f]

                    # Check whether the mesh without midside nodes is genus-0 ("water-tight") using Euler's formula
                    if len(v) - 3*len(f)/2 + len(f) != 2:
                        print('The provided mesh is not a genus-0 closed triangular mesh.')
                        if self.config.get('SDEM_projection').get('shrink_wrap_stl', None) is not None:
                            print('Attempting projection to shrink-wrapped STL')

                            wrap = self.stl_to_dpf_mesh(self.config.get('SDEM_projection').get('shrink_wrap_stl'))

                            wrap_grid = wrap.grid
                            v = wrap_grid.points
                            f = wrap_grid.cells_dict
                            f = f[list(f.keys())[0]]
                            v_used = np.unique(f)

                        else:
                            raise ValueError('Skin mesh not genous-0, and no stl file provided for shrink-wrapping.')

                    S = pySDEM.pySDEM(v, f, face_areas,
                               float(self.config.get('SDEM_projection')['SDEM_dt']),
                               float(self.config.get('SDEM_projection')['SDEM_eps']),
                               int(self.config.get('SDEM_projection')['SDEM_itt']))

                    # Scale coordinates of sphere to preserve initial surface area of the skin mesh
                    scaling_factor_sphere_area_normalization = np.sqrt(skin_mesh_grid.area / (4 * np.pi))
                    coordinates = S * scaling_factor_sphere_area_normalization

                    if 'wrap' in locals():
                        print("Reading ANSYS .rst file. Please be patient...")
                        op_mapping_workflow = dpf.operators.mapping.prepare_mapping_workflow()
                        op_mapping_workflow.inputs.input_support.connect(skin_mesh)
                        op_mapping_workflow.inputs.output_support.connect(wrap)
                        op_mapping_workflow.inputs.filter_radius.connect(float(self.config.get('SDEM_projection').get('shrink_wrap_map_filter_radius')))
                        mapping_workflow = op_mapping_workflow.outputs.mapping_workflow()
                        mapping_workflow.connect(
                            'source',
                            get_modal_normal_displacement_fc_from_skin_mesh(model, skin_mesh, tfreq, skin_mesh_scoping, nodal_normals)
                        )
                        mapping_workflow.progress_bar=False
                        vel_disp_fc = mapping_workflow.get_output('target', output_type="fields_container")
                        field_mesh = vel_disp_fc[0].meshed_region
                        print("Done... phew...")
                    else:
                        print("Reading ANSYS .rst file. Please be patient...")
                        vel_disp_fc = get_modal_normal_displacement_fc_from_skin_mesh(
                            model, skin_mesh, tfreq, skin_mesh_scoping, nodal_normals
                        )
                        field_mesh = vel_disp_fc[0].meshed_region
                        print("Done... phew...")
                    spherical_skin_mesh = field_mesh.deep_copy()
                    spherical_skin_mesh.nodes.coordinates_field.data[v_used] = coordinates

                else:
                    print("Reading ANSYS .rst file. Please be patient...")
                    # Get normal displacements for modal analysis
                    vel_disp_fc = get_modal_normal_displacement_fc_from_skin_mesh(
                        model,
                        skin_mesh,
                        tfreq,
                        skin_mesh_scoping,
                        nodal_normals
                    ).eval()
                    field_mesh = vel_disp_fc[0].meshed_region
                    print("Done... phew...")
            else:
                self.logger.error(f"Failed to extract velocity; unknown analysis_type: {self.config.get('analysis')['analysis_type']}")


            return {
                "coor": coordinates,
                "normals": nodal_normals,
                "skin_mesh": skin_mesh,
                "skin_mesh_scoping": get_scoping_from_mesh(skin_mesh, "Nodal"),
                "vel_disp_fc": vel_disp_fc,
                "field_mesh": field_mesh,
                "field_mesh_scoping": get_scoping_from_mesh(field_mesh, "Nodal"),
                "n_nodes": field_mesh.nodes.n_nodes,
                "spherical_skin_mesh": spherical_skin_mesh if self.config.get('SDEM_projection').get('SDEM', False) else None,
                "tfreq": tfreq,
                "v_used": v_used if self.config.get('SDEM_projection').get('SDEM', False) else None
            }

        except Exception as e:
            self.logger.error(f"Failed to process mesh: {str(e)}")
            raise