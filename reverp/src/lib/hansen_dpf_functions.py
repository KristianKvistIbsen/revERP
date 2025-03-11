from ansys.dpf import core as dpf
import numpy as np

# =============================================================================
# # Named Selection Utilities
# =============================================================================
def get_skin_mesh_from_ns(NS,model):
    # Get Mesh Scoping based on named selection
    opNamedSel = dpf.operators.scoping.on_named_selection()
    opNamedSel.inputs.named_selection_name.connect(NS)
    opNamedSel.inputs.int_inclusive.connect(0)
    opNamedSel.inputs.data_sources.connect(model)
    opNamedSel.inputs.requested_location.connect('Nodal')

    op_skin_mesh = dpf.operators.mesh.skin()
    op_skin_mesh.inputs.mesh.connect(model.metadata.meshed_region)
    op_skin_mesh.inputs.mesh_scoping.connect(opNamedSel)

    return op_skin_mesh.outputs.mesh()
def get_solid_mesh_from_ns(NS,model):
    # Get Mesh Scoping based on named selection
    opNamedSel = dpf.operators.scoping.on_named_selection()
    opNamedSel.inputs.named_selection_name.connect(NS)
    opNamedSel.inputs.int_inclusive.connect(0)
    opNamedSel.inputs.data_sources.connect(model)
    opNamedSel.inputs.requested_location.connect('Nodal')

    op_mesh = dpf.operators.mesh.from_scoping() # operator instantiation
    op_mesh.inputs.scoping.connect(opNamedSel)
    op_mesh.inputs.mesh.connect(model.metadata.meshed_region)

    return op_mesh.outputs.mesh()

# =============================================================================
# # Mesh Utilities
# =============================================================================
def get_normals(skin_mesh):
    # Get normal vectors at each nodal location
    op = dpf.operators.geo.normals_provider_nl()
    op.inputs.mesh.connect(skin_mesh)
    op.inputs.requested_location.connect("Nodal")

    return op.outputs.field()
def get_nodal_area_matrix(skin_mesh):
    op = dpf.operators.geo.elements_volume() # operator instantiation
    op.inputs.mesh.connect(skin_mesh)
    areas_elem = op.outputs.field()
    areas_elem.meshed_region = skin_mesh
    op = dpf.operators.averaging.to_nodal() # operator instantiation
    op.inputs.field.connect(areas_elem)

    # my_field = op.outputs.field()
    # areas_nodes = my_field.data/2

    return op.outputs.field()
def get_scoping_from_mesh(mesh,location):
    # Get scoping on skin mesh nodes + misc skin info
    opScopingFromMesh = dpf.operators.scoping.from_mesh()
    opScopingFromMesh.inputs.mesh.connect(mesh)
    opScopingFromMesh.inputs.requested_location.connect(location)

    return opScopingFromMesh.outputs.scoping()

# Metadata Utilities
def get_interface_named_selections(model):
    # Filter out named selections starting with '_'
    NS_contact = [NS for NS in model.metadata.available_named_selections if NS.endswith('CONTACT')]
    NS_target = [NS for NS in model.metadata.available_named_selections if NS.endswith('TARGET')]

    return NS_contact, NS_target

# =============================================================================
# # DPF Field Extractors
# =============================================================================
def get_displacement_from_mesh(model,mesh,tfreq):

    mesh_scoping = get_scoping_from_mesh(mesh,"Nodal")

    # Get xyz velocities at all nodal locations on skin mesh
    op_displacements = dpf.operators.result.displacement()
    op_displacements.inputs.time_scoping.connect(tfreq)
    op_displacements.inputs.mesh_scoping.connect(mesh_scoping)
    op_displacements.inputs.data_sources.connect(model)
    op_displacements.inputs.mesh.connect(mesh)
    return op_displacements.outputs.fields_container()



def compute_faces_area(skin_mesh,skin_mesh_scoping):
    op = dpf.operators.geo.elements_volume() # operator instantiation
    op.inputs.mesh.connect(skin_mesh)
    op.inputs.mesh_scoping.connect(skin_mesh_scoping)
    return op.outputs.field()



def compute_nodal_area_matrix(skin_mesh):
    op = dpf.operators.geo.elements_volume() # operator instantiation
    op.inputs.mesh.connect(skin_mesh)
    areas_elem = op.outputs.field()
    areas_elem.meshed_region = skin_mesh
    op = dpf.operators.averaging.to_nodal() # operator instantiation
    op.inputs.field.connect(areas_elem)

    areas_nodes = op.outputs.field()
    return areas_nodes
def get_velocity_from_mesh(model,mesh,tfreq):

    mesh_scoping = get_scoping_from_mesh(mesh,'Nodal')

    # Get xyz velocities at all nodal locations on skin mesh
    op_velocity = dpf.operators.result.velocity()
    op_velocity.inputs.time_scoping.connect(tfreq)
    op_velocity.inputs.mesh_scoping.connect(mesh_scoping)
    op_velocity.inputs.data_sources.connect(model)
    op_velocity.inputs.mesh.connect(mesh)

    return op_velocity.outputs.fields_container()
def get_stress_from_mesh(model,mesh,tfreq):

    mesh_scoping = get_scoping_from_mesh(mesh,'Nodal')

    op_stress = dpf.operators.result.stress() # operator instantiation
    op_stress.inputs.time_scoping.connect(tfreq)# optional
    op_stress.inputs.mesh_scoping.connect(mesh_scoping)# optional
    op_stress.inputs.data_sources.connect(model)
    op_stress.inputs.mesh.connect(mesh)# optional

    return op_stress.outputs.fields_container()
def get_normal_velocities(model,skin_mesh,tfreq,skin_mesh_scoping,nodal_normals):
    # Get xyz velocities at all nodal locations on skin mesh
    op_velocities = dpf.operators.result.velocity()
    op_velocities.inputs.time_scoping.connect(tfreq)
    op_velocities.inputs.mesh_scoping.connect(skin_mesh_scoping)
    op_velocities.inputs.data_sources.connect(model)
    op_velocities.inputs.mesh.connect(skin_mesh)

    # Get nodal normal velocities on skin mesh by dot product + misc field info
    opDot = dpf.operators.math.generalized_inner_product_fc()
    opDot.inputs.field_or_fields_container_A.connect(op_velocities)
    opDot.inputs.field_or_fields_container_B.connect(nodal_normals)
    return opDot.outputs.fields_container()

def get_normal_velocitiy_fc_from_skin_mesh(model,skin_mesh,tfreq,skin_mesh_scoping,nodal_normals):
    # Get xyz velocities at all nodal locations on skin mesh
    op_velocities = dpf.operators.result.velocity()
    op_velocities.inputs.time_scoping.connect(tfreq)
    op_velocities.inputs.mesh_scoping.connect(skin_mesh_scoping)
    op_velocities.inputs.data_sources.connect(model)
    op_velocities.inputs.mesh.connect(skin_mesh)

    # Get nodal normal velocities on skin mesh by dot product + misc field info
    opDot = dpf.operators.math.generalized_inner_product_fc()
    opDot.inputs.field_or_fields_container_A.connect(op_velocities)
    opDot.inputs.field_or_fields_container_B.connect(nodal_normals)
    return opDot

# def get_modal_normal_displacement_fc_from_skin_mesh(model,skin_mesh,tfreq,skin_mesh_scoping,nodal_normals):
#     op_disp = dpf.operators.result.displacement() # operator instantiation
#     op_disp.inputs.time_scoping.connect(tfreq)# optional
#     op_disp.inputs.mesh_scoping.connect(skin_mesh_scoping)# optional
#     op_disp.inputs.data_sources.connect(model)
#     op_disp.inputs.mesh.connect(skin_mesh)# optional
#     # my_fields_container = op_disp.outputs.fields_container()

#     # Get nodal normal velocities on skin mesh by dot product + misc field info
#     opDot = dpf.operators.math.generalized_inner_product_fc()
#     opDot.inputs.field_or_fields_container_A.connect(op_disp)
#     opDot.inputs.field_or_fields_container_B.connect(nodal_normals)
#     return opDot

def get_modal_normal_displacement_fc_from_skin_mesh(model, skin_mesh, tfreq, skin_mesh_scoping, nodal_normals):
    # Get displacement field for positive frequencies
    op_disp = dpf.operators.result.displacement()
    op_disp.inputs.time_scoping.connect(tfreq)
    op_disp.inputs.mesh_scoping.connect(skin_mesh_scoping)
    op_disp.inputs.data_sources.connect(model)
    op_disp.inputs.mesh.connect(skin_mesh)

    # Project displacements onto normal directions
    op_dot = dpf.operators.math.generalized_inner_product_fc()
    op_dot.inputs.field_or_fields_container_A.connect(op_disp)
    op_dot.inputs.field_or_fields_container_B.connect(nodal_normals)

    return op_dot

def get_normal_displacements(model,skin_mesh,tfreq,skin_mesh_scoping,nodal_normals):
    op_displacements = dpf.operators.result.displacement()
    op_displacements.inputs.time_scoping.connect(tfreq)
    op_displacements.inputs.mesh_scoping.connect(skin_mesh_scoping)
    op_displacements.inputs.data_sources.connect(model)
    op_displacements.inputs.mesh.connect(skin_mesh)

    opDot = dpf.operators.math.generalized_inner_product_fc()
    opDot.inputs.field_or_fields_container_A.connect(op_displacements)
    opDot.inputs.field_or_fields_container_B.connect(nodal_normals)
    return opDot.outputs.fields_container()
# def get_elastic_strain_energy_density(model,mesh,tfreq):


def bandpass_tfreq(tfreq,tf_min,tf_max):
    op_bandpass = dpf.operators.filter.timefreq_band_pass() # operator instantiation
    op_bandpass.inputs.time_freq_support.connect(tfreq)
    op_bandpass.inputs.min_threshold.connect(float(tf_min))
    op_bandpass.inputs.max_threshold.connect(float(tf_max))
    my_time_freq_support = op_bandpass.outputs.time_freq_support()
    # my_scoping = op_bandpass.outputs.scoping()

    return my_time_freq_support



# =============================================================================
# Compute workflows
# =============================================================================
def compute_flux_through_skin_mesh(flux_vector_field,skin_mesh):

    skin_mesh_scoping = get_scoping_from_mesh(skin_mesh,"Elemental")

    op = dpf.operators.mapping.solid_to_skin() # operator instantiation
    op.inputs.field.connect(flux_vector_field)
    op.inputs.mesh.connect(skin_mesh)
    op.inputs.solid_mesh.connect(flux_vector_field.meshed_region)# optional
    interface_flux = op.outputs.field()

    normals = get_normals(skin_mesh)

    op = dpf.operators.math.generalized_inner_product() # operator instantiation
    op.inputs.fieldA.connect(interface_flux)
    op.inputs.fieldB.connect(normals)
    normal_interface_flux = op.outputs.field()

    op_integrator = dpf.operators.geo.integrate_over_elements()
    op_integrator.inputs.field.connect(normal_interface_flux)
    op_integrator.inputs.scoping.connect(skin_mesh_scoping)
    op_integrator.inputs.mesh.connect(skin_mesh)
    integrated_normal_interface_flux = op_integrator.outputs.field()
    integrated_normal_interface_flux.meshed_region = skin_mesh

    return integrated_normal_interface_flux

def get_strain_energy_density_damped_harmonic(model, mesh, tfreq):
    # Initialize reusable operators
    op_product = dpf.operators.math.cplx_multiply()
    op_add = dpf.operators.math.add_fc()
    op_scale = dpf.operators.math.scale_fc()

    def scale_field(field, factor):
        """Helper function to scale field using reused operator"""
        op_scale.inputs.fields_container.connect(field)
        op_scale.inputs.ponderation.connect(float(factor))
        return op_scale.outputs.fields_container()

    def multiply_fields(field1, field2):
        """Helper function to multiply two fields using reused operator"""
        op_product.inputs.fields_containerA.connect(field1)
        op_product.inputs.fields_containerB.connect(field2)
        return op_product.outputs.fields_container()

    def add_fields(field1, field2):
        """Helper function to add two fields using reused operator"""
        op_add.inputs.fields_container1.connect(field1)
        op_add.inputs.fields_container2.connect(field2)
        return op_add.outputs.fields_container()

    op_strain = dpf.operators.result.elastic_strain() # operator instantiation
    op_strain.inputs.time_scoping.connect(tfreq)# optional
    op_strain.inputs.data_sources.connect(model)
    op_strain.inputs.mesh.connect(mesh)# optional
    op_strain.inputs.requested_location.connect("Nodal")# optional
    eps = op_strain.outputs.fields_container()

    # Extract strain tensor components
    epsx = eps.select_component(0)
    epsy = eps.select_component(1)
    epsz = eps.select_component(2)
    epsxy = eps.select_component(3)
    epsyz = eps.select_component(4)
    epsxz = eps.select_component(5)

    op_stress = dpf.operators.result.stress() # operator instantiation
    op_stress.inputs.time_scoping.connect(tfreq)# optional
    op_stress.inputs.data_sources.connect(model)
    op_stress.inputs.mesh.connect(mesh)# optional
    op_stress.inputs.requested_location.connect("Nodal")# optional
    sig = op_stress.outputs.fields_container()

    # Extract stress tensor components
    sigx = sig.select_component(0)
    sigy = sig.select_component(1)
    sigz = sig.select_component(2)
    sigxy = sig.select_component(3)
    sigyz = sig.select_component(4)
    sigxz = sig.select_component(5)

    sxex = multiply_fields(sigx, epsx)
    syey = multiply_fields(sigy, epsy)
    szez = multiply_fields(sigz, epsz)
    sxyexy = multiply_fields(sigxy, epsxy) # I THINK MAYBE THESE SHOULD NOT BE MULTIPLIED WITH 2!!!
    sxzexz = multiply_fields(sigxz, epsxz)
    syzeyz = multiply_fields(sigyz, epsyz)

    # sxyexy = scale_field(multiply_fields(sigxy, epsxy),2) # I THINK MAYBE THESE SHOULD NOT BE MULTIPLIED WITH 2!!!
    # sxzexz = scale_field(multiply_fields(sigxz, epsxz),2)
    # syzeyz = scale_field(multiply_fields(sigyz, epsyz),2)

    sxex_syey = add_fields(sxex,syey)
    sxex_syey_szez = add_fields(sxex_syey,szez)
    sxex_syey_szez_sxyexy = add_fields(sxex_syey_szez,sxyexy)
    sxex_syey_szez_sxyexy_sxzexz = add_fields(sxex_syey_szez_sxyexy,sxzexz)
    sxex_syey_szez_sxyexy_sxzexz_syzeyz = add_fields(sxex_syey_szez_sxyexy_sxzexz,syzeyz)

    return scale_field(sxex_syey_szez_sxyexy_sxzexz_syzeyz,1/2)


def perform_complex_tensor_product(T1, T2, scale_factor=1.0):
    """
    Perform tensor product between ij and j tensors of complex values and rescale using scale factor,
    including component extraction and vectorization.

    Parameters:
    T1 (DPF FieldsContainer): e.g. Stress tensor fields container of 6 fields
    T2 (DPF FieldsContainer): Conjugate velocity vector fields container
    scale_factor (float): Scale factor to apply to the final result (default: 1)

    Returns:
    DPF Field: Vectorized and scaled result
    """
    # Initialize reusable operators
    op_product = dpf.operators.math.cplx_multiply()
    op_add = dpf.operators.math.add_fc()
    op_vectorize = dpf.operators.utility.assemble_scalars_to_vectors()

    def multiply_fields(field1, field2):
        """Helper function to multiply two fields using reused operator"""
        op_product.inputs.fields_containerA.connect(field1)
        op_product.inputs.fields_containerB.connect(field2)
        return op_product.outputs.fields_container()

    def add_fields(field1, field2):
        """Helper function to add two fields using reused operator"""
        op_add.inputs.fields_container1.connect(field1)
        op_add.inputs.fields_container2.connect(field2)
        return op_add.outputs.fields_container()

    # Extract stress tensor components
    T1x = T1.select_component(0)
    T1y = T1.select_component(1)
    T1z = T1.select_component(2)
    T1xy = T1.select_component(3)
    T1yz = T1.select_component(4)
    T1xz = T1.select_component(5)

    # Extract velocity vector components
    T2x = T2.select_component(0)
    T2y = T2.select_component(1)
    T2z = T2.select_component(2)

    # Calculate x component: T1x*T2x + T1xy*T2y + T1xz*T2z
    x_comp1 = multiply_fields(T1x, T2x)
    x_comp2 = multiply_fields(T1xy, T2y)
    x_comp3 = multiply_fields(T1xz, T2z)
    x_partial = add_fields(x_comp1, x_comp2)
    x_component = add_fields(x_partial, x_comp3)

    # Calculate y component: T1xy*T2x + T1y*T2y + T1yz*T2z
    y_comp1 = multiply_fields(T1xy, T2x)
    y_comp2 = multiply_fields(T1y, T2y)
    y_comp3 = multiply_fields(T1yz, T2z)
    y_partial = add_fields(y_comp1, y_comp2)
    y_component = add_fields(y_partial, y_comp3)

    # Calculate z component: T1xz*T2x + T1yz*T2y + T1z*T2z
    z_comp1 = multiply_fields(T1xz, T2x)
    z_comp2 = multiply_fields(T1yz, T2y)
    z_comp3 = multiply_fields(T1z, T2z)
    z_partial = add_fields(z_comp1, z_comp2)
    z_component = add_fields(z_partial, z_comp3)

    # Vectorize the real result
    op_vectorize.inputs.x.connect(x_component.get_fields({"complex":0})[0])
    op_vectorize.inputs.y.connect(y_component.get_fields({"complex":0})[0])
    op_vectorize.inputs.z.connect(z_component.get_fields({"complex":0})[0])
    T1T2_real_unscaled = op_vectorize.outputs.field()

    # Scale the results
    op_scale = dpf.operators.math.scale()
    op_scale.inputs.field.connect(T1T2_real_unscaled)
    op_scale.inputs.ponderation.connect(scale_factor)
    T1T2_real = op_scale.outputs.field()

    # Vectorize the real result
    op_vectorize.inputs.x.connect(x_component.get_fields({"complex":1})[0])
    op_vectorize.inputs.y.connect(y_component.get_fields({"complex":1})[0])
    op_vectorize.inputs.z.connect(z_component.get_fields({"complex":1})[0])
    T1T2_imag_unscaled = op_vectorize.outputs.field()

    # Scale the results
    op_scale = dpf.operators.math.scale()
    op_scale.inputs.field.connect(T1T2_imag_unscaled)
    op_scale.inputs.ponderation.connect(scale_factor)
    T1T2_imag = op_scale.outputs.field()

    return T1T2_real, T1T2_imag
def compute_erp(model,skin_mesh,tfreq,skin_mesh_scoping,RHO,C,REFERENCE_POWER):
    op_displacement = dpf.operators.result.displacement()
    op_displacement.inputs.time_scoping.connect(tfreq)
    op_displacement.inputs.mesh_scoping.connect(skin_mesh_scoping)
    op_displacement.inputs.data_sources.connect(model)
    op_displacement.inputs.mesh.connect(skin_mesh)
    displacements_fc = op_displacement.outputs.fields_container()

    # Perform the dpf ERP calculation
    op_erp = dpf.operators.result.equivalent_radiated_power()
    op_erp.inputs.fields_container.connect(displacements_fc)
    op_erp.inputs.mesh.connect(skin_mesh)
    op_erp.inputs.mass_density.connect(RHO)
    op_erp.inputs.speed_of_sound.connect(C)
    op_erp.inputs.erp_type.connect(0)
    op_erp.inputs.factor.connect(REFERENCE_POWER)
    erp_fc = op_erp.outputs.fields_container()
    return erp_fc

# =============================================================================
# # Export Utilities
# =============================================================================
def export_field_to_vtk(mesh,field,path):
    op_export = dpf.operators.serialization.vtk_export() # operator instantiation
    op_export.inputs.file_path.connect(path)
    op_export.inputs.mesh.connect(mesh)
    op_export.inputs.fields1.connect(field)
    op_export.run()

def export_mesh_to_vtk(mesh,path):
    op_export = dpf.operators.serialization.vtk_export() # operator instantiation
    op_export.inputs.file_path.connect(path)
    op_export.inputs.mesh.connect(mesh)
    op_export.run()

def field_to_csv(field, export_path=None, user_header=None):

    if field.location != "Nodal":
        field = field.to_nodal()

    nodal_coordinates = field.meshed_region.nodes.coordinates_field.data
    mapping = field.meshed_region.nodes.mapping_id_to_index

    # Initialize arrays
    n_nodes = field.meshed_region.nodes.n_nodes
    mapped_values = np.zeros(n_nodes)
    # Map thermal conductivity values to correct indices using the mapping
    for node_id, node_index in mapping.items():
        mapped_values[node_index] = field.get_entity_data_by_id(node_id)


    result = {
        'coordinates': nodal_coordinates,
        'values': mapped_values
    }

    # Export to CSV if path is provided
    if export_path:
        # Create header and data structure
        if user_header:
            header = user_header
        data = np.column_stack((nodal_coordinates, mapped_values))

        # Save to CSV using numpy
        np.savetxt(
            export_path,
            data,
            delimiter=',',
            header=','.join(header),
            comments='',  # This prevents the '#' character before the header
            fmt='%.16e'    # Format to 6 decimal places
        )

    return result

def generate_nodal_csv(mesh, field_data, export_path=None, user_header=None):
    nodal_coordinates = mesh.nodes.coordinates_field.data

    result = {
        'coordinates': nodal_coordinates,
        'values': field_data
    }

    # Export to CSV if path is provided
    if export_path:
        # Create header and data structure
        if user_header:
            header = user_header

        data = np.column_stack((nodal_coordinates, field_data))

        # Save to CSV using numpy
        np.savetxt(
            export_path,
            data,
            delimiter=',',
            header=','.join(user_header),
            comments='',  # This prevents the '#' character before the header
            fmt='%.16e'    # Format to 6 decimal places
        )

    return result

