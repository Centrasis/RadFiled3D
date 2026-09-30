from radfiled3d import CartesianRadiationField, RadiationFieldMetadataV1, RadiationFieldMetadataHeaderV1, RadiationFieldSimulationMetadataV1, RadiationFieldXRayTubeMetadataV1, RadiationFieldSoftwareMetadataV1
from radfiled3d.glm import vec2, vec3, vec4, uvec3
import pickle


def test_vec2():
    v1 = vec2(1.0, 2.0)
    assert v1.x == 1.0
    assert v1.y == 2.0
    assert v1 == vec2(1.0, 2.0)
    assert v1 != vec2(2.0, 1.0)
    assert v1 + vec2(1.0, 1.0) == vec2(2.0, 3)
    assert v1 - vec2(1.0, 1.0) == vec2(0.0, 1.0)
    assert v1 * 2.0 == vec2(2.0, 4)
    assert v1 * 2 == vec2(2.0, 4.0)
    assert v1 / 2.0 == vec2(0.5, 1.0)
    assert v1 * 2.0 == vec2(2.0, 4.0)
    assert v1 / 2.0 == 0.5 * v1

def test_vec3():
    v1 = vec3(1.0, 2.0, 3.0)
    assert v1.x == 1.0
    assert v1.y == 2.0
    assert v1.z == 3.0
    assert v1 == vec3(1.0, 2.0, 3.0)
    assert v1 != vec3(2.0, 1.0, 3.0)
    assert v1 + vec3(1.0, 1.0, 1.0) == vec3(2.0, 3, 4)
    assert v1 - vec3(1.0, 1.0, 1.0) == vec3(0.0, 1.0, 2.0)
    assert v1 * 2.0 == vec3(2.0, 4, 6)
    assert v1 / 2.0 == vec3(0.5, 1.0, 1.5)
    assert v1 * 2.0 == vec3(2.0, 4.0, 6.0)
    assert v1 / 2.0 == 0.5 * v1


def test_vec4():
    v1 = vec4(1.0, 2.0, 3.0, 4.0)
    assert v1.x == 1.0
    assert v1.y == 2.0
    assert v1.z == 3.0
    assert v1.w == 4.0
    assert v1 == vec4(1.0, 2.0, 3.0, 4.0)
    assert v1 != vec4(2.0, 1.0, 3.0, 4.0)
    assert v1 + vec4(1.0, 1.0, 1.0, 1.0) == vec4(2.0, 3, 4, 5)
    assert v1 - vec4(1.0, 1.0, 1.0, 1.0) == vec4(0.0, 1.0, 2.0, 3.0)
    assert v1 * 2.0 == vec4(2.0, 4, 6, 8)
    assert v1 / 2.0 == vec4(0.5, 1.0, 1.5, 2.0)
    assert v1 * 2.0 == vec4(2.0, 4.0, 6.0, 8.0)
    assert v1 / 2.0 == 0.5 * v1


def test_uvec3():
    v1 = uvec3(1, 2, 3)
    assert v1.x == 1
    assert v1.y == 2
    assert v1.z == 3
    assert v1 == uvec3(1, 2, 3)
    assert v1 != uvec3(2, 1, 3)
    assert v1 + uvec3(1, 1, 1) == uvec3(2, 3, 4)
    assert v1 - uvec3(1, 1, 1) == uvec3(0, 1, 2)
    assert v1 * 2 == uvec3(2, 4, 6)
    assert v1 / 2 == uvec3(0, 1, 1)
    assert v1 * 2 == uvec3(2, 4, 6)
    assert v1 / 2 == uvec3(0, 1, 1)

def test_radfield():
    field_dim = vec3(10, 10, 10)
    voxel_dim = vec3(1, 1, 1)
    field = CartesianRadiationField(field_dim, voxel_dim)
    field_voxels_count = field.get_voxel_counts()
    assert field_voxels_count.x == 10 and field_voxels_count.y == 10 and field_voxels_count.z == 10

    field_dim = vec3(0.768, 0.2, 0.768)
    voxel_dim = vec3(0.003, 0.004, 0.003)
    field = CartesianRadiationField(field_dim, voxel_dim)
    field_voxels_count = field.get_voxel_counts()
    assert field_voxels_count.x == 256 and field_voxels_count.y == 50 and field_voxels_count.z == 256

    field_dim = vec3(0.768, 0.204, 0.768)
    voxel_dim = vec3(0.003, 0.004, 0.003)
    field = CartesianRadiationField(field_dim, voxel_dim)
    field_voxels_count = field.get_voxel_counts()
    assert field_voxels_count.x == 256 and field_voxels_count.y == 51 and field_voxels_count.z == 256


def test_pickle_support():
    v = vec2(1, 2)
    pickled = pickle.dumps(v)
    v2 = pickle.loads(pickled)
    assert v == v2

    v = vec3(1, 2, 3)
    pickled = pickle.dumps(v)
    v2 = pickle.loads(pickled)
    assert v == v2

    v = vec4(1, 2, 3, 4)
    pickled = pickle.dumps(v)
    v2 = pickle.loads(pickled)
    assert v == v2

    # dropping metadata header support for now
    return
    metadatav1 = RadiationFieldMetadataV1(
        RadiationFieldSimulationMetadataV1(
            100,
            "",
            "Phys",
            RadiationFieldXRayTubeMetadataV1(
                vec3(0, 0, 0),
                vec3(0, 0, 0),
                0,
                "TubeID"
            )
        ),
        RadiationFieldSoftwareMetadataV1(
            "RadFiled3D",
            "0.1.0",
            "repo",
            "commit"
        )
    )
    pickled = pickle.dumps(metadatav1)
    metadatav1_loaded: RadiationFieldMetadataV1 = pickle.loads(pickled)
    loaded_header: RadiationFieldMetadataHeaderV1 = metadatav1_loaded.get_header()
    original_header: RadiationFieldMetadataHeaderV1 = metadatav1.get_header()
    assert loaded_header.simulation.primary_particle_count == original_header.simulation.primary_particle_count
    assert loaded_header.simulation.geometry == original_header.simulation.geometry
    assert loaded_header.software.name == original_header.software.name
    assert loaded_header.software.version == original_header.software.version


def test_float64_voxel():
    """DType.FLOAT64 layers must yield a usable Float64Voxel, not a bare Voxel.

    ScalarVoxel<double> used to be missing from the bindings, so get_voxel() on such a
    layer returned the base class without get_data().
    """
    from radfiled3d import CartesianRadiationField, DType, Float64Voxel, Voxel
    from radfiled3d.glm import vec3

    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.5, 0.5, 0.5))
    channel = field["channel"]
    channel.add_layer("energy", "Gy", DType.FLOAT64)
    assert channel["energy"].dtype == "float64"

    voxel = channel.get_voxel("energy", 0, 0, 0)
    assert isinstance(voxel, Float64Voxel), f"expected Float64Voxel, got {type(voxel).__name__}"
    assert isinstance(voxel, Voxel)

    voxel.set_data(2.5)
    assert voxel.get_data() == 2.5
    # the voxel is a view onto the layer, so the write is visible in the ndarray
    assert channel["energy"][0, 0, 0, 0] == 2.5

    channel["energy"][0, 0, 1] = 4.0
    assert channel.get_voxel("energy", 0, 0, 1).get_data() == 4.0
