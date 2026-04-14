from RadFiled3D.RadFiled3D import CartesianRadiationField, vec3, DType, SphericalVoxel, OwningSphericalVoxel
from RadFiled3D.utils import FieldStore, StoreVersion
from RadFiled3D.metadata.v1 import Metadata
import numpy as np
import os


def test_spherical_voxel_creation():
    """Test creating a field with a SphericalVoxel layer and accessing it."""
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.5, 0.5, 0.5))
    field.add_channel("beam")
    ch = field.get_channel("beam")
    ch.add_spherical_layer("angular", 12, 6, "")

    sph = ch.get_voxel_flat("angular", 0)
    assert isinstance(sph, SphericalVoxel)
    assert sph.get_phi_segments() == 12
    assert sph.get_theta_segments() == 6
    assert sph.get_total_segments() == 72


def test_spherical_numpy_export_flat():
    """Test that get_segments_data returns a valid flat numpy array."""
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.5, 0.5, 0.5))
    field.add_channel("beam")
    ch = field.get_channel("beam")
    ch.add_spherical_layer("angular", 6, 3, "")

    sph = ch.get_voxel_flat("angular", 0)
    sph.add_value(1.0, 0.5, 5.0)

    data = sph.get_segments_data()
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float32
    assert data.shape[0] == 18
    assert data.sum() == 5.0


def test_spherical_numpy_export_2d():
    """Test that get_data returns a 2D numpy array with correct shape (theta, phi)."""
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.5, 0.5, 0.5))
    field.add_channel("beam")
    ch = field.get_channel("beam")
    ch.add_spherical_layer("angular", 12, 6, "")

    sph = ch.get_voxel_flat("angular", 0)
    sph.add_value(1.0, 0.5, 7.0)

    data = sph.get_data()
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float32
    assert data.shape == (6, 12)  # (theta, phi)
    assert data.sum() == 7.0


def test_spherical_store_and_load():
    """Test that SphericalVoxel data survives store/load roundtrip."""
    filename = "test_spherical_py.rf3"

    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.5, 0.5, 0.5))
    field.add_channel("beam")
    ch = field.get_channel("beam")
    ch.add_spherical_layer("angular", 8, 4, "")
    ch.add_layer("hits", "counts", DType.FLOAT32)

    sph = ch.get_voxel_flat("angular", 0)
    for i in range(32):
        sph.get_segments_data()[i] = float(i)

    metadata = Metadata.default()
    FieldStore.store(field, metadata, filename, StoreVersion.V1)

    loaded = FieldStore.load(filename)
    loaded_ch = loaded.get_channel("beam")
    loaded_sph = loaded_ch.get_voxel_flat("angular", 0)

    assert loaded_sph.get_phi_segments() == 8
    assert loaded_sph.get_theta_segments() == 4

    loaded_data = loaded_sph.get_segments_data()
    for i in range(32):
        assert loaded_data[i] == float(i), f"Mismatch at index {i}: {loaded_data[i]} != {float(i)}"

    os.remove(filename)


def test_spherical_numpy_while_field_alive():
    """Test that numpy arrays are valid while the field is still in scope.
    Note: Like HistogramVoxel, voxel-level numpy views are not safe after
    the parent field goes out of scope. Use get_layer_as_ndarray for that."""
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.5, 0.5, 0.5))
    field.add_channel("beam")
    ch = field.get_channel("beam")
    ch.add_spherical_layer("angular", 6, 3, "")
    sph = ch.get_voxel_flat("angular", 0)
    sph.add_value(1.0, 0.5, 42.0)

    data = sph.get_segments_data()
    assert data.sum() == 42.0
    assert data.shape == (18,)


def test_owning_spherical_voxel():
    """Test OwningSphericalVoxel creation and numpy export."""
    voxel = OwningSphericalVoxel(12, 6)
    assert voxel.get_phi_segments() == 12
    assert voxel.get_theta_segments() == 6
    assert voxel.get_total_segments() == 72

    voxel.add_value(1.0, 0.5, 3.0)
    data = voxel.get_segments_data()
    assert isinstance(data, np.ndarray)
    assert data.sum() == 3.0


def test_spherical_get_layer_as_ndarray():
    """Test that get_layer_as_ndarray returns shape (phi, theta, x, y, z) for SphericalVoxel layers."""
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.5, 0.5, 0.5))
    field.add_channel("beam")
    ch = field.get_channel("beam")
    ch.add_spherical_layer("angular", 12, 6, "")

    sph = ch.get_voxel_flat("angular", 0)
    sph.add_value(1.0, 0.5, 5.0)

    array = ch.get_layer_as_ndarray("angular", copy=False)
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32
    assert array.shape == (12, 6, 2, 2, 2)  # phi, theta, x, y, z
    assert array.sum() == 5.0

    # Verify copy mode works too
    array_copy = ch.get_layer_as_ndarray("angular", copy=True)
    assert array_copy.shape == (12, 6, 2, 2, 2)
    assert array_copy.sum() == 5.0
