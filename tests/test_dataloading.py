from RadFiled3D.RadFiled3D import CartesianRadiationField, FieldShape, CartesianFieldAccessor, StoreVersion, DType, vec2, vec3, uvec3, RadiationFieldMetadataHeaderV1
from RadFiled3D.utils import FieldStore
from RadFiled3D.metadata.v1 import Metadata
import numpy as np


def setup_test_file(name: str):
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("doserate", "Gy/s", DType.FLOAT32)
    assert field.get_channel("channel1").get_layer_unit("doserate") == "Gy/s"
    assert field.get_field_dimensions() == vec3(1, 1, 1)
    assert field.get_voxel_dimensions() == vec3(0.1, 0.1, 0.1)
    assert field.get_voxel_counts() == uvec3(10, 10, 10)

    array = field.get_channel("channel1").get_layer_as_ndarray("doserate")
    assert array.shape == (10, 10, 10)
    assert array.dtype == "float32"

    array[:, :, :] = 1.0
    array[2:5, 2:5, 2:5] = 2.0

    array = field.get_channel("channel1").get_layer_as_ndarray("doserate")
    assert array[0, 0, 0] == 1.0
    assert array[2, 2, 2] == 2.0
    assert array.min() == 1.0
    assert array.max() == 2.0

    field.get_channel("channel1").get_voxel_by_coord("doserate", 0.99, 0.0, 0.0).set_data(3.0)
    field.get_channel("channel1").get_voxel_by_coord("doserate", 0.0, 0.99, 0.0).set_data(4.0)
    field.get_channel("channel1").get_voxel_by_coord("doserate", 0.0, 0.0, 0.99).set_data(5.0)

    assert array[9, 0, 0] == 3.0
    assert array[0, 9, 0] == 4.0
    assert array[0, 0, 9] == 5.0

    metadata = Metadata.default()
    metadata.simulation.tube.tube_id = "TestTube"
    metadata.simulation.tube.radiation_origin = vec3(0, -1, 0)
    metadata.simulation.tube.radiation_direction = vec3(0, 1, 0)
    metadata.simulation.tube.max_energy_eV = 1500.0
    metadata.simulation.tube.field_shape = FieldShape.ELLIPSIS
    metadata.software.name = "RadFiled3DTest"
    metadata.software.version = "0.0.0"
    metadata.software.repository = "test"
    metadata.software.commit = "commit"
    FieldStore.store(field, metadata, name, StoreVersion.V1)


def test_creation():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    assert field.get_channel("channel1").get_layer_unit("layer1") == "unit1"
    assert field.get_field_dimensions() == vec3(1, 1, 1)
    assert field.get_voxel_dimensions() == vec3(0.1, 0.1, 0.1)
    assert field.get_voxel_counts() == uvec3(10, 10, 10)

    array = field.get_channel("channel1").get_layer_as_ndarray("layer1")
    assert array.shape == (10, 10, 10)
    assert array.dtype == "float32"
    assert array.min() == 0.0
    assert array.max() == 0.0


def test_modification_via_ndarray():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    assert field.get_channel("channel1").get_layer_unit("layer1") == "unit1"
    assert field.get_field_dimensions() == vec3(1, 1, 1)
    assert field.get_voxel_dimensions() == vec3(0.1, 0.1, 0.1)
    assert field.get_voxel_counts() == uvec3(10, 10, 10)

    array = field.get_channel("channel1").get_layer_as_ndarray("layer1")
    assert array.shape == (10, 10, 10)
    assert array.dtype == "float32"

    array[:, :, :] = 1.0

    array[2:5, 2:5, 2:5] = 2.0

    array = field.get_channel("channel1").get_layer_as_ndarray("layer1")
    assert array[0, 0, 0] == 1.0
    assert array[2, 2, 2] == 2.0
    assert array.min() == 1.0
    assert array.max() == 2.0


def test_modification_via_voxels():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    assert field.get_channel("channel1").get_layer_unit("layer1") == "unit1"
    assert field.get_field_dimensions() == vec3(1, 1, 1)
    assert field.get_voxel_dimensions() == vec3(0.1, 0.1, 0.1)
    assert field.get_voxel_counts() == uvec3(10, 10, 10)

    vx = field.get_channel("channel1").get_voxel("layer1", 0, 4, 0)
    assert vx.get_data() == 0.0
    vx.set_data(1.23)
    assert abs(vx.get_data() - 1.23) < 1e-6

    vx = field.get_channel("channel1").get_voxel("layer1", 0, 4, 0)
    assert abs(vx.get_data() - 1.23) < 1e-6

    array = field.get_channel("channel1").get_layer_as_ndarray("layer1")
    assert abs(array[0, 4, 0] - 1.23) < 1e-6
    assert array.min() == 0.0
    assert abs(array.max() - 1.23) < 1e-6


def test_metadata_store_and_peek():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    metadata = Metadata.default()
    metadata.simulation.tube.tube_id = "TubeID"
    metadata.simulation.tube.radiation_origin = vec3(0, 1, 0)
    metadata.simulation.tube.radiation_direction = vec3(0, 0, 0)
    metadata.simulation.tube.max_energy_eV = 0
    metadata.simulation.primary_particle_count = 101
    metadata.software.name = "RadFiled3DTest"
    metadata.software.version = "0.0.0"
    metadata.software.repository = "test"
    metadata.software.commit = "commit"
    FieldStore.store(field, metadata, "test01.rf3", StoreVersion.V1)

    metadata2: RadiationFieldMetadataHeaderV1 = FieldStore.peek_metadata("test01.rf3").get_header()

    assert metadata2.simulation.primary_particle_count == 101
    assert metadata2.software.name == "RadFiled3DTest"
    assert metadata2.software.version == "0.0.0"
    assert metadata2.software.repository == "test"
    assert metadata2.software.commit == "commit"
    assert metadata2.simulation.tube.radiation_origin == vec3(0, 1, 0)
    assert metadata2.simulation.tube.radiation_direction == vec3(0, 0, 0)
    assert metadata2.simulation.tube.max_energy_eV == 0
    assert metadata2.simulation.tube.tube_id == "TubeID"


def test_metadata_store_and_load():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    metadata = Metadata.default()
    metadata.simulation.tube.tube_id = "TubeID"
    metadata.simulation.tube.radiation_origin = vec3(0, 1, 0)
    metadata.simulation.tube.radiation_direction = vec3(0, 0, 0)
    metadata.simulation.tube.max_energy_eV = 1500.0
    metadata.simulation.primary_particle_count = 101
    metadata.software.name = "RadFiled3DTest"
    metadata.software.version = "0.0.0"
    metadata.software.repository = "test"
    metadata.software.commit = "commit"
    metadata.simulation.tube.field_shape = FieldShape.ELLIPSIS
    metadata.simulation.tube.field_ellipsis_opening_angles_deg = vec2(30.0, 20.0)

    spectrum = np.zeros((150, 2), dtype=np.float32)
    spectrum[:, 0] = np.arange(150, dtype=np.float32) * 10.0
    spectrum[:, 1] = 1.0 / 150.0
    metadata.simulation.tube.spectrum = spectrum
    spec2 = metadata.simulation.tube.spectrum
    assert np.isclose(spectrum, spec2).all()

    FieldStore.store(field, metadata, "test02.rf3", StoreVersion.V1)

    metadata2 = FieldStore.load_metadata("test02.rf3")
    metadata2_header = metadata2.get_header()

    assert metadata2_header.simulation.primary_particle_count == 101
    assert metadata2_header.software.name == "RadFiled3DTest"
    assert metadata2_header.software.version == "0.0.0"
    assert metadata2_header.software.repository == "test"
    assert metadata2_header.software.commit == "commit"
    assert metadata2_header.simulation.tube.radiation_origin == vec3(0, 1, 0)
    assert metadata2_header.simulation.tube.radiation_direction == vec3(0, 0, 0)
    assert metadata2_header.simulation.tube.tube_id == "TubeID"

    assert metadata2.simulation.tube.field_shape == FieldShape.ELLIPSIS
    assert metadata2.simulation.tube.field_ellipsis_opening_angles_deg.x == 30.0
    assert metadata2.simulation.tube.field_ellipsis_opening_angles_deg.y == 20.0
    assert metadata2.simulation.tube.max_energy_eV == 1500.0
    assert metadata2.simulation.tube.spectrum.shape == (150, 2)
    assert np.isclose(metadata2.simulation.tube.spectrum[0, 0], 0.0)
    assert np.isclose(metadata2.simulation.tube.spectrum[-1, 0], 1490.0)
    assert np.isclose(metadata2.simulation.tube.spectrum[0, 1], 1.0 / 150.0)
    assert np.isclose(metadata2.simulation.tube.spectrum[-1, 1], 1.0 / 150.0)
    assert np.isclose(np.sum(metadata2.simulation.tube.spectrum[:, 1]), 1.0)


def test_store_and_load():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    assert field.get_channel("channel1").get_layer_unit("layer1") == "unit1"
    assert field.get_field_dimensions() == vec3(1, 1, 1)
    assert field.get_voxel_dimensions() == vec3(0.1, 0.1, 0.1)
    assert field.get_voxel_counts() == uvec3(10, 10, 10)

    array = field.get_channel("channel1").get_layer_as_ndarray("layer1")
    assert array.shape == (10, 10, 10)
    assert array.dtype == "float32"

    array[:, :, :] = 1.0

    array[2:5, 2:5, 2:5] = 2.0

    array = field.get_channel("channel1").get_layer_as_ndarray("layer1")
    assert array[0, 0, 0] == 1.0
    assert array[2, 2, 2] == 2.0
    assert array.min() == 1.0
    assert array.max() == 2.0

    metadata = Metadata.default()
    FieldStore.store(field, metadata, "test02.rf3", StoreVersion.V1)

    field2: CartesianRadiationField = FieldStore.load("test02.rf3")
    assert field2.get_channel("channel1").get_layer_unit("layer1") == "unit1"
    assert field2.get_field_dimensions() == vec3(1, 1, 1)
    assert field2.get_voxel_dimensions() == vec3(0.1, 0.1, 0.1)
    assert field2.get_voxel_counts() == uvec3(10, 10, 10)

    array2 = field2.get_channel("channel1").get_layer_as_ndarray("layer1")
    assert array2.shape == (10, 10, 10)
    assert array2.dtype == "float32"

    arr1 = field.get_channel("channel1").get_layer_as_ndarray("layer1")
    arr2 = field2.get_channel("channel1").get_layer_as_ndarray("layer1")
    assert (arr1 == arr2).all()
    assert (arr1 == array).all()
    assert (arr2.min() == 1.0)
    assert (arr2.max() == 2.0)
    assert (arr2[2, 2, 2] == 2.0)
    assert (arr2[0, 0, 0] == 1.0)


def test_single_channel_loading():
    setup_test_file("test03.rf3")
    accessor: CartesianFieldAccessor = FieldStore.construct_field_accessor("test03.rf3")
    data = open("test03.rf3", "rb").read()
    
    field_from_file = accessor.access_field("test03.rf3")
    field_from_buffer = accessor.access_field_from_buffer(data)
    channels_ff = field_from_file.get_channel_names()
    channels_fb = field_from_buffer.get_channel_names()
    assert len(channels_ff) == len(channels_fb)
    assert "channel1" in channels_ff
    assert "channel1" in channels_fb
    channel = accessor.access_channel("test03.rf3", "channel1")
    assert channel.has_layer("doserate")
    channel_fb = accessor.access_channel_from_buffer(data, "channel1")
    assert channel_fb.has_layer("doserate")

def test_single_layer_loading():
    setup_test_file("test03.rf3")
    accessor: CartesianFieldAccessor = FieldStore.construct_field_accessor("test03.rf3")
    data = open("test03.rf3", "rb").read()
    
    doserate = accessor.access_layer_from_buffer(data, "channel1", "doserate")
    doserate = doserate.get_as_ndarray()
    doserate[:, :, :] = 1.0  # just to check that we can modify it
    doserate[2, 2, 2] = 2.0
    assert doserate.shape == (10, 10, 10)
    assert doserate.dtype == "float32"
    assert doserate[0, 0, 0] == 1.0
    assert doserate[2, 2, 2] == 2.0
    assert doserate.min() == 1.0
    assert doserate.max() == 2.0

    doserate = accessor.access_layer("test03.rf3", "channel1", "doserate")
    doserate = doserate.get_as_ndarray()
    doserate[:, :, :] = 1.0  # just to check that we can modify it
    doserate[2, 2, 2] = 2.0
    assert doserate.shape == (10, 10, 10)
    assert doserate.dtype == "float32"
    assert doserate[0, 0, 0] == 1.0
    assert doserate[2, 2, 2] == 2.0
    assert doserate.min() == 1.0
    assert doserate.max() == 2.0

def test_single_voxel_loading():
    setup_test_file("test03.rf3")
    accessor: CartesianFieldAccessor = FieldStore.construct_field_accessor("test03.rf3")
    data = open("test03.rf3", "rb").read()
    
    vx = accessor.access_voxel_flat("test03.rf3", "channel1", "doserate", 0)
    vx_data = vx.get_data()
    assert vx_data == 1.0
    vx = accessor.access_voxel_flat_from_buffer(data, "channel1", "doserate", 0)
    assert vx.get_data() == 1.0

    vx = accessor.access_voxel_from_buffer(data, "channel1", "doserate", uvec3(0, 0, 0))
    vx_data = vx.get_data()
    assert vx_data == 1.0
    vx = accessor.access_voxel("test03.rf3", "channel1", "doserate", uvec3(0, 0, 0))
    assert vx.get_data() == 1.0

    vx = accessor.access_voxel_by_coord("test03.rf3", "channel1", "doserate", vec3(0.5, 0.5, 0.5))
    assert vx.get_data() == 1.0
    vx = accessor.access_voxel_by_coord_from_buffer(data, "channel1", "doserate", vec3(0.5, 0.5, 0.5))
    assert vx.get_data() == 1.0
