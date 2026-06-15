from RadFiled3D.RadFiled3D import CartesianFieldAccessor, PolarFieldAccessor, uvec2, PolarRadiationField, FieldType, FieldStore, StoreVersion, CartesianRadiationField, DType, vec3, RadiationFieldMetadataV1, RadiationFieldSimulationMetadataV1, RadiationFieldXRayTubeMetadataV1, RadiationFieldSoftwareMetadataV1, VoxelCollectionAccessor, VoxelCollectionRequest, VoxelCollection
import numpy as np
import pickle


METADATA = RadiationFieldMetadataV1(
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


def test_get_store_version_first_call_in_fresh_process(tmp_path):
    import subprocess
    import sys
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    path = str(tmp_path / "version.rf3")
    FieldStore.store(field, METADATA, path, StoreVersion.V1)

    # get_store_version must work as the very first FieldStore call (registry auto-init).
    code = f"from RadFiled3D.RadFiled3D import FieldStore; print(FieldStore.get_store_version({path!r}))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "V1" in result.stdout


def test_construction():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    FieldStore.store(field, METADATA, "test05.rf3", StoreVersion.V1)

    accessor = FieldStore.construct_field_accessor("test05.rf3")
    a_repr = repr(accessor)
    vx_count = accessor.get_voxel_count()
    true_vx_count = field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z
    b_repr = repr(accessor)
    assert a_repr == b_repr
    assert vx_count == true_vx_count
    assert accessor.get_field_type() == FieldType.CARTESIAN


def test_float16_layer_roundtrip(tmp_path):
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.5, 0.5, 0.5))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("half", "unit", DType.FLOAT16)

    view = field.get_channel("channel1").get_layer_as_ndarray("half")
    assert view.dtype == np.float16
    values = np.array([1.5, 2.25, -3.5, 0.125, 100.0, 0.0, 65504.0, -1.0], dtype=np.float16).reshape(view.shape)
    view[...] = values

    voxel = field.get_channel("channel1").get_voxel("half", 0, 0, 0)
    assert voxel.get_data() == 1.5

    path = str(tmp_path / "float16.rf3")
    FieldStore.store(field, METADATA, path, StoreVersion.V1)

    loaded = FieldStore.load(path)
    loaded_view = loaded.get_channel("channel1").get_layer_as_ndarray("half")
    assert loaded_view.dtype == np.float16
    assert np.array_equal(loaded_view, values)

    accessor = FieldStore.construct_field_accessor(path)
    arrays = accessor.access_field_arrays(path, ["channel1"], ["half"], True)
    channel_first = arrays["channel1"]["half"]
    assert channel_first.dtype == np.float16
    assert channel_first.shape == (1, 2, 2, 2)
    assert np.array_equal(channel_first[0], values.reshape(2, 2, 2))


def test_pickle_cartesian():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    FieldStore.store(field, METADATA, "test06.rf3", StoreVersion.V1)

    accessor = FieldStore.construct_field_accessor("test06.rf3")
    pickled_accessor = pickle.dumps(accessor)
    loaded_accessor: CartesianFieldAccessor = pickle.loads(pickled_accessor)

    assert isinstance(loaded_accessor, CartesianFieldAccessor)

    a_repr = repr(accessor)
    b_repr = repr(loaded_accessor)

    assert loaded_accessor.get_voxel_count() == field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z
    assert loaded_accessor.get_field_type() == accessor.get_field_type()
    assert b_repr == a_repr


def test_pickle_polar():
    field = PolarRadiationField(uvec2(10, 10))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)

    FieldStore.store(field, METADATA, "test06_2.rf3", StoreVersion.V1)

    accessor = FieldStore.construct_field_accessor("test06_2.rf3")
    pickled_accessor = pickle.dumps(accessor)
    loaded_accessor: PolarFieldAccessor = pickle.loads(pickled_accessor)

    assert isinstance(loaded_accessor, PolarFieldAccessor)

    a_repr = repr(accessor)
    b_repr = repr(loaded_accessor)

    assert loaded_accessor.get_voxel_count() == field.get_segments_count().x * field.get_segments_count().y
    assert loaded_accessor.get_field_type() == accessor.get_field_type()
    assert b_repr == a_repr


def test_pickle_preserves_channel_layer_access(tmp_path):
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.add_channel("channel2")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    field.get_channel("channel2").add_layer("layer2", "unit2", DType.FLOAT32)
    field.get_channel("channel2").add_histogram_layer("hist", 6, 0.1, "unit3")
    field.get_channel("channel1").get_layer_as_ndarray("layer1")[:, :, :] = 7.5
    field.get_channel("channel2").get_layer_as_ndarray("layer2")[:, :, :] = 9.25

    path = str(tmp_path / "pickle_access.rf3")
    FieldStore.store(field, METADATA, path, StoreVersion.V1)

    accessor = FieldStore.construct_field_accessor(path)
    loaded = pickle.loads(pickle.dumps(accessor))

    assert loaded.access_voxel_flat(path, "channel1", "layer1", 0).get_data() == 7.5
    assert loaded.access_voxel_flat(path, "channel2", "layer2", 0).get_data() == 9.25
    layer = loaded.access_layer(path, "channel2", "layer2")
    assert layer.get_layer().get_unit() == "unit2"
    assert layer.get_as_ndarray().dtype == np.float32


def test_get_as_ndarray_does_not_leak(tmp_path):
    import os
    import gc
    if not os.path.exists("/proc/self/statm"):
        return

    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_histogram_layer("hist", 32, 0.1, "u")
    path = str(tmp_path / "leak.rf3")
    FieldStore.store(field, METADATA, path, StoreVersion.V1)

    accessor = FieldStore.construct_field_accessor(path)
    vx_accessor = VoxelCollectionAccessor(accessor, ["channel1"], ["hist"])
    indices = (np.arange(5000) % 1000).astype(np.int32)
    collection = vx_accessor.access([VoxelCollectionRequest(path, indices)])

    page = os.sysconf("SC_PAGE_SIZE")

    def rss_bytes():
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * page

    for _ in range(50):
        collection.get_as_ndarray("channel1", "hist")
    gc.collect()
    start = rss_bytes()
    for _ in range(2000):
        arr = collection.get_as_ndarray("channel1", "hist")
        del arr
    gc.collect()
    growth = rss_bytes() - start
    assert growth < 100 * 1024 * 1024, f"resident memory grew by {growth / 1e6:.1f} MB across repeated calls"


def test_multi_voxel_distinct_values(tmp_path):
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("flux", "u", DType.FLOAT32)
    VX = field.get_voxel_counts().x
    xs = np.arange(VX)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    field.get_channel("channel1").get_layer_as_ndarray("flux")[..., 0] = (X + Y * VX + Z * VX * VX).astype(np.float32)

    path = str(tmp_path / "distinct.rf3")
    FieldStore.store(field, METADATA, path, StoreVersion.V1)

    accessor = FieldStore.construct_field_accessor(path)
    vx_accessor = VoxelCollectionAccessor(accessor, ["channel1"], ["flux"])
    indices = np.array([5, 100, 999, 0, 42, 0, 5], dtype=np.int32)
    collection = vx_accessor.access([VoxelCollectionRequest(path, indices)])
    values = collection.get_as_ndarray("channel1", "flux").flatten()
    assert np.array_equal(values, indices.astype(np.float32)), f"{values} != {indices}"


def test_access_channel_roundtrip(tmp_path):
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("c")
    field.get_channel("c").add_histogram_layer("hist", 8, 0.1, "u")
    field.get_channel("c").add_layer("flux", "u", DType.FLOAT32)
    rng = np.random.default_rng(3)
    h = rng.random(field.get_channel("c").get_layer_as_ndarray("hist").shape).astype(np.float32)
    f = rng.random(field.get_channel("c").get_layer_as_ndarray("flux").shape).astype(np.float32)
    field.get_channel("c").get_layer_as_ndarray("hist")[:] = h
    field.get_channel("c").get_layer_as_ndarray("flux")[:] = f

    path = str(tmp_path / "channel.rf3")
    FieldStore.store(field, METADATA, path, StoreVersion.V1)
    data = open(path, "rb").read()
    accessor = FieldStore.construct_field_accessor(path)

    for getter in (lambda: accessor.access_channel(path, "c"),
                   lambda: accessor.access_channel_from_buffer(data, "c")):
        ch = getter()
        assert np.array_equal(ch.get_layer_as_ndarray("hist"), h)
        assert np.array_equal(ch.get_layer_as_ndarray("flux"), f)
    full = FieldStore.load(path)
    assert np.array_equal(full.get_channel("c").get_layer_as_ndarray("hist"), h)
    assert np.array_equal(full.get_channel("c").get_layer_as_ndarray("flux"), f)


def test_access_layer_histogram_roundtrip(tmp_path):
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_histogram_layer("hist", 8, 0.1, "u")
    field.get_channel("channel1").add_layer("flux", "u", DType.FLOAT32)
    rng = np.random.default_rng(2)
    h = rng.random(field.get_channel("channel1").get_layer_as_ndarray("hist").shape).astype(np.float32)
    f = rng.random(field.get_channel("channel1").get_layer_as_ndarray("flux").shape).astype(np.float32)
    field.get_channel("channel1").get_layer_as_ndarray("hist")[:] = h
    field.get_channel("channel1").get_layer_as_ndarray("flux")[:] = f

    path = str(tmp_path / "hist_layer.rf3")
    FieldStore.store(field, METADATA, path, StoreVersion.V1)
    data = open(path, "rb").read()
    accessor = FieldStore.construct_field_accessor(path)

    for ln, expected in (("hist", h), ("flux", f)):
        from_file = accessor.access_layer(path, "channel1", ln).get_as_ndarray()
        from_buffer = accessor.access_layer_from_buffer(data, "channel1", ln).get_as_ndarray()
        assert np.array_equal(from_file, expected), f"{ln} file mismatch"
        assert np.array_equal(from_buffer, expected), f"{ln} buffer mismatch"


def test_multichannel_store_load_roundtrip(tmp_path):
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("chA")
    field.add_channel("chB")
    field.get_channel("chA").add_layer("flux", "u", DType.FLOAT32)
    field.get_channel("chB").add_layer("flux", "u", DType.FLOAT32)
    field.get_channel("chB").add_histogram_layer("hist", 5, 0.1, "u")
    rng = np.random.default_rng(1)
    a = rng.random(field.get_channel("chA").get_layer_as_ndarray("flux").shape).astype(np.float32)
    b = rng.random(field.get_channel("chB").get_layer_as_ndarray("flux").shape).astype(np.float32)
    h = rng.random(field.get_channel("chB").get_layer_as_ndarray("hist").shape).astype(np.float32)
    field.get_channel("chA").get_layer_as_ndarray("flux")[:] = a
    field.get_channel("chB").get_layer_as_ndarray("flux")[:] = b
    field.get_channel("chB").get_layer_as_ndarray("hist")[:] = h

    path = str(tmp_path / "multichannel.rf3")
    FieldStore.store(field, METADATA, path, StoreVersion.V1)
    loaded = FieldStore.load(path)

    assert np.array_equal(loaded.get_channel("chA").get_layer_as_ndarray("flux"), a)
    assert np.array_equal(loaded.get_channel("chB").get_layer_as_ndarray("flux"), b)
    assert np.array_equal(loaded.get_channel("chB").get_layer_as_ndarray("hist"), h)


def test_buffer_based_access_paths(tmp_path):
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.add_channel("channel2")
    field.get_channel("channel1").add_layer("shared", "unit1", DType.FLOAT32)
    field.get_channel("channel2").add_layer("shared", "unit1", DType.FLOAT32)
    field.get_channel("channel1").get_layer_as_ndarray("shared")[:, :, :] = 1.5
    field.get_channel("channel2").get_layer_as_ndarray("shared")[:, :, :] = 2.5

    path = str(tmp_path / "buf_access.rf3")
    FieldStore.store(field, METADATA, path, StoreVersion.V1)
    data = open(path, "rb").read()

    accessor = FieldStore.construct_field_accessor(path)

    ch = accessor.access_channel_from_buffer(data, "channel1")
    assert ch.get_voxel_count() == field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z
    layer = accessor.access_layer_from_buffer(data, "channel2", "shared")
    assert layer.get_as_ndarray().flatten()[0] == 2.5
    across = accessor.access_layer_across_channels_from_buffer(data, "shared")
    assert set(across.keys()) == {"channel1", "channel2"}
    assert across["channel1"].get_as_ndarray().flatten()[0] == 1.5
    assert accessor.access_voxel_flat_from_buffer(data, "channel2", "shared", 0).get_data() == 2.5

    import pytest
    with pytest.raises(Exception):
        accessor.access_field_from_buffer(data[:8])


def test_accessing_field():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    FieldStore.store(field, METADATA, "test06.rf3", StoreVersion.V1)

    accessor = FieldStore.construct_field_accessor("test06.rf3")
    field2 = accessor.access_field("test06.rf3")
    assert field2.get_voxel_counts() == field.get_voxel_counts()
    assert field2.get_channel("channel1").get_layer_as_ndarray("layer1").dtype == np.float32

    data = open("test06.rf3", "rb").read()
    field2 = accessor.access_field_from_buffer(data)
    assert field2.get_voxel_counts() == field.get_voxel_counts()
    assert field2.get_channel("channel1").get_layer_as_ndarray("layer1").dtype == np.float32


def test_accessing_layer():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    FieldStore.store(field, METADATA, "test07.rf3", StoreVersion.V1)

    accessor: CartesianFieldAccessor = FieldStore.construct_field_accessor("test07.rf3")
    layer = accessor.access_layer("test07.rf3", "channel1", "layer1")
    assert layer.get_as_ndarray().dtype == np.float32
    assert layer.get_layer().get_unit() == "unit1"

    data = open("test07.rf3", "rb").read()
    layer = accessor.access_layer_from_buffer(data, "channel1", "layer1")
    assert layer.get_as_ndarray().dtype == np.float32
    assert layer.get_layer().get_unit() == "unit1"


def test_accessing_voxel():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    field.get_channel("channel1").get_layer_as_ndarray("layer1")[:, :, :] = 2.34
    FieldStore.store(field, METADATA, "test08.rf3", StoreVersion.V1)

    accessor = FieldStore.construct_field_accessor("test08.rf3")
    voxel = accessor.access_voxel_flat("test08.rf3", "channel1", "layer1", 0)
    assert abs(voxel.get_data() - 2.34) < 0.001

    data = open("test08.rf3", "rb").read()
    voxel = accessor.access_voxel_flat_from_buffer(data, "channel1", "layer1", 0)
    assert abs(voxel.get_data() - 2.34) < 0.001


def test_multi_voxel_accessing():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    field.get_channel("channel1").add_layer("layer2", "unit2", DType.FLOAT32)
    field.get_channel("channel1").add_histogram_layer("histogram1", 6, 0.1, "unit3")
    field.get_channel("channel1").get_layer_as_ndarray("layer1")[:, :, :] = 1.0
    field.get_channel("channel1").get_layer_as_ndarray("layer2")[:, :, :] = 2.0
    hist1_target = field.get_channel("channel1").get_layer_as_ndarray("histogram1")
    for i in range(0, 6):
        hist1_target[:, :, :, i] = i + 1.0

    FieldStore.store(field, METADATA, "test09.rf3", StoreVersion.V1)

    field.get_channel("channel1").get_voxel_flat("layer1", 0).set_data(3.0)
    field.get_channel("channel1").get_voxel_flat("layer1", 1).set_data(4.0)
    field.get_channel("channel1").get_voxel_flat("layer1", 2).set_data(5.0)
    hist2_target = field.get_channel("channel1").get_layer_as_ndarray("histogram1")
    assert (hist2_target - hist1_target).sum() == 0.0, "Histograms should be equal before storing the second field"
    FieldStore.store(field, METADATA, "test10.rf3", StoreVersion.V1)


    accessor = FieldStore.construct_field_accessor("test09.rf3")
    vx_accessor = VoxelCollectionAccessor(accessor, ["channel1"], ["layer1", "layer2", "histogram1"])

    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    req = [
        VoxelCollectionRequest(
            "test09.rf3",
            indices
        ),
        VoxelCollectionRequest(
            "test10.rf3",
            np.array([0, 1, 2], dtype=np.int32)
        )
    ]

    collection: VoxelCollection = vx_accessor.access(req)
    layer1 = collection.get_as_ndarray("channel1", "layer1", copy=False)
    layer2 = collection.get_as_ndarray("channel1", "layer2")
    histogram1 = collection.get_as_ndarray("channel1", "histogram1")

    assert layer1.dtype == np.float32
    assert layer2.dtype == np.float32
    assert layer1.shape[0] == 7
    assert layer2.shape[0] == 7

    assert histogram1.dtype == np.float32
    assert histogram1.shape[0] == 7
    assert histogram1.shape[-1] == hist1_target.shape[-1]

    for i in range(0, 4):
        assert layer1[i] == 1.0
        assert layer2[i] == 2.0

    for i in range(4, 7):
        assert layer1[i] == 3.0 + (i - 4)
        assert layer2[i] == 2.0
