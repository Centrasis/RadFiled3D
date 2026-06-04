try:
    import torch
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False


def test_radfield3d_voxelwise_dataset():
    if TORCH_INSTALLED:
        from RadFiled3D.RadFiled3D import CartesianRadiationField, vec3, DType, FieldShape
        from RadFiled3D.utils import FieldStore, StoreVersion
        from RadFiled3D.metadata.v1 import Metadata
        from RadFiled3D.pytorch.datasets.radfield3d import RadField3DVoxelwiseDataset, TrainingInputData
        import os
        import random
        import numpy as np

        spectrum = np.zeros((150, 2), dtype=np.float32)
        spectrum[:, 0] = np.arange(150, dtype=np.float32) * 10.0
        spectrum[:, 1] = 1.0 / 150.0

        METADATA = Metadata.default()
        METADATA.simulation.tube.max_energy_eV = 1500.0
        METADATA.simulation.tube.spectrum = spectrum
        METADATA.simulation.tube.field_shape = FieldShape.CONE
        METADATA.simulation.tube.opening_angle_deg = 30.0

        field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
        field.add_channel("scatter_field")
        field.add_channel("direct_beam")
        field.get_channel("scatter_field").add_layer("flux", "unit1", DType.FLOAT32)
        field.get_channel("scatter_field").add_layer("error", "unit1", DType.FLOAT32)
        field.get_channel("scatter_field").add_histogram_layer("spectrum", 32, 0.1, "unit1")
        field.get_channel("direct_beam").add_layer("flux", "unit1", DType.FLOAT32)
        field.get_channel("direct_beam").add_layer("error", "unit1", DType.FLOAT32)
        field.get_channel("direct_beam").add_histogram_layer("spectrum", 32, 0.1, "unit1")

        os.makedirs("test_dataset", exist_ok=True)

        FieldStore.store(field, METADATA, "test_dataset/test01.rf3", StoreVersion.V1)
        FieldStore.store(field, METADATA, "test_dataset/test02.rf3", StoreVersion.V1)
        FieldStore.store(field, METADATA, "test_dataset/test03.rf3", StoreVersion.V1)

        dataset = RadField3DVoxelwiseDataset(file_paths=["test_dataset/test01.rf3", "test_dataset/test02.rf3", "test_dataset/test03.rf3"])
        ds_len = 3 * field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z
        assert len(dataset) == ds_len, f"Dataset length does not match expected voxel count: {len(dataset)} != {ds_len}"

        test_in: TrainingInputData = dataset.__getitems__([random.randint(0, len(dataset)) for _ in range(100)])
        assert test_in.ground_truth.scatter_field.error.shape[0] == 100, "Ground truth error shape does not match expected batch size."
        assert test_in.ground_truth.scatter_field.flux.shape[0] == 100, "Ground truth fluence shape does not match expected batch size."
        assert test_in.ground_truth.scatter_field.spectrum.shape[0] == 100, "Ground truth spectrum shape does not match expected batch size."
        assert test_in.ground_truth.direct_beam.error.shape[0] == 100, "X-ray beam error shape does not match expected batch size."
        assert test_in.ground_truth.direct_beam.flux.shape[0] == 100, "X-ray beam fluence shape does not match expected batch size."
        assert test_in.ground_truth.direct_beam.spectrum.shape[0] == 100, "X-ray beam spectrum shape does not match expected batch size."
        assert test_in.input.direction.shape[0] == 100, "Input direction shape does not match expected batch size."
        assert test_in.input.position.shape[0] == 100, "Input position shape does not match expected batch size."
        assert test_in.input.spectrum.shape[0] == 100, "Input tube spectrum shape does not match expected batch size."


def _make_simple_field():
    from RadFiled3D.RadFiled3D import CartesianRadiationField, vec3, DType
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("doserate", "Gy/s", DType.FLOAT32)
    return field


def test_file_paths_is_plain_list(tmp_path):
    """file_paths must be a plain in-memory list."""
    if not TORCH_INSTALLED:
        return
    from RadFiled3D.utils import FieldStore, StoreVersion
    from RadFiled3D.metadata.v1 import Metadata
    from RadFiled3D.pytorch.datasets.cartesian import CartesianFieldDataset

    files = []
    for i in range(3):
        p = tmp_path / f"f{i}.rf3"
        FieldStore.store(_make_simple_field(), Metadata.default(), str(p), StoreVersion.V1)
        files.append(str(p))

    ds = CartesianFieldDataset(file_paths=files)
    assert type(ds.file_paths) is list, f"file_paths is {type(ds.file_paths)}, expected plain list"
    assert "multiprocessing" not in type(ds.file_paths).__module__
    assert len(ds) == 3
    assert ds.file_paths[0] == files[0]


def test_zip_file_paths_populated_from_archive(tmp_path):
    """When only a zip_file is given (file_paths=None), the file list derived from
    the archive must survive construction and not be reset to None."""
    if not TORCH_INSTALLED:
        return
    import zipfile
    from RadFiled3D.utils import FieldStore, StoreVersion
    from RadFiled3D.metadata.v1 import Metadata
    from RadFiled3D.pytorch.datasets.cartesian import CartesianFieldDataset

    names = []
    for i in range(2):
        p = tmp_path / f"z{i}.rf3"
        FieldStore.store(_make_simple_field(), Metadata.default(), str(p), StoreVersion.V1)
        names.append(p)
    zip_path = tmp_path / "ds.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in names:
            zf.write(p, arcname=p.name)

    ds = CartesianFieldDataset(file_paths=None, zip_file=str(zip_path))
    assert ds.file_paths is not None, "zip-derived file_paths must be populated"
    assert type(ds.file_paths) is list
    assert len(ds) == 2


def _make_voxelwise_dataset(tmp_path, n_files=4, tube_bins=32, hist_bins=32):
    """Build a small RadField3DVoxelwiseDataset whose scatter_field.flux encodes
    (file, linear voxel index) and whose radiation_direction encodes the file
    index, so batch alignment can be checked unambiguously. tube_bins and hist_bins
    can differ to exercise the prefetch cache sizing."""
    from RadFiled3D.RadFiled3D import CartesianRadiationField, vec3, DType, FieldShape
    from RadFiled3D.utils import FieldStore, StoreVersion
    from RadFiled3D.metadata.v1 import Metadata
    from RadFiled3D.pytorch.datasets.radfield3d import RadField3DVoxelwiseDataset
    import numpy as np

    spectrum = np.zeros((tube_bins, 2), dtype=np.float32)
    spectrum[:, 0] = np.arange(tube_bins, dtype=np.float32) * 10.0
    spectrum[:, 1] = 1.0 / tube_bins

    VX = 10
    xs = np.arange(VX)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    linidx = (X + Y * VX + Z * VX * VX).astype(np.float32)

    files = []
    for f in range(n_files):
        md = Metadata.default()
        md.simulation.tube.max_energy_eV = 1500.0
        md.simulation.tube.spectrum = spectrum
        md.simulation.tube.field_shape = FieldShape.CONE
        md.simulation.tube.opening_angle_deg = 30.0
        md.simulation.tube.radiation_direction = vec3(float(f), 0.0, 0.0)
        md.simulation.tube.radiation_origin = vec3(float(f) * 0.1, 0.2, 0.3)

        field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
        field.add_channel("scatter_field")
        field.add_channel("direct_beam")
        for ch_name in ("scatter_field", "direct_beam"):
            ch = field.get_channel(ch_name)
            ch.add_layer("flux", "unit1", DType.FLOAT32)
            ch.add_layer("error", "unit1", DType.FLOAT32)
            ch.add_histogram_layer("spectrum", hist_bins, 0.1, "unit1")
        flux = field.get_channel("scatter_field").get_layer_as_ndarray("flux", copy=False)
        flux[..., 0] = f * 1_000_000.0 + linidx

        p = tmp_path / f"vox{f}.rf3"
        FieldStore.store(field, md, str(p), StoreVersion.V1)
        files.append(str(p))

    return RadField3DVoxelwiseDataset(file_paths=files)


def test_getitems_without_prefetch(tmp_path):
    """The non-prefetched __getitems__ branch must work when no cache is set up."""
    if not TORCH_INSTALLED:
        return
    from RadFiled3D.pytorch.datasets.radfield3d import TrainingInputData
    ds = _make_voxelwise_dataset(tmp_path, n_files=4)
    assert ds.cached_fields is None and ds.cached_metadata is None
    vpf = ds.voxels_per_field
    indices = [2 * vpf + 7, 0 * vpf + 3, 1 * vpf + 5, 3 * vpf + 9, 1 * vpf + 2]
    out = ds.__getitems__(indices)
    assert isinstance(out, TrainingInputData)
    assert out.input.position.shape[0] == len(indices)
    assert out.ground_truth.scatter_field.flux.shape[0] == len(indices)


def test_getitems_alignment(tmp_path):
    """In a batched __getitems__ spanning multiple files, every per-sample field
    (position, direction, ground truth) must correspond to the same requested index.
    Data is encoded so flux == file*1e6 + voxel_index and direction[0] == file."""
    if not TORCH_INSTALLED:
        return
    import torch
    ds = _make_voxelwise_dataset(tmp_path, n_files=4)
    vpf = ds.voxels_per_field
    indices = [2 * vpf + 7, 0 * vpf + 3, 1 * vpf + 5, 3 * vpf + 9,
               1 * vpf + 2, 0 * vpf + 100, 3 * vpf + 50, 2 * vpf + 1]
    batch = ds.__getitems__(indices)

    for k, i in enumerate(indices):
        ref = ds.__getitem__(i)
        assert torch.allclose(batch.input.position[k], ref.input.position), f"position mismatch at row {k}"
        assert torch.allclose(batch.input.direction[k], ref.input.direction), f"direction mismatch at row {k}"
        assert torch.allclose(batch.ground_truth.scatter_field.flux[k],
                              ref.ground_truth.scatter_field.flux), f"flux mismatch at row {k}"

    for k, i in enumerate(indices):
        f, v = i // vpf, i % vpf
        flux_val = round(float(batch.ground_truth.scatter_field.flux[k].flatten()[0]))
        assert flux_val == f * 1_000_000 + v, f"row {k}: ground truth flux {flux_val} != encoded {(f, v)}"
        assert round(float(batch.input.direction[k][0])) == f, f"row {k}: direction encodes file {f}"


def test_prefetch_roundtrip(tmp_path):
    """prefetch_data must not crash when tube spectrum bins differ from histogram
    bins, and the cached path must return the same origin / beam-shape / spectrum /
    ground truth as the non-cached single __getitem__."""
    if not TORCH_INSTALLED:
        return
    import torch
    from RadFiled3D.pytorch.datasets.radfield3d import RadField3DVoxelwiseDataset

    ds = _make_voxelwise_dataset(tmp_path, n_files=4, tube_bins=64, hist_bins=32)
    files = list(ds.file_paths)
    vpf = ds.voxels_per_field
    indices = [0, vpf + 5, 2 * vpf + 9, 3 * vpf + 123]

    ref = [ds.__getitem__(i) for i in indices]

    ds_cache = RadField3DVoxelwiseDataset(file_paths=files)
    ds_cache.prefetch_data()
    assert ds_cache.cached_metadata is not None and ds_cache.cached_fields is not None

    for k, i in enumerate(indices):
        cached = ds_cache.__getitem__(i)
        assert torch.allclose(cached.input.origin, ref[k].input.origin), f"origin mismatch row {k}"
        assert torch.allclose(cached.input.direction, ref[k].input.direction), f"direction mismatch row {k}"
        assert torch.allclose(cached.input.spectrum, ref[k].input.spectrum), f"tube spectrum mismatch row {k}"
        assert torch.allclose(cached.input.beam_shape_parameters, ref[k].input.beam_shape_parameters), f"beam params row {k}"
        assert torch.allclose(cached.input.beam_shape_type, ref[k].input.beam_shape_type), f"beam type row {k}"
        assert torch.allclose(cached.ground_truth.scatter_field.spectrum,
                              ref[k].ground_truth.scatter_field.spectrum), f"gt spectrum mismatch row {k}"


def test_pickle_drops_accessor(tmp_path):
    """The field accessor is kept in-process but dropped on pickling, and the
    dataset still works after unpickling."""
    if not TORCH_INSTALLED:
        return
    import pickle
    from RadFiled3D.utils import FieldStore, StoreVersion
    from RadFiled3D.metadata.v1 import Metadata
    from RadFiled3D.pytorch.datasets.cartesian import CartesianSingleVoxelDataset

    files = []
    for i in range(3):
        p = tmp_path / f"sv{i}.rf3"
        FieldStore.store(_make_simple_field(), Metadata.default(), str(p), StoreVersion.V1)
        files.append(str(p))

    ds = CartesianSingleVoxelDataset(file_paths=files)
    ds.set_channel_and_layer("channel1", "doserate")
    n = len(ds)
    assert ds._field_accessor is not None, "accessor must be available after __len__"
    assert ds._voxel_count is not None, "voxel count must be cached"

    ds2 = pickle.loads(pickle.dumps(ds))
    assert ds2._field_accessor is None, "accessor must be dropped on pickling"
    assert len(ds2) == n, "dataset must work (and rebuild the accessor) after unpickling"


def test_zip_buffer_cache(tmp_path):
    """Repeated reads of the same zipped member reuse a cached buffer instead of
    re-reading the whole file each time."""
    if not TORCH_INSTALLED:
        return
    import zipfile
    from RadFiled3D.utils import FieldStore, StoreVersion
    from RadFiled3D.metadata.v1 import Metadata
    from RadFiled3D.pytorch.datasets.cartesian import CartesianFieldDataset

    names = []
    for i in range(2):
        p = tmp_path / f"zc{i}.rf3"
        FieldStore.store(_make_simple_field(), Metadata.default(), str(p), StoreVersion.V1)
        names.append(p)
    zip_path = tmp_path / "cache.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in names:
            zf.write(p, arcname=p.name)

    ds = CartesianFieldDataset(file_paths=[p.name for p in names], zip_file=str(zip_path))
    assert ds._buffer_cache is None
    b0 = ds.load_file_buffer(0)
    assert ds._buffer_cache is not None and ds._buffer_cache[0] == ds.file_paths[0]
    b0_again = ds.load_file_buffer(0)
    assert b0_again is b0, "same member should return the cached buffer object"
    ds.load_file_buffer(1)
    assert ds._buffer_cache[0] == ds.file_paths[1], "cache should switch to the newly read member"


def _make_radfield3d_files(tmp_path, n_files=1, with_geometry=False):
    from RadFiled3D.RadFiled3D import CartesianRadiationField, vec3, DType, FieldShape
    from RadFiled3D.utils import FieldStore, StoreVersion
    from RadFiled3D.metadata.v1 import Metadata
    import numpy as np

    spectrum = np.zeros((150, 2), dtype=np.float32)
    spectrum[:, 0] = np.arange(150, dtype=np.float32) * 10.0
    spectrum[:, 1] = 1.0 / 150.0
    rng = np.random.default_rng(7)
    files = []
    for fi in range(n_files):
        md = Metadata.default()
        md.simulation.tube.max_energy_eV = 1500.0
        md.simulation.tube.spectrum = spectrum
        md.simulation.tube.field_shape = FieldShape.CONE
        md.simulation.tube.opening_angle_deg = 30.0
        md.simulation.tube.radiation_origin = vec3(0.1 * fi, 0.2, 0.3)
        field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
        for ch in ("scatter_field", "direct_beam"):
            field.add_channel(ch)
            c = field.get_channel(ch)
            c.add_layer("flux", "u", DType.FLOAT32)
            c.add_layer("error", "u", DType.FLOAT32)
            c.add_histogram_layer("spectrum", 32, 0.1, "u")
            for ly in ("flux", "error", "spectrum"):
                c.get_layer_as_ndarray(ly)[:] = rng.random(c.get_layer_as_ndarray(ly).shape).astype(np.float32)
        if with_geometry:
            field.add_channel("geometry")
            field.get_channel("geometry").add_layer("density", "u", DType.FLOAT32)
            field.get_channel("geometry").get_layer_as_ndarray("density")[:] = rng.random(
                field.get_channel("geometry").get_layer_as_ndarray("density").shape).astype(np.float32)
        p = tmp_path / f"rf3_{fi}.rf3"
        FieldStore.store(field, md, str(p), StoreVersion.V1)
        files.append(str(p))
    return files


def test_radfield3d_dataset_access_field_arrays_matches_legacy(tmp_path):
    if not TORCH_INSTALLED:
        return
    import torch
    from RadFiled3D.pytorch.datasets.radfield3d import RadField3DDataset

    files = _make_radfield3d_files(tmp_path, n_files=1)
    ds = RadField3DDataset(file_paths=files)
    new = ds.__getitem__(0)
    legacy = ds.transform2training_input(ds._get_field(0), ds._get_metadata(0))

    for ch in ("scatter_field", "direct_beam"):
        for ly in ("spectrum", "flux", "error"):
            a = getattr(getattr(new.ground_truth, ch), ly)
            b = getattr(getattr(legacy.ground_truth, ch), ly)
            assert a.shape == b.shape, f"{ch}/{ly} shape {a.shape} != {b.shape}"
            assert torch.equal(a, b), f"{ch}/{ly} value mismatch"
    assert torch.equal(new.input.direction, legacy.input.direction)
    assert torch.equal(new.input.origin, legacy.input.origin)
    assert torch.equal(new.input.spectrum, legacy.input.spectrum)
    assert new.ground_truth.scatter_field.spectrum.shape[0] == 32
    assert new.ground_truth.scatter_field.flux.shape[0] == 1


def test_radfield3d_dataset_with_geometry(tmp_path):
    if not TORCH_INSTALLED:
        return
    import torch
    from RadFiled3D.pytorch.datasets.radfield3d import RadField3DDatasetWithGeometry

    files = _make_radfield3d_files(tmp_path, n_files=1, with_geometry=True)
    ds = RadField3DDatasetWithGeometry(file_paths=files)
    out = ds.__getitem__(0)
    assert out.input.geometry is not None
    assert out.input.geometry.shape[0] == 1
    legacy = ds._get_field(0).get_channel("geometry").get_layer_as_ndarray("density")
    assert torch.equal(out.input.geometry, torch.from_numpy(legacy.astype("float32", copy=False)).permute(-1, 0, 1, 2))


def test_radfield3d_dataset():
    if TORCH_INSTALLED:
        from RadFiled3D.RadFiled3D import CartesianRadiationField, vec3, DType, FieldShape
        from RadFiled3D.utils import FieldStore, StoreVersion
        from RadFiled3D.metadata.v1 import Metadata
        from RadFiled3D.pytorch.datasets.radfield3d import RadField3DDataset, TrainingInputData
        from RadFiled3D.pytorch.radiationfieldloader import DataLoaderBuilder
        import os
        import numpy as np

        spectrum = np.zeros((150, 2), dtype=np.float32)
        spectrum[:, 0] = np.arange(150, dtype=np.float32) * 10.0
        spectrum[:, 1] = 1.0 / 150.0

        METADATA = Metadata.default()
        METADATA.simulation.tube.max_energy_eV = 1500.0
        METADATA.simulation.tube.spectrum = spectrum
        METADATA.simulation.tube.field_shape = FieldShape.CONE
        METADATA.simulation.tube.opening_angle_deg = 30.0

        field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
        field.add_channel("scatter_field")
        field.add_channel("direct_beam")
        ch = field.get_channel("scatter_field")
        ch_xray = field.get_channel("direct_beam")
        ch.add_layer("flux", "unit1", DType.FLOAT32)
        ch.add_layer("error", "unit1", DType.FLOAT32)
        ch.add_histogram_layer("spectrum", 32, 0.1, "unit1")
        ch_xray.add_layer("flux", "unit1", DType.FLOAT32)
        ch_xray.add_layer("error", "unit1", DType.FLOAT32)
        ch_xray.add_histogram_layer("spectrum", 32, 0.1, "unit1")

        spectrum = ch.get_layer_as_ndarray("spectrum", copy=True)
        assert not np.isnan(spectrum).any(), "Spectrum contains NaN values."
        assert not np.isinf(spectrum).any(), "Spectrum contains Inf values."
        spectrum = np.random.rand(*spectrum.shape)
        assert not np.isnan(spectrum).any(), "Spectrum contains NaN values."
        assert not np.isinf(spectrum).any(), "Spectrum contains Inf values."
        spectrum_sums = spectrum.sum(axis=-1, keepdims=True)
        spectrum /= spectrum_sums

        spectrum1 = spectrum.copy()

        first_hist1 = spectrum[0, 0, 0, :].copy()

        spectrum_empty = ch.get_layer_as_ndarray("spectrum", copy=True)
        assert np.allclose(spectrum_empty, 0.0), "Spectrum layer is not empty as expected."
        ch.get_layer_as_ndarray("spectrum", copy=False)[:] = spectrum

        spectrum = ch.get_layer_as_ndarray("spectrum", copy=True)
        first_hist2 = spectrum[0, 0, 0, :].copy()
        assert np.allclose(first_hist1, first_hist2), "Spectrum values changed unexpectedly between accesses."
        assert np.allclose(spectrum1, spectrum), "Spectrum values changed unexpectedly."
        spectrum_sums = spectrum.sum(axis=-1, keepdims=True)
        assert np.allclose(spectrum_sums[spectrum_sums != 0], 1.0), "Spectrum normalization failed."

        os.makedirs("test_dataset", exist_ok=True)

        for i in range(10):
            FieldStore.store(field, METADATA, f"test_dataset/test0{i}.rf3", StoreVersion.V1)

        dataset = RadField3DDataset(file_paths=[f"test_dataset/test0{i}.rf3" for i in range(10)])
        ds_len = 10
        assert len(dataset) == ds_len, f"Dataset length does not match expected voxel count: {len(dataset)} != {ds_len}"

        builder = DataLoaderBuilder(
            dataset_class=RadField3DDataset,
            dataset_path="test_dataset",
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0
        )
        ds = builder.build_train_dataset()
        assert len(ds) == ds_len, f"Dataset length does not match expected voxel count: {len(ds)} != {ds_len}"

        elem = ds.__getitem__(0)
        assert isinstance(elem, TrainingInputData), "Dataset element is not of type TrainingInputData."
        assert len(elem.input.beam_shape_parameters.shape) == 1, "Beam shape parameters length does not match expected value."
        assert elem.input.beam_shape_parameters[0] == 30.0, "Beam shape parameter does not match expected value."
