from RadFiled3D.RadFiled3D import CartesianRadiationField, vec3, DType, VMFMixtureVoxel, OwningVMFMixtureVoxel, VoxelGridBuffer, FieldJoinMode, FieldJoinCheckMode
from RadFiled3D.utils import FieldStore, StoreVersion
from RadFiled3D.metadata.v1 import Metadata
import numpy as np
import pytest
from typing import cast


def create_field(lobes: int = 3) -> tuple[CartesianRadiationField, VoxelGridBuffer]:
    # 4 x 4 x 4 voxels, so the asymmetric voxel (1, 2, 3) detects transposed axes
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.25, 0.25, 0.25))
    field.add_channel("beam")
    ch = cast(VoxelGridBuffer, field.get_channel("beam"))
    ch.add_vmf_layer("vmf", lobes, "")
    return field, ch


def fill_lobes(vx: VMFMixtureVoxel) -> None:
    vx.set_lobe(0, 0.5, vec3(1, 0, 0), 10.0)
    vx.set_lobe(1, 0.3, [0.0, 0.6, 0.8], 2.5)
    vx.set_lobe(2, 0.2, (0.0, 0.0, -1.0), 0.0)


def test_vmf_voxel_access():
    _, ch = create_field()
    vx = cast(VMFMixtureVoxel, ch.get_voxel("vmf", 1, 2, 3))
    assert isinstance(vx, VMFMixtureVoxel)
    assert vx.get_lobes() == 3
    assert ch.get_layer_voxel_type("vmf") == "vmf_mixture"

    fill_lobes(vx)
    assert vx.get_weight(0) == pytest.approx(0.5)
    mean = vx.get_mean(1)
    assert (mean.x, mean.y, mean.z) == pytest.approx((0.0, 0.6, 0.8))
    assert vx.get_kappa(0) == pytest.approx(10.0)

    data = vx.get_data()
    assert data.shape == (3, 5)
    assert data.dtype == np.float32
    assert data[1] == pytest.approx([0.3, 0.0, 0.6, 0.8, 2.5])
    assert vx.get_lobes_data().shape == (15,)

    with pytest.raises(IndexError):
        vx.get_weight(3)

    vx.clear()
    assert vx.get_lobes_data().sum() == 0.0


def test_vmf_density():
    vx = OwningVMFMixtureVoxel(1)
    vx.set_lobe(0, 1.0, vec3(0, 0, 1), 0.0)
    assert vx.density(vec3(1, 0, 0)) == pytest.approx(1.0 / (4.0 * np.pi))

    kappa = 20.0
    vx.set_lobe(0, 1.0, vec3(0, 0, 1), kappa)
    peak = kappa / (2.0 * np.pi * (1.0 - np.exp(-2.0 * kappa)))
    assert vx.density([0.0, 0.0, 1.0]) == pytest.approx(peak, rel=1e-5)

    # Fibonacci sphere quadrature
    n = 200000
    i = np.arange(n)
    z = 1.0 - (2.0 * i + 1.0) / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    dirs = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    total = sum(vx.density(d) for d in dirs[::10]) * 4.0 * np.pi / (n / 10)
    assert total == pytest.approx(1.0, abs=1e-2)


def test_vmf_merge():
    a = OwningVMFMixtureVoxel(2)
    b = OwningVMFMixtureVoxel(2)
    a.set_lobe(0, 1.0, vec3(1, 0, 0), 5.0)
    b.set_lobe(0, 1.0, vec3(0, 1, 0), 5.0)
    out = OwningVMFMixtureVoxel(2)
    VMFMixtureVoxel.merge(a, 0.25, b, 0.75, out)
    weights = sorted([out.get_weight(0), out.get_weight(1)])
    assert weights == pytest.approx([0.25, 0.75])


def test_vmf_get_layer_as_ndarray():
    _, ch = create_field()
    fill_lobes(cast(VMFMixtureVoxel, ch.get_voxel("vmf", 1, 2, 3)))

    for copy in (False, True):
        array = ch.get_layer_as_ndarray("vmf", copy=copy)
        assert array.shape == (4, 4, 4, 3, 5)  # x, y, z, lobes, [weight, mean_x, mean_y, mean_z, kappa]
        assert array.dtype == np.float32
        assert array[1, 2, 3, 0] == pytest.approx([0.5, 1.0, 0.0, 0.0, 10.0])
        assert array[1, 2, 3, 2] == pytest.approx([0.2, 0.0, 0.0, -1.0, 0.0])
        assert array[:, :, :, :, 0].sum() == pytest.approx(1.0)

    view = ch.get_layer_as_ndarray("vmf")
    view[3, 0, 0, 1, 4] = 7.0
    assert ch.get_voxel("vmf", 3, 0, 0).get_kappa(1) == pytest.approx(7.0)


def test_vmf_add_layer_requires_lobes():
    _, ch = create_field()
    with pytest.raises(Exception):
        ch.add_layer("other", "", DType.VMF_MIXTURE)


def test_vmf_store_load_join(tmp_path):
    filename = str(tmp_path / "test_vmf_py.rf3")
    field, ch = create_field()
    fill_lobes(cast(VMFMixtureVoxel, ch.get_voxel("vmf", 1, 2, 3)))
    FieldStore.store(field, Metadata.default(), filename, StoreVersion.V1)

    loaded = FieldStore.load(filename)
    loaded_vx = cast(VMFMixtureVoxel, cast(VoxelGridBuffer, loaded.get_channel("beam")).get_voxel("vmf", 1, 2, 3))
    assert loaded_vx.get_lobes() == 3
    assert loaded_vx.get_data()[1] == pytest.approx([0.3, 0.0, 0.6, 0.8, 2.5])

    layer = FieldStore.load_single_grid_layer(filename, "beam", "vmf")
    assert layer.get_voxel(1, 2, 3).get_kappa(0) == pytest.approx(10.0)
    assert layer.get_as_ndarray().shape == (4, 4, 4, 3, 5)

    accessor = FieldStore.construct_field_accessor(filename)
    flat = ch.get_voxel_idx(1, 2, 3)
    voxel = accessor.access_voxel_flat(filename, "beam", "vmf", flat)
    assert isinstance(voxel, VMFMixtureVoxel)
    assert voxel.get_weight(1) == pytest.approx(0.3)

    # Joining identical mixtures keeps the lobes
    FieldStore.join(field, Metadata.default(), filename, FieldJoinMode.ADD, FieldJoinCheckMode.NO_CHECKS)
    joined_vx = cast(VMFMixtureVoxel, cast(VoxelGridBuffer, FieldStore.load(filename).get_channel("beam")).get_voxel("vmf", 1, 2, 3))
    weights = [joined_vx.get_weight(k) for k in range(3)]
    assert sum(weights) == pytest.approx(1.0)
    assert sorted(weights) == pytest.approx([0.2, 0.3, 0.5])
    k0 = weights.index(max(weights))
    assert joined_vx.get_kappa(k0) == pytest.approx(10.0, rel=1e-3)


def test_vmf_join_subtract_raises(tmp_path):
    filename = str(tmp_path / "test_vmf_py_subtract.rf3")
    field, _ = create_field()
    FieldStore.store(field, Metadata.default(), filename, StoreVersion.V1)
    with pytest.raises(Exception, match="vMF mixture"):
        FieldStore.join(field, Metadata.default(), filename, FieldJoinMode.SUBTRACT, FieldJoinCheckMode.NO_CHECKS)
