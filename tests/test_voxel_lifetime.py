from radfiled3d import CartesianRadiationField, DType, VoxelGridBuffer, OwningAngularResolvedVoxel, OwningVMFMixtureVoxel, HistogramVoxel, AngularResolvedVoxel, VMFMixtureVoxel
from radfiled3d.glm import vec3, uvec2
from radfiled3d.store import FieldStore, StoreVersion
from radfiled3d.metadata.v1 import Metadata
import numpy as np
import gc
import os
import sys
import pytest
from typing import cast

# Large per-voxel data (~40 KB), so a leaked data buffer per iteration is clearly visible in the RSS
BINS = 10000
SEGMENTS = (100, 100)
LOBES = 2000
ITERATIONS = 2000
# A leak of one data buffer per iteration would grow the RSS by ~80 MB per type
RSS_BOUND = 16 * 1024 * 1024

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RSS is read from /proc/self/statm")


def rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def rss_growth(action, iterations: int = ITERATIONS) -> int:
    for _ in range(50):
        action()
    gc.collect()
    before = rss_bytes()
    for _ in range(iterations):
        action()
    gc.collect()
    return rss_bytes() - before


def create_field() -> CartesianRadiationField:
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.5, 0.5, 0.5))
    field.add_channel("beam")
    ch = cast(VoxelGridBuffer, field.get_channel("beam"))
    ch.add_layer("scalar", "Gy", DType.FLOAT32)
    ch.add_histogram_layer("hist", BINS, 0.5, "counts")
    ch.add_spherical_layer("angular", SEGMENTS[0], SEGMENTS[1], "counts")
    ch.add_vmf_layer("vmf", LOBES, "")
    ch.get_voxel_flat("scalar", 7).set_data(3.5)
    ch.get_voxel_flat("hist", 7).get_histogram()[-1] = 4.5
    ch.get_voxel_flat("angular", 7).get_segments_data()[-1] = 5.5
    ch.get_voxel_flat("vmf", 7).set_lobe(LOBES - 1, 1.0, vec3(0, 0, 1), 6.5)
    return field


@pytest.fixture(scope="module")
def field_file(tmp_path_factory) -> str:
    filename = str(tmp_path_factory.mktemp("lifetime") / "test_voxel_lifetime.rf3")
    FieldStore.store(create_field(), Metadata.default(), filename, StoreVersion.V1)
    return filename


def test_owning_voxels_created_in_python_are_freed():
    def angular():
        vx = OwningAngularResolvedVoxel(uvec2(*SEGMENTS))
        vx.get_segments_data()[-1] = 1.0

    def vmf():
        vx = OwningVMFMixtureVoxel(LOBES)
        vx.set_lobe(LOBES - 1, 1.0, vec3(0, 0, 1), 2.0)

    for action in (angular, vmf):
        assert rss_growth(action) < RSS_BOUND, action.__name__


@pytest.mark.parametrize("layer", ["scalar", "hist", "angular", "vmf"])
def test_owning_voxels_from_accessor_are_freed(field_file, layer):
    accessor = FieldStore.construct_field_accessor(field_file)

    def access():
        vx = accessor.access_voxel_flat(field_file, "beam", layer, 7)
        vx.get_data()

    assert rss_growth(access, 1000) < RSS_BOUND


def test_voxels_of_loaded_fields_are_freed(field_file):
    def load_and_drop():
        field = FieldStore.load(field_file)
        ch = cast(VoxelGridBuffer, field.get_channel("beam"))
        voxels = [ch.get_voxel_flat(layer, 7) for layer in ("scalar", "hist", "angular", "vmf")]
        views = [voxels[1].get_histogram(), voxels[2].get_segments_data(), voxels[3].get_data()]
        del field, ch
        assert views[0][-1] == 4.5

    assert rss_growth(load_and_drop, 200) < RSS_BOUND


def reuse_freed_memory() -> list:
    # Allocations of the sizes of the freed layer buffers (8 voxels) and voxel data, which would overwrite them if they had been freed
    sizes = [8 * BINS, 8 * SEGMENTS[0] * SEGMENTS[1], 8 * 5 * LOBES, BINS, 5 * LOBES]
    return [np.full(size, -1.0, dtype=np.float32) for size in sizes for _ in range(20)]


def test_voxel_data_outlives_its_container(field_file):
    field = FieldStore.load(field_file)
    ch = cast(VoxelGridBuffer, field.get_channel("beam"))
    scalar = ch.get_voxel_flat("scalar", 7)
    hist = cast(HistogramVoxel, ch.get_voxel_flat("hist", 7))
    angular = cast(AngularResolvedVoxel, ch.get_voxel_flat("angular", 7))
    vmf = cast(VMFMixtureVoxel, ch.get_voxel("vmf", 1, 1, 1))
    del field, ch
    gc.collect()
    garbage = reuse_freed_memory()

    assert scalar.get_data() == 3.5
    assert hist.get_bins() == BINS
    assert hist.get_histogram()[-1] == 4.5
    assert angular.get_segments_data()[-1] == 5.5
    assert vmf.get_lobes() == LOBES
    assert vmf.get_kappa(LOBES - 1) == pytest.approx(6.5)
    del garbage


def test_voxel_views_outlive_voxel_and_container(field_file):
    field = FieldStore.load(field_file)
    ch = cast(VoxelGridBuffer, field.get_channel("beam"))
    hist_view = cast(HistogramVoxel, ch.get_voxel_flat("hist", 7)).get_histogram()
    vmf_view = cast(VMFMixtureVoxel, ch.get_voxel_flat("vmf", 7)).get_data()
    del field, ch
    gc.collect()
    garbage = reuse_freed_memory()
    assert hist_view[-1] == 4.5
    assert vmf_view[LOBES - 1, 4] == 6.5
    del garbage

    accessor = FieldStore.construct_field_accessor(field_file)
    grid = accessor.access_layer(field_file, "beam", "vmf")
    grid_voxel = cast(VMFMixtureVoxel, grid.get_voxel(1, 1, 1))
    del grid, accessor
    gc.collect()
    garbage = reuse_freed_memory()
    assert grid_voxel.get_kappa(LOBES - 1) == 6.5
    grid_view = grid_voxel.get_data()
    del grid_voxel
    gc.collect()
    garbage = reuse_freed_memory()
    assert grid_view[LOBES - 1, 4] == 6.5


def test_owning_voxel_views_keep_voxel_alive():
    vx = OwningVMFMixtureVoxel(LOBES)
    vx.set_lobe(3, 0.5, vec3(1, 0, 0), 9.0)
    view = vx.get_data()
    del vx
    gc.collect()
    garbage = reuse_freed_memory()
    assert view[3, 4] == 9.0
    del garbage
