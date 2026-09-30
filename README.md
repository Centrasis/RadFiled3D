# RadFiled3D

![Tests](https://github.com/Centrasis/RadFiled3D/actions/workflows/package-test-publish.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/RadFiled3D)](https://pypi.org/project/RadFiled3D/)
[![Python versions](https://img.shields.io/pypi/pyversions/RadFiled3D)](https://pypi.org/project/RadFiled3D/)
[![License](https://img.shields.io/github/license/Centrasis/RadFiled3D)](./LICENSE)

This Repository contains the file format and API according to the Paper: "[RadField3D: A Data Generator and Data Format for Deep Learning in Radiation-Protection Dosimetry for Medical Applications](https://iopscience.iop.org/article/10.1088/1361-6498/add53d)".

The aim of this library is, to provide a simple to use API for a structured, binary file format, that can store all relevant information from a three dimensional radiation field calculated by applications that use algorithms like Monte-Carlo radiation transport simulations. Such a binary file format is useful, when one needs to process a huge amount of radiation field files like when training a neural network. With that use-case in mind, RadFiled3D also provides a python interface with a pyTorch integration. In order to directly iterate a dataset generated with the RadField3D tool, just jump to the section [RadField3D Datasets](#direct-integration-with-radfield3d-datasets).

## 🌟 Why Use RadFiled3D
- **Efficient Storage**: Structured, binary file format for storing large amounts of radiation field data.
- **Easy Integration**: Simple API for C++ and Python with pyTorch support.
- **High Performance**: Optimized for fast data access and manipulation.
- **Versatile**: Supports both Cartesian and Polar coordinate systems.
- **Extensible**: Easily extendable to include additional metadata and data types.

## Table of Contents
- [Building and Installing](#building-and-installing)
- [Getting Started](#getting-started)
  - [From Python](#from-python)
  - [Integrating with pyTorch](#integrating-with-pytorch)
    - [RadField3D Datasets](#direct-integration-with-radfield3d-datasets)
  - [Tracing paths in Cartesian Coordinate Systems](#tracing-paths-in-cartesian-coordinate-systems)
  - [Faster loading of field series](#faster-loading-of-field-series)
  - [From C++](#from-c)
    - [Available Voxel Datatypes](#available-voxel-datatypes)
- [Field Structure](#field-structure)
- [Dependencies](#dependencies)
- [Citation](#citation)

## Building and Installing
### Installing from PyPi
``pip install RadFiled3D``

Prebuilt wheels are published for:

| Platform | Architectures | Python |
| -------- | ------------- | ------ |
| Linux (manylinux 2.28 / 2.17, musllinux 1.2) | x86_64 | 3.11 – 3.14 |
| Windows | x64 | 3.12 – 3.14 |
| macOS 11+ (Apple silicon) | arm64 | 3.11 – 3.14 |

On any other platform — including **Intel macs**, whose wheels were dropped as Apple winds down
x86 support — pip falls back to the source distribution and compiles the module on the fly. That
needs a C++20 compiler (and macOS 10.15 or newer, where `std::filesystem` became available),
while CMake and Ninja are provisioned automatically by the build backend.

### Installing from Source
You can build and install this library and python module from source by using CMake and a C++ compiler. The CMake Project will be 
built automatically, but will take some time.

#### Prerequisites
- C++20 compiler
  - g++ >= 10 or clang for Linux
  - MSVC or clang from Visual Studio 2022 for Windows
  - Apple clang (Xcode command line tools) for macOS
- CMake >= 3.16 (only needed for standalone C++ builds; `pip` provisions its own)
- Python >= 3.11

#### CMake
In order to use the module directly from another C++ Project, you can integrate it by adding the local location of this repository via `add_subdirectory()` and then link against the target `radfiled3d`. All classes are then available from the namespace `radfiled3d`. Check the [Example](./examples/cxx/example01.cpp) or the [First Test File](./tests/basic.cpp) as a first reference.

> **Naming (since 1.4.0).** Namespaces and file names are lowercase — `radfiled3d`,
> `radfiled3d::storage`, `radfiled3d::typing`, and headers such as
> `<radfiled3d/storage/radiation_field_store.hpp>`. Class, struct and enum names keep their
> PascalCase (`CartesianRadiationField`, `DType`). The CMake target is `radfiled3d`.

#### Python
The Python package is built with [scikit-build-core](https://scikit-build-core.readthedocs.io/), the standard PEP 517 backend for CMake projects. It drives the CMake/pybind11 build automatically; no `setup.py` is required. CMake and Ninja are provisioned by the build backend if they are not already present.
##### Installing locally
`python -m pip install .`

##### Building a wheel
`python -m build --wheel`

The package version is taken from the release tag (`GITHUB_REF` / `CI_COMMIT_TAG`) at build time and falls back to `0.0.0` for non-release builds.

## Getting Started
Disclaimer: Not all methods support keyword arguments as they need to be defined manually in the bindings. For some methods like `add_layer` or the Metadata methods those are implemented.

## From Python
Simple example on how to create and store a radiation field. Find more in the example file: [Example](./examples/python/example01.py)
```python
from radfiled3d import CartesianRadiationField, DType
from radfiled3d.glm import vec3
from radfiled3d.store import FieldStore, StoreVersion
from radfiled3d.metadata.v1 import Metadata


# Creating a cartesian radiation field
field = CartesianRadiationField(vec3(2.5, 2.5, 2.5), vec3(0.05, 0.05, 0.05))
# channels are dict-like: indexing returns the channel, creating it if needed.
# A layer needs a unit and a dtype, so it is always created explicitly.
field["channel1"].add_layer("layer1", "unit1", DType.FLOAT32)

# checking what a field or a channel holds
assert "channel1" in field
assert "layer1" in field["channel1"]

# indexing a channel with a layer name gives the voxels as a numpy array
array = field["channel1"]["layer1"]
assert array.shape == (50, 50, 50, 1)

# modify voxels content by using numpy array as no data is copied, just referenced
array[2:5, 2:5, 2:5] = 2.0

# addressing a voxel by providing a point in space
voxel = field["channel1"].get_voxel_by_coord("layer1", 0.1, 2.4, 2.1)

# Store changes to a file
metadata = Metadata.default()
FieldStore.store(field, metadata, "test01.rf3", StoreVersion.V1)

# load data
field2 = FieldStore.load("test01.rf3")
metadata2 = FieldStore.load_metadata("test01.rf3")
```

### Channels and layers as mappings
A field behaves like a dict of channels and a channel like a dict of layers:

| Expression | Meaning |
| ---------- | ------- |
| `field["channel"]` | the channel, **created on first access** if it does not exist |
| `"channel" in field` | whether the channel exists |
| `channel["layer"]` | the layer's voxels as a zero-copy numpy view |
| `"layer" in channel` | whether the layer exists |

Layers are the one asymmetry: they are never created implicitly, because a layer needs a unit
and a data type. Indexing a layer that does not exist raises a `KeyError` telling you to call
`add_layer(name, unit, dtype)` first. The explicit `add_channel` / `get_channel` /
`get_layer_as_ndarray` methods remain available and do exactly the same thing.

### Integrating with pyTorch
RadFiled3D comes with a submodule at `radfiled3d.pytorch`. This module provides some dataset classes to support the usage. Datasets can be loaded from folders or .zip-Files.
```python
import torch
from radfiled3d.pytorch import DataLoaderBuilder
from radfiled3d.pytorch.datasets import MetadataLoadMode
from radfiled3d.pytorch.datasets.cartesian import CartesianFieldSingleLayerDataset
from radfiled3d.pytorch.helpers import RadiationFieldHelper
from radfiled3d.pytorch.types import DirectionalInput, TrainingInputData


# Extend one of the provided dataset classes to match the output to the current needs
class MyLayerDataset(CartesianFieldSingleLayerDataset):
    def __getitem__(self, idx: int) -> TrainingInputData:
        layer, metadata = super().__getitem__(idx)
        tube = metadata.get_header().simulation.tube
        tube_dir, tube_pos = tube.radiation_direction, tube.radiation_origin
        # transform the layers data to a tensor
        return TrainingInputData(
            input=DirectionalInput(
                direction=torch.tensor([tube_dir.x, tube_dir.y, tube_dir.z]),
                origin=torch.tensor([tube_pos.x, tube_pos.y, tube_pos.z]),
                # the tube spectrum lives in the dynamic metadata, so reading it
                # would require MetadataLoadMode.FULL below
                spectrum=None,
            ),
            ground_truth=RadiationFieldHelper.load_tensor_from_layer(layer),
        )


# Optional: provide a finalizer to configure the dataset once the builder created it
def finalize_dataset(dataset: MyLayerDataset) -> None:
    dataset.set_channel_and_layer("test_channel", "test_layer")
    dataset.metadata_load_mode = MetadataLoadMode.HEADER


# Guard the entry point: with more than one worker, python re-imports this module
# in every worker process (the default start method is spawn on Windows/macOS and
# forkserver on Linux as of python 3.14).
if __name__ == "__main__":
    # Pass the dataset class and other options to the DataLoaderBuilder
    builder = DataLoaderBuilder(
        "./test_dataset.zip",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        dataset_class=MyLayerDataset,
        on_dataset_created=finalize_dataset,
    )

    # Build the training dataset
    train_dl = builder.build_train_dataloader(
        batch_size=8,
        shuffle=True,
        worker_count=4,
    )

    # iterate over the dataset; every batch is a TrainingInputData of stacked tensors
    for train_data in train_dl:
        model_input = train_data.input        # DirectionalInput, direction: (8, 3)
        ground_truth = train_data.ground_truth  # (8, c, x, y, z)
```

#### Direct integration with RadField3D datasets
Directly iterate RadField3D datasets either by loading whole fields or iterating each voxel independently. The dataset classes will return pyTorch compatible NamedTuples, that preserve the structure of the raw radiation fields and layers.
```python
from radfiled3d.pytorch import DataLoaderBuilder
from radfiled3d.pytorch.datasets.radfield3d import RadField3DDataset, RadField3DVoxelwiseDataset
# import the pyTorch compatible datatypes
from radfiled3d.pytorch.types import DirectionalInput, PositionalInput, RadiationField, TrainingInputData


if __name__ == "__main__":
    builder = DataLoaderBuilder(
        "./test_dataset_folder/",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        dataset_class=RadField3DDataset
    )

    train_dl = builder.build_train_dataloader(
        batch_size=8,
        shuffle=True,
        worker_count=4
    )

    # iterate over the dataset using fully useable pyTorch classes
    for train_data in train_dl:
        model_input: DirectionalInput | PositionalInput = train_data.input
        field: RadiationField = train_data.ground_truth
```

Swap `dataset_class` for `RadField3DVoxelwiseDataset` to iterate single voxels instead of whole
fields, or for `RadField3DDatasetWithGeometry` to also receive the phantom density map.
**TrainingInputData** consists of two components
**metadata** (as ``DirectionalInput`` or ``PositionalInput``) contains the following information 
- radiation direction (x, y, z)
- radiation origin (x, y, z)
- field shape (Cone, Rectangle, Ellipsis)
- field shape parameters (opening angle, size at origin, ...)
- x-ray tube output spectrum

**field** (as ``RadiationField``) contains the following information
- direct x-ray beam component (as ``RadiationFieldChannel``)
    - spectrum per voxel
    - fluence per voxel
    - statistical error per voxel
- scatter field component (as ``RadiationFieldChannel``)
    - spectrum per voxel
    - fluence per voxel
    - statistical error per voxel
- geometry (binary density map)

### Tracing paths in Cartesian Coordinate Systems
In order to integrate RadFiled3D with other simulation frameworks or applications, one can either take the final results and write it voxel-wise to RadFiled3D or one can already use RadFiled3D during the particle tracking. Therefore, this library offers `GridTracers`. Each of them implements a different line-segment tracing algorithm to find consecutive voxels that are intersected.

The following `GridTracers` exists:
- `SamplingGridTracer`: Traces a line between two points in the grid using a sampling approach.	In this approach the minimum sampling size is the length of the line segment. If the line segment is longer than the minimum sampling size, which is half the L2-Norm of the voxel size, the line is divided into segments of the minimum sampling size. This approach counts the hits if the line segment is incident to a voxel, only!
- `BresenhamGridTracer`: Traces a line between two points in the grid using the [Bresenham](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm) algorithm. This algorithm is a line rasterization algorithm that is used to trace a line between two points in a grid. The starting point is excluded as this can only exit a voxel.
- `LinetracingGridTracer`: This class traces a line between two points in the grid using a combination of the `SamplingGridTracer` and a line tracing algorithm. First the lossy sampling tracer is used to trace the line. Then all adjacent voxels to the voxels that were hit are tested using a line-segment intersection test algorithm.

All those tracers can be created by calling the `GridTracerFactory.construct(..)` method. The tracers share one single interface method:
```python
def trace(self, p1: vec3, p2: vec3) -> list[int]:
```
This method takes two points as the definition of the considered line-segment and returns the flat indices of all voxels intersected, that are inside the grid.

[Example](./examples/python/example02.py) usage:
```python
from radfiled3d import GridTracerFactory, GridTracerAlgorithm, CartesianRadiationField, DType
from radfiled3d.glm import vec3

field = CartesianRadiationField(vec3(1.0, 1.0, 1.0), vec3(0.01, 0.01, 0.01))
field["test"].add_layer("flux", "counts", DType.INT32)

tracer = GridTracerFactory.construct(field, GridTracerAlgorithm.SAMPLING)
indices = tracer.trace(vec3(0.5, 0.5, 0.0), vec3(0.5, 0.85, 1.0))

# The layer array is a zero-copy view of shape (x, y, z, elements_per_voxel) in
# which x varies fastest, exactly like the flat voxel indices the tracer returns.
# Flattening it with order="F" therefore keeps the view, and the increments land
# in the field itself -- .flatten() would copy and the writes would be lost.
hits_counts = field["test"]["flux"]
hits_counts.reshape(-1, order="F")[indices] += 1

# ... and the 3D view of the same data, for plotting or further processing
hits_per_voxel = hits_counts[..., 0]
```

### Faster loading of field series
As the *RadFiled3D* format possesses a dynamic structure, the loading of a radiation field requires the discovery of channels and layers as well as calculating the binary entry points of channels, layers and voxels. When loading datasets for machine learning, the structure of the fields loaded will likely be constant for each dataset. Therefore, the binary entry points can be precalculated to access only those parts of the *RadFiled3D* files that are really needed to increase the loading speed and to reduce the needed memory. This is relealized by the **FieldAccessors** objects.
```python
from radfiled3d import FieldType
from radfiled3d.glm import uvec3
from radfiled3d.store import CartesianFieldAccessor, FieldStore
from radfiled3d.metadata.v1 import Metadata

accessor: CartesianFieldAccessor = FieldStore.construct_field_accessor("a_file.rf3")
field_type = accessor.get_field_type()
assert field_type == FieldType.CARTESIAN

print(accessor)
field = accessor.access_field("a_similar_file.rf3")
layer = accessor.access_layer("a_similar_file.rf3", "channel1", "layer1")
voxel = accessor.access_voxel("a_similar_file.rf3", "channel1", "layer1", uvec3(0, 0, 0))
```
**FieldAccessors** are implemented for the two currently supported coordinate systems: CartesianFieldAccessor and PolarFieldAccessor. Depending on the actual field type, ``FieldStore.construct_field_accessor(AFile)`` returns one of them. The pyTorch Datasets are implemented using the **FieldAccessor** objects to allow for quicker access of datasets. The tests shall act as example code see [test_field_accessor.py](tests/test_field_accessor.py).


## From C++

Simple example on how to create and store a radiation field. Find more in the example file: [Example](./examples/cxx/example01.cpp)
```c++
#include <radfiled3d/storage/radiation_field_store.hpp>
#include <radfiled3d/radiation_field.hpp>
#include <memory>

using namespace radfiled3d;
using namespace radfiled3d::storage;

int main() {
    auto field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f)); // field extents: 2.5 m x 2.5 m x 2.5 m and voxel extents: 5 cm x 5 cm x 5 cm

    auto metadata = std::make_shared<radfiled3d::storage::v1::RadiationFieldMetadata>(
        // learn about the existing data fields from the example file in ./examples/cxx/example01.cpp
    );

    FieldStore::store(field, metadata, "test_field.rf3", StoreVersion::V1);

    auto field2 = FieldStore::load("test_field.rf3");
    return 0;
}
```

### Available Voxel Datatypes
In general, a C++ Scalar- or HistogramVoxel (and thus layers) can hold any datatype. But in order to deserialize them from a file or use them from Python, there is only a specific list implemented. The Available datatypes are:
| C++ Type   | radfiled3d.DType  |
| --------   | ------------  |
| float      | DType.FLOAT32 |
| double     | DType.FLOAT64 |
| int        | DType.INT32   |
| uint8_t    | DType.BYTE  |
| unsigned char    | DType.BYTE  |
| char    | DType.SCHAR  |
| uint32_t   | DType.UINT32  |
| uint64_t   | DType.UINT64  |
| int64_t    | DType.INT64   |
| unsigned long long | DType.UINT64  |
| _Float16   | DType.FLOAT16 |
| glm::vec2     | DType.VEC2 |
| glm::vec3     | DType.VEC3 |
| glm::vec4     | DType.VEC4 |
| HistogramVoxel<float> | DType.HISTOGRAM |
| AngularResolvedVoxel<float> | DType.ANGULAR |
| VMFMixtureVoxel<float> | DType.VMF_MIXTURE |

The three composite types are not created through `add_layer`, because each needs extra
structure: use `add_histogram_layer(name, bins, bin_width, unit)`,
`add_spherical_layer(name, segments, unit)` or `add_vmf_layer(name, lobes, unit)`. A vMF mixture
stores `lobes` x 5 floats per voxel (`weight, mean_x, mean_y, mean_z, kappa`), so a layer has
shape `(x, y, z, lobes, 5)`. It is part of the core format and always available -- unlike
float16, it has no compiler or library prerequisite.

`DType.FLOAT16` requires a compiler that provides `_Float16` (GCC >= 12, a recent clang). Builds
without it compile the type out and raise a clear error when it is used, so check
`radfiled3d.HAS_FLOAT16` before relying on it — it is not available in every published
wheel (for 1.3.6 the manylinux_2_28 and musllinux wheels have it, the manylinux2014 and Windows
ones do not).


## Field Structure
RadFiled3D defines a field structure, that provides the user with the possibility to first define in which kind of space he wants to operate. Therefore one can choose between `CartesianRadiationField` and `PolarRadiationField`.
- *CartesianRadiationField*: Segments a room defined by an extent of the room itself and each cuboid voxel into a set of voxels. Each voxel can be addressed by a 3D position (coordinate: x, y, z), a 3D index (number of the voxel in each dimension) or a flat 1D index.
- *PolarRadiationField*: Segements the surface of a unit sphere into surface segments. Each segment (voxel) can be addressed by a 2D position (coordinate: theta, phi), a 2D index (number of the segment in each dimension) or a flat 1D index.

Fields are then partitioned into channels (`VoxelGridBuffer`/`PolarSegmentsBuffer`). All channels share the same size and resolution. A channel is again partitioned into layers (`VoxelGrid`/`PolarSegment`). Each layer holds the actual voxel data and can be constructed from various data types (float, double, uint32_t, uint64_t, glm::vec2, glm::vec3, glm::vec4, N-D-Histogram (list of floats), angular-resolved segment grids and von Mises-Fisher mixtures). Additionally, a layer has a unit string assigned to it as well as a statistical uncertainty to perserve those information.

## Dependencies
RadFiled3D comes with a possibly low amount of dependencies. We integrated the OpenGL Math Library (GLM) just to provide those datatypes out of the box and as GLM is a head-only library we suspect no issues by doing so.

All C++ dependencies (Will be fetched by CMake):
- [GLM](https://github.com/g-truc/glm)

All python dependencies:
- [PyBind11](https://github.com/pybind/pybind11)
- [rich](https://github.com/Textualize/rich)
- [numpy](https://numpy.org/)
- Optional:
  - [pyTorch](https://pytorch.org/)

## Citation
If you use RadFiled3D in academic work, please cite the paper describing the format and the data generator:

> Lehner, F., Lombardo, P., Castillo, S., Hupe, O. and Magnor, M. (2025).
> *RadField3D: a data generator and data format for deep learning in radiation-protection dosimetry for medical applications.*
> Journal of Radiological Protection **45**(2), 021508. https://doi.org/10.1088/1361-6498/add53d

```bibtex
@article{Lehner2025RadField3D,
  author  = {Lehner, Felix and Lombardo, Pasquale and Castillo, Susana and Hupe, Oliver and Magnor, Marcus},
  title   = {RadField3D: a data generator and data format for deep learning in radiation-protection dosimetry for medical applications},
  journal = {Journal of Radiological Protection},
  volume  = {45},
  number  = {2},
  pages   = {021508},
  year    = {2025},
  doi     = {10.1088/1361-6498/add53d}
}
```
