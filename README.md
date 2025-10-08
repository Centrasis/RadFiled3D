# RadFiled3D

![Tests](https://github.com/Centrasis/RadFiled3D/actions/workflows/package-test-publish.yml/badge.svg)

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

## Building and Installing
### Installing from PyPi
Prebuilt versions of this module for python 3.11, 3.12 and 3.13 for Windows and most Linuxsystems can be installed directly by using pip.

``pip install RadFiled3D``

### Installing from Source
You can build and install this library and python module from source by using CMake and a C++ compiler. The CMake Project will be 
built automatically, but will take some time.

#### Prerequisites
- C++ Compiler
  - g++ or clang for Linux
  - MSVC or clang from Visual Studio 2022 for Windows
- CMake >= 3.30
- Python >= 3.11

#### CMake
In order to use the module directly from another C++ Project, you can integrate it by adding the local location of this repository via `add_submodule()` and then link against the target `libRadFiled3D`. All classes are then available from the namespace `RadFiled3D`. Check the [Example](./examples/cxx/example01.cpp) or the [First Test File](./tests/basic.cpp) as a first reference.

#### Python
In order to use the Module from Python, we provide a setup.py file that handles the compilation and integration automatically from the python setuptools.
##### Installing locally
`python -m pip install .`

##### Building a wheel
`python -m build --wheel`

## Getting Started
Disclaimer: Not all methods support keyword arguments as they need to be defined manually in the bindings. For some methods like `add_layer` or the Metadata methods those are implemented.

## From Python
Simple example on how to create and store a radiation field. Find more in the example file: [Example](./examples/python/example01.py)
```python
from RadFiled3D.RadFiled3D import CartesianRadiationField, DType
from RadFiled3D.utils import FieldStore, StoreVersion
from RadFiled3D.metadata.v1 import Metadata


# Creating a cartesian radiation field
field = CartesianRadiationField(vec3(2.5, 2.5, 2.5), vec3(0.05, 0.05, 0.05))
# defining a channel and a layer on it
field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)

# accessing the voxels by using numpy arrays
array = field.get_channel("channel1").get_layer_as_ndarray("layer1")
assert array.shape == (50, 50, 50)
# modify voxels content by using numpy array as no data is copied, just referenced
array[2:5, 2:5, 2:5] = 2.0

# addressing a voxel by providing a point in space
voxel = field.get_channel("channel1").get_voxel_by_coord("layer1", 0.1, 2.4, 5)

# Store changes to a file
metadata = Metadata.default()
FieldStore.store(field, metadata, "test01.rf3", StoreVersion.V1)

# load data
field2 = FieldStore.load("test01.rf3")
metadata2 = FieldStore.load_metadata("test01.rf3")
```

### Integrating with pyTorch
RadFiled3D comes with a submodule at `RadFiled3D.pytorch`. This module provides some dataset classes to support the usage. Datasets can be loaded from folders or .zip-Files.
```python
from RadFiled3D.pytorch.datasets import MetadataLoadMode
from RadFiled3D.pytorch.datasets.cartesian import CartesianFieldSingleLayerDataset
from RadFiled3D.pytorch import DataLoaderBuilder
from RadFiled3D.pytorch.helpers import RadiationFieldHelper
from RadFiled3D.RadFiled3D import VoxelGrid
from torch import Tensor
from RadFiled3D.metadata.v1 import Metadata
from RadFiled3D.pytorch.types import TrainingInputData, DirectionalInput


# Extend one of the provided dataset classes to match the output to the current needs
class MyLayerDataset(CartesianFieldSingleLayerDataset):
    def __getitem____(self, idx: int) -> TrainingInputData:
        layer, metadata = super().__getitem__(idx)
        tube_dir = metadata.get_header().simulation.tube.radiation_direction
        tube_pos = metadata.get_header().simulation.tube.radiation_origin
        # transform the layers data to a tensor
        return TrainingInputData(
            input=DirectionalInput(
                direction=torch.tensor([tube_dir.x, tube_dir.y, tube_dir.z]),
                origin=torch.tensor([tube_pos.x, tube_pos.y, tube_pos.z]),
                spectrum=None
            )
            ground_truth=RadiationFieldHelper.load_tensor_from_layer(layer)
        )


def finalize_dataset(dataset: MyLayerDataset)
    dataset.set_channel_and_layer("test_channel", "test_layer")
    dataset.metadata_load_mode = MetadataLoadMode.HEADER

# Pass the dataset class and other options to the DataLoaderBuilder
builder = DataLoaderBuilder(
    "./test_dataset.zip",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    dataset_class=MyLayerDataset,
    on_dataset_created=finalize_dataset     # Optional: provide a finalizer to perform configuration of the dataset once it was created by the builder
)

# Build the training dataset
train_dl = builder.build_train_dataloader(
    batch_size=8,
    shuffle=True,
    worker_count=4
)

# iterate over the dataset
for field, metadata in train_dl:
    pass
```

#### Direct integration with RadField3D datasets
Directly iterate RadField3D datasets either by loading whole fields or iterating each voxel independently. The dataset classes will return pyTorch compatible NamedTuples, that preserve the structure of the raw radiation fields and layers.
```python
from RadField3D.pytorch.datasets.radfield3d import RadField3DDataset
from RadField3D.pytorch.datasets.radfield3d import RadField3DVoxelwiseDataset
# import the pyTorch compatible datatypes
from RadField3D.pytorch import DataLoaderBuilder
from RadField3D.pytorch.types import DirectionalInput, PositionalInput, TrainingInputData, RadiationField


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
    input: DirectionalInput | PositionalInput = train_data.input
    field: RadiationField = train_data.ground_truth
```
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
from RadFiled3D.RadFiled3D import vec3, GridTracerFactory, GridTracerAlgorithm, CartesianRadiationField, DType

field = CartesianRadiationField(vec3(1.0, 1.0, 1.0), vec3(0.01, 0.01, 0.01))
field.add_channel("test").add_layer("hits", "counts", DType.INT32)
hits_counts = field.get_channel("test").get_layer_as_ndarray("hits")
hits_counts = hits_counts.flatten()

tracer = GridTracerFactory.construct(field, GridTracerAlgorithm.SAMPLING)

indices = tracer.trace(vec3(0.5, 0.5, 0.0), vec3(0.5, 0.85, 1.0))
hits_counts[indices] += 1
grid_shape = field.get_voxel_counts()
hits_counts.reshape((grid_shape.x, grid_shape.y, grid_shape.z))
```

### Faster loading of field series
As the *RadFiled3D* format possesses a dynamic structure, the loading of a radiation field requires the discovery of channels and layers as well as calculating the binary entry points of channels, layers and voxels. When loading datasets for machine learning, the structure of the fields loaded will likely be constant for each dataset. Therefore, the binary entry points can be precalculated to access only those parts of the *RadFiled3D* files that are really needed to increase the loading speed and to reduce the needed memory. This is relealized by the **FieldAccessors** objects.
```python
from RadFiled3D.RadFiled3D import CartesianFieldAccessor, FieldType, uvec3
from RadFiled3D.utils import FieldStore
from RadFiled3D.metadata.v1 import Metadata

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
#include <RadFiled3D/storage/RadiationFieldStore.hpp>
#include <RadFiled3D/RadiationField.hpp>

using namespace RadFiled3D;
using namespace RadFiled3D::Storage;

void main() {
    auto field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f)); // field extents: 2.5 m x 2.5 m x 2.5 m and voxel extents: 5 cm x 5 cm x 5 cm

    auto metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
        // learn about the existing data fields from the example file in ./examples/cxx/examples01.cpp
    )

    FieldStore::store(field, metadata, "test_field.rf3", StoreVersion::V1);

    auto field2 = FieldStore::load("test_field.rf3");
}
```

### Available Voxel Datatypes
In general, a C++ Scalar- or HistogramVoxel (and thus layers) can hold any datatype. But in order to deserialize them from a file or use them from Python, there is only a specific list implemented. The Available datatypes are:
| C++ Type   | RadFiled3D.DType  |
| --------   | ------------  |
| float      | DType.FLOAT32 |
| double     | DType.FLOAT64 |
| int        | DType.INT32   |
| uint8_t    | DType.BYTE  |
| unsigned char    | DType.BYTE  |
| char    | DType.SCHAR  |
| uint32_t   | DType.UINT32  |
| uint64_t   | DType.UINT64  |
| unsigned long long | DType.UINT64  |
| glm::vec2     | DType.VEC2 |
| glm::vec3     | DType.VEC3 |
| glm::vec4     | DType.VEC4 |
| HistogramVoxel<float> | DType.HISTOGRAM |


## Field Structure
RadFiled3D defines a field structure, that provides the user with the possibility to first define in which kind of space he wants to operate. Therefore one can choose between `CartesianRadiationField` and `PolarRadiationField`.
- *CartesianRadiationField*: Segments a room defined by an extent of the room itself and each cuboid voxel into a set of voxels. Each voxel can be addressed by a 3D position (coordinate: x, y, z), a 3D index (number of the voxel in each dimension) or a flat 1D index.
- *PolarRadiationField*: Segements the surface of a unit sphere into surface segments. Each segment (voxel) can be addressed by a 2D position (coordinate: theta, phi), a 2D index (number of the segment in each dimension) or a flat 1D index.

Fields are then partitioned into channels (`VoxelGridBuffer`/`PolarSegmentsBuffer`). All channels share the same size and resolution. A channel is again partitioned into layers (`VoxelGrid`/`PolarSegment`). Each layer holds the actual voxel data and can be constructed from various data types (float, double, uint32_t, uint64_t, glm::vec2, glm::vec3, glm::vec4, N-D-Histogram (list of floats)). Additionally, a layer has a unit string assigned to it as well as a statistical uncertainty to perserve those information.

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