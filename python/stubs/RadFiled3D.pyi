import numpy as np
from typing import Any, Tuple
from enum import Enum


class FieldShape(Enum):
    CONE = 0
    RECTANGLE = 1
    ELLIPSIS = 2


class DType(Enum):
    FLOAT32 = 0
    FLOAT64 = 1
    INT32 = 2
    SCHAR = 3
    VEC2 = 4
    VEC3 = 5
    VEC4 = 6
    HISTOGRAM = 7
    UINT64 = 8
    UINT32 = 9
    BYTE = 10


class FieldType(Enum):
    CARTESIAN = 0
    POLAR = 1


class FieldJoinMode(Enum):
    """
    The mode how to join two radiation fields.
    """

    IDENTITY = 0
    "Use the value of the target field"

    ADD = 1
    "Add the values of the target and the additional source field"

    MEAN = 2
    "Calculate the mean of the values of the target and the additional source field"

    SUBTRACT = 3
    "Subtract the values of the additional source field from the target field"

    DIVIDE = 4
    "Divide the values of the target field by the values of the additional source field"

    MULTIPLY = 5
    "Multiply the values of the target field by the values of the additional source field"

    ADD_WEIGHTED = 6
    "Add the values of the target and the additional source field with a weighting ratio based on the primary particles"


class FieldJoinCheckMode(Enum):
    """
    The mode to join the fields.
	All modes stack on each other.
    So STRICT includes all checks.
    The following mode includes all modes except for STRICT and so on.
    """

    STRICT = 0
    "Check if the metadata and field structure is equal. If not, throw an exception"

    METADATA_SIMULATION_SIMILAR = 1
    "Check if the metadata is similar (e.g. Geometry, Radiation-Direction, xray-tube). If not, throw an exception"

    METADATA_SOFTWARE_EQUAL = 2
    "Check if the software metadata is equal. If not, throw an exception"

    METADATA_SOFTWARE_SIMILAR = 3
    "Check if the software metadata is similar (e.g. Software-Name, Software-Repository). If not, throw an exception"

    FIELD_STRUCTURE_ONLY = 4
    "Check if the fields share the same channel-layer structure. If not, throw an exception"

    FIELD_UNITS_ONLY = 5
    "Check if the fields layers share the same units. If not, throw an exception"

    NO_CHECKS = 6
    "Do not perform any semantic checks. Technical checks will still be performed"


class GridTracerAlgorithm(Enum):
    SAMPLING = 0
    BRESENHAM = 1
    LINETRACING = 2


class StoreVersion(Enum):
    V1 = 0


class vec4:
    x: float
    y: float
    z: float
    w: float

    def __init__(self, x: float, y: float, z: float, w: float) -> None: ...


class uvec4:
    x: int
    y: int
    z: int
    w: int

    def __init__(self, x: int, y: int, z: int, w: int) -> None: ...


class vec3:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float) -> None: ...


class uvec3:
    x: int
    y: int
    z: int

    def __init__(self, x: int, y: int, z: int) -> None: ...


class vec2:
    x: float
    y: float

    def __init__(self, x: float, y: float) -> None: ...


class uvec2:
    x: int
    y: int

    def __init__(self, x: int, y: int) -> None: ...


class RadiationFieldXRayTubeMetadataV1(object):
    radiation_direction: vec3
    radiation_origin: vec3
    max_energy_eV: float
    tube_id: str

    def __init__(self, radiation_direction: vec3, radiation_origin: vec3, max_energy_eV: float, tube_id: str) -> None: ...


class RadiationFieldSimulationMetadataV1(object):
    primary_particle_count: int
    geometry: str
    physics_list: str
    tube: RadiationFieldXRayTubeMetadataV1

    def __init__(self, primary_particle_count: int, geometry: str, physics_list: str, tube: RadiationFieldXRayTubeMetadataV1) -> None: ...


class RadiationFieldSoftwareMetadataV1(object):
    name: str
    version: str
    repository: str
    commit: str
    doi: str

    def __init__(self, name: str, version: str, repository: str, commit: str, doi: str = "") -> None: ...


class RadiationFieldMetadataHeaderV1(object):
    simulation: RadiationFieldSimulationMetadataV1
    software: RadiationFieldSoftwareMetadataV1

    def __init__(self, simulation: RadiationFieldSimulationMetadataV1, software: RadiationFieldSoftwareMetadataV1) -> None: ...


class RadiationFieldMetadata(object):
    pass


class RadiationFieldMetadataV1(RadiationFieldMetadata):
    def __init__(self, simulation: RadiationFieldSimulationMetadataV1, software: RadiationFieldSoftwareMetadataV1) -> None: ...
    def get_header(self) -> RadiationFieldMetadataHeaderV1:
        """
        Returns the mandatory header of the metadata.
        The header contains the simulation and software metadata.
        
        :return: The header of the metadata.
        """
        ...

    def set_header(self, header: RadiationFieldMetadataHeaderV1) -> None:
        """
        Sets the mandatory header of the metadata.
        The header contains the simulation and software metadata.
        
        :param header: The header to set.
        """
        ...

    def get_dynamic_metadata_keys(self) -> list[str]:
        """
        Returns a list of all dynamic metadata keys.
        Dynamic metadata keys are used to store additional information about the radiation field.
        The keys are used to store additional information about the radiation field.
        The values of the keys are Voxel objects that can be used to store any data type.

        :return: A list of all dynamic metadata keys.
        """
        ...

    def get_dynamic_metadata(self, key: str) -> Voxel:
        """
        Returns the dynamic metadata for a given key.
        The key is used to store additional information about the radiation field.
        The value of the key is a Voxel that can be used to store any data type.

        :param key: The key of the metadata.
        :return: The Voxel that can be used to store the value.
        """
        ...

    def add_dynamic_metadata(self, key: str, dtype: DType) -> Voxel:
        """
        Adds a dynamic metadata key to the radiation field metadata.
        The key is used to store additional information about the radiation field.
        The value of the key is a Voxel that can be used to store any data type.

        :param key: The key to add.
        :param dtype: The data type of the value.
        :return: The Voxel that can be used to store the value.
        """
        ...

    def add_dynamic_histogram_metadata(self, key: str, bins: int, bin_width: float) -> HistogramVoxel:
        """
        Adds a dynamic histogram metadata to the radiation field metadata.

        :param key: The key of the metadata.
        :param bins: The number of bins in the histogram.
        :param bin_width: The width of each bin in the histogram.
        :return: The histogram voxel representing the dynamic metadata.
        """
        ...


class Voxel(object):
    """
    A Voxel is a single element in a VoxelBuffer. It can be of any type, but must be able to be converted to a raw byte array.
	Voxels do not store their own data, but rather point to a buffer that contains the data. This is to allow for the data to be
	stored in a single contiguous block of memory, which is more efficient for the GPU to read. The buffer is managed by the VoxelBuffer
	class, and the Voxel class is only a view into the buffer. This means that the Voxel class does not own the data, and should not
	be used to manage the data. The VoxelBuffer class is responsible for managing the data, and the Voxel class is only a view into
	that data. The VoxelBuffer class is responsible for allocating and deallocating the data, and the Voxel class is only responsible
	for reading and writing the data.
    """

    def get_data(self) -> Any:
        """
        Returns the reference to the value of the voxel.
        """
        ...

    def set_data(self, value: Any) -> None:
        """
        Sets the value of the voxel.

        :param value: The value to set the voxel to.
        """
        ...

    def __eq__(self, value: "Voxel") -> bool: ...


class Float32Voxel(Voxel):
    def get_data(self) -> float: ...
    def __eq__(self, value: "Float32Voxel") -> bool: ...


class SCharVoxel(Voxel):
    def get_data(self) -> int: ...
    def __eq__(self, value: "SCharVoxel") -> bool: ...


class ByteVoxel(Voxel):
    def get_data(self) -> int: ...
    def __eq__(self, value: "ByteVoxel") -> bool: ...


class UInt32Voxel(Voxel):
    def get_data(self) -> int: ...
    def __eq__(self, value: "UInt32Voxel") -> bool: ...


class UInt64Voxel(Voxel):
    def get_data(self) -> int: ...
    def __eq__(self, value: "UInt64Voxel") -> bool: ...


class Int32Voxel(Voxel):
    def get_data(self) -> int: ...
    def __eq__(self, value: "Int32Voxel") -> bool: ...


class Float64Voxel(Voxel):
    def get_data(self) -> float: ...
    def __eq__(self, value: "Float64Voxel") -> bool: ...


class Vec2Voxel(Voxel):
    def get_data(self) -> vec2: ...
    def __eq__(self, value: "Vec2Voxel") -> bool: ...


class Vec3Voxel(Voxel):
    def get_data(self) -> vec3: ...
    def __eq__(self, value: "Vec3Voxel") -> bool: ...


class Vec4Voxel(Voxel):
    def get_data(self) -> vec4: ...
    def __eq__(self, value: "Vec4Voxel") -> bool: ...


class HistogramVoxel(Voxel):
    def get_histogram(self) -> np.ndarray:
        """
        Returns the histogram data of the voxel.

        :return: The histogram data of the voxel.
        """
        ...

    def get_data(self) -> np.ndarray:
        """
        Returns the histogram data of the voxel.

        :return: The histogram data of the voxel.
        """
        ...

    def __eq__(self, value: "HistogramVoxel") -> bool: ...

    def get_bins(self) -> int:
        """
        Returns the number of bins in the histogram.

        :return: The number of bins in the histogram.
        """
        ...

    def get_histogram_bin_width(self) -> int:
        """
        Returns the bin width of the histogram.

        :return: The bin width of the histogram.
        """
        ...

    def add_value(self, value: float) -> None:
        """
        Adds a positive value to the histogram and scores it into the correct bin.
		If the value is greater than the maximum value, it is scored in the last bin.
		If the value is less than 0, it is scored in the first bin.

        :param value: The value to add.
        """
        ...

    def normalize(self) -> None:
        """
        Normalizes the histogram so that the sum of all bins is 1, if possible.
        """
        ...


class VoxelBuffer(object):
    def get_voxel_count(self) -> int:
        """
        Returns the linear number of voxels in the buffer.
        """
        ...

    def has_layer(self, layer_name: str) -> bool:
        """
        Check if a layer exists in the buffer.

        :param layer_name: The name of the layer to check for.
        :return: True if the layer exists, otherwise False.
        """
        ...

    def get_layers(self) -> list[str]:
        """
        Returns a list of all layer names in the buffer.
        """
        ...

    def get_layer_unit(self, layer_name: str) -> str:
        """
        Returns the unit of a layer.

        :param layer_name: The name of the layer.
        :return: The unit of the layer.
        """
        ...

    def get_statistical_error(self, layer_name: str) -> float:
        """
        Returns the statistical error of a layer.

        :param layer_name: The name of the layer.
        :return: The statistical error of the layer.
        """
        ...

    def set_statistical_error(self, layer_name: str, statistical_error: float) -> None:
        """
        Set the statistical error of a layer.

        :param layer_name: The name of the layer.
        :param statistical_error: The statistical error to set.
        """
        ...

    def get_layer_voxel_type(self, layer_name: str) -> str:
        """
        Returns the type of the voxels in a layer.

        :param layer_name: The name of the layer.
        :return: The type of the voxels in the layer.
        """
        ...

    def get_layer_as_ndarray(self, layer_name: str) -> np.ndarray:
        """
        Returns the layer as a numpy ndarray.
        The ndarray will have the shape depending on the concrete BufferType.
        :see VoxelGridBuffer and PolarSegmentsBuffer
        VoxelGridBuffer will return a 3D ndarray with the shape (x, y, z).
        PolarSegmentsBuffer will return a 2D ndarray with the shape (x, y).
        Depending if the layer is a histogram or not, the ndarray will have an additional dimension for the histogram bins.
        The ndarray will have the dtype depending on the concrete VoxelScalar-Type.

        :param layer_name: The name of the layer.
        :return: The layer as a numpy ndarray.
        """
        ...

    def add_layer(self, layer_name: str, unit: str, dtype: DType) -> None:
        """
        Add a new layer to the buffer.

        :param layer_name: The name of the layer.
        :param unit: The unit of the layer.
        :param dtype: The data type of the layer.
        """
        ...

    def add_histogram_layer(self, layer_name: str, bins: int, bin_width: float, unit: str) -> None:
        """
        Add a new histogram layer to the buffer.

        :param layer_name: The name of the layer.
        :param bins: The number of bins in the histogram.
        :param bin_width: The bin width of the histogram.
        :param unit: The unit of the layer.
        """
        ...


class VoxelLayer(object):
    """
    Interface for voxel layers.
    A voxel layer is a collection of voxels.
    The voxel layer is responsible for managing the voxels.
    Class is intended to be used as a read-only interface.
    """

    def get_voxel_flat(self, idx: int) -> Voxel: ...

    def get_unit(self) -> str: ...

    def get_statistical_error(self) -> float: ...
    
    def get_voxel_count(self) -> int: ...


class VoxelGrid(object):
    """
    Interface for voxel grids.
    A voxel grid is a collection of voxels in a 3D grid.
    The voxel grid is responsible for managing the voxels.
    Class is intended to be used as a read-only interface.
    """

    def __init__(self, field_dimensions: vec3, voxel_dimensions: vec3, layer: VoxelLayer = None) -> None: ...

    def get_voxel_dimensions(self) -> vec3: ...

    def get_voxel_counts(self) -> uvec3: ...

    def get_voxel_idx(self, x: int, y: int, z: int) -> int: ...

    def get_voxel_idx_by_coord(self, x: float, y: float, z: float) -> int: ...

    def get_voxel(self, x: int, y: int, z: int) -> Voxel: ...

    def get_voxel_by_coord(self, x: float, y: float, z: float) -> Voxel: ...

    def get_layer(self) -> VoxelLayer: ...

    def get_as_ndarray(self) -> np.ndarray: ...


class PolarSegments(object):
    """
    Interface for polar segments.
    A polar segment is a collection of voxels in a 2D grid on a unit sphere.
    The polar segment is responsible for managing the voxels.
    Class is intended to be used as a read-only interface.
    """

    def __init__(self, segments_counts: uvec2, layer: VoxelLayer) -> None: ...

    def get_segments_count(self) -> uvec2: ...

    def get_segment_idx(self, x: int, y: int) -> int: ...

    def get_segment_idx_by_coord(self, x: float, y: float) -> int: ...

    def get_segment(self, x: int, y: int) -> Voxel: ...

    def get_segment_by_coord(self, x: float, y: float) -> Voxel: ...

    def get_layer(self) -> VoxelLayer: ...

    def get_as_ndarray(self) -> np.ndarray: ...


class VoxelGridBuffer(VoxelBuffer):
    def get_voxel_counts(self) -> uvec3:
        """
        Returns the number of voxels in each dimension.

        :return: The number of voxels in each dimension.
        """
        ...

    def get_voxel_dimensions(self) -> vec3:
        """
        Returns the dimensions of each voxel.

        :return: The dimensions of each voxel.
        """
        ...

    def get_voxel(self, layer: str, x: int, y: int, z: int) -> Voxel:
        """
        Get a voxel at a specific quantized index for each dimension.

        :param layer: The name of the layer.
        :param x: The x index of the voxel.
        :param y: The y index of the voxel.
        :param z: The z index of the voxel.
        :return: The voxel at the specified index.
        """
        ...

    def get_voxel_by_coord(self, layer: str, x: float, y: float, z: float) -> Voxel:
        """
        Get a voxel at specific continuous coordinates.

        :param layer: The name of the layer.
        :param x: The x coordinate of the voxel.
        :param y: The y coordinate of the voxel.
        :param z: The z coordinate of the voxel.
        :return: The voxel at the specified coordinates.
        """
        ...

    def get_voxel_flat(self, layer: str, idx: int) -> Voxel:
        """
        Get a voxel at a specific linear index.

        :param layer: The name of the layer.
        :param idx: The index of the voxel.
        :return: The voxel at the specified index.
        """
        ...

    def get_voxel_idx(self, x: int, y: int, z: int) -> int:
        """
        Get the linear index of a voxel at specific quantized indices.

        :param x: The x index of the voxel.
        :param y: The y index of the voxel.
        :param z: The z index of the voxel.
        :return: The linear index of the voxel.
        """
        ...

    def get_voxel_idx_by_coord(self, x: float, y: float, z: float) -> int:
        """
        Get the linear index of a voxel at specific continuous coordinates.

        :param x: The x coordinate of the voxel.
        :param y: The y coordinate of the voxel.
        :param z: The z coordinate of the voxel.
        :return: The linear index of the voxel.
        """
        ...


class PolarSegmentsBuffer(VoxelBuffer):
    def get_segments_count(self) -> uvec2:
        """
        Returns the number of segments in each dimension.

        :return: The number of segments in each dimension.
        """
        ...

    def get_segment_by_coord(self, layer: str, phi: float, theta: float) -> Voxel:
        """
        Get a segment at specific continuous coordinates.

        :param layer: The name of the layer.
        :param phi: The phi coordinate of the segment.
        :param theta: The theta coordinate of the segment.
        :return: The segment at the specified coordinates.
        """
        ...

    def get_segment(self, layer: str, x: int, y: int) -> Voxel:
        """
        Get a segment at specific quantized indices.

        :param layer: The name of the layer.
        :param x: The x index of the segment.
        :param y: The y index of the segment.
        :return: The segment at the specified indices.
        """
        ...

    def get_segment_flat(self, layer: str, idx: int) -> Voxel:
        """
        Get a segment at a specific linear index.

        :param layer: The name of the layer.
        :param idx: The index of the segment.
        :return: The segment at the specified index.
        """
        ...

    def get_segment_idx(self, x: int, y: int) -> int:
        """
        Get the linear index of a segment at specific quantized indices.

        :param x: The x index of the segment.
        :param y: The y index of the segment.
        :return: The linear index of the segment.
        """
        ...

    def get_segment_idx_by_coord(self, x: float, y: float) -> int:
        """
        Get the linear index of a segment at specific continuous coordinates.

        :param x: The x coordinate of the segment.
        :param y: The y coordinate of the segment.
        :return: The linear index of the segment.
        """
        ...


class RadiationField(object):
    """
    Interface for radiation fields.
	A radiation field is a collection of channels which are a layered collection of voxels and therefore a collection of VoxelBuffers.
	Each channel is a collection of layered voxels aka a VoxelBuffer.
	The channels are identified by a name.
	The radiation field is responsible for managing the channels.
	The radiation field is responsible for creating the channels.
	The radiation field is responsible for providing the channels.
	The radiation field is responsible for providing the type name of the concrete class.
    """

    def get_typename(self) -> str:
        """
        Returns the type name of the concrete class.
        """
        ...

    def has_channel(self, name: str) -> bool:
        """
        Check if a channel exists in the radiation field.

        :param name: The name of the channel.
        :return: True if the channel exists, otherwise False.
        """
        ...

    def get_channel(self, name: str) -> VoxelBuffer:
        """
        Get a channel by name.

        :param name: The name of the channel.
        :return: The channel.
        """
        ...

    def get_channel_names(self) -> list[str]:
        """
        Returns a list of all channel names in the radiation field.
        """
        ...

    def get_channels(self) -> list[Tuple[str, VoxelBuffer]]:
        """
        Returns a list of all channels in the radiation field.
        """
        ...

    def add_channel(self, name: str) -> VoxelBuffer:
        """
        Add a new channel to the radiation field.

        :param name: The name of the channel.
        :return: The new channel.
        """
        ...

    def get_channel(self, name: str) -> VoxelBuffer:
        """
        Get a channel by name.

        :param name: The name of the channel.
        :return: The channel.
        """
        ...

    def copy(self) -> "RadiationField":
        """
		Returns a deep copy of the radiation field.
		"""
        ...

    def __enter__(self) -> "RadiationField":
        """
        Prepare the radiation field.
        """
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Clean up the radiation field.
        """
        ...


class CartesianRadiationField(RadiationField):
    """
    A Cartesian radiation field.
	A Cartesian radiation field is a radiation field with a Cartesian grid of voxels (VoxelGridBuffer).
	The field is defined by the dimensions of the field and the dimensions of a voxel.
	The field is responsible for creating channels of voxels.
    """

    def __init__(self, field_dimensions: vec3, voxel_dimensions: vec3) -> None:
        """
        Create a new Cartesian radiation field.

        :param field_dimensions: The dimensions of the field in meter.
        :param voxel_dimensions: The dimensions of a voxel in meter.
        """
        ...

    def get_field_dimensions(self) -> vec3:
        """
        Returns the dimensions of the field in meter.

        :return: The dimensions of the field.
        """
        ...

    def get_voxel_dimensions(self) -> vec3:
        """
        Returns the dimensions of a voxel in meter.

        :return: The dimensions of a voxel.
        """
        ...

    def get_voxel_counts(self) -> uvec3:
        """
        Returns the number of voxels in each dimension.

        :return: The number of voxels in each dimension.
        """
        ...

    def add_channel(self, name: str) -> VoxelGridBuffer:
        """
        Add a new channel to the radiation field.

        :param name: The name of the channel.
        :return: The new channel.
        """
        ...

    def get_channel(self, name: str) -> VoxelGridBuffer:
        """
        Get a channel by name.

        :param name: The name of the channel.
        :return: The channel.
        """
        ...

    def copy(self) -> "CartesianRadiationField":
        """
        Returns a deep copy of the radiation field.
        """
        ...


class PolarRadiationField(RadiationField):
    """
    A Polar radiation field.
	A Polar radiation field is a radiation field with a spherical grid of voxels (PolarSegmentsBuffer).
	The field is defined by the number of segments in each dimension.
	The field is responsible for creating channels of voxels.
    """

    def __init__(self, segments_count: uvec2) -> None:
        """
        Create a new Polar radiation field.

        :param segments_count: The number of segments in each dimension.
        """
        ...

    def get_segments_count(self) -> uvec2:
        """
        Returns the number of segments in each dimension.

        :return: The number of segments in each dimension.
        """
        ...

    def add_channel(self, name: str) -> PolarSegmentsBuffer:
        """
        Add a new channel to the radiation field.

        :param name: The name of the channel.
        :return: The new channel.
        """
        ...

    def get_channel(self, name: str) -> PolarSegmentsBuffer:
        """
        Get a channel by name.

        :param name: The name of the channel.
        :return: The channel.
        """
        ...

    def copy(self) -> "PolarRadiationField":
        """
        Returns a deep copy of the radiation field.
        """
        ...



class FieldAccessor:
    def get_field_type(self) -> FieldType:
        """
        Returns the type of the radiation field.
        """
        ...

    def get_voxel_count(self) -> int:
        """
        Returns the linear number of voxels in the buffer.
        """
        ...
    
    @staticmethod
    def get_store_version(data: bytes) -> StoreVersion:
        """
        Returns the store version of the radiation field buffer
        """
        ...

    def access_voxel_flat_from_buffer(self, buffer: bytes, channel_name: str, layer_name: str, idx: int) -> Voxel:
        """
        Get a voxel at a specific linear index from a data buffer.

        :param buffer: The buffer to load the radiation field from.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :param idx: The index of the voxel.
        :return: The voxel at the specified index.
        """
        ...

    def access_voxel_flat(self, file: str, channel_name: str, layer_name: str, idx: int) -> Voxel:
        """
        Get a voxel at a specific linear index from a file.

        :param file: The file path to the stored radiation field.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :param idx: The index of the voxel.
        :return: The voxel at the specified index.
        """
        ...

    def access_field_from_buffer(self, buffer: bytes) -> RadiationField:
        """
        Get a radiation field from a data buffer.

        :param buffer: The buffer to load the radiation field from.
        :return: The radiation field.
        """
        ...

    def access_field(self, file: str) -> RadiationField:
        """
        Get a radiation field from a file.

        :param file: The file path to the stored radiation field.
        :return: The radiation field.
        """
        ...


class CartesianFieldAccessor(FieldAccessor):
    def access_channel_from_buffer(self, buffer: bytes, channel_name: str) -> VoxelGridBuffer:
        """
        Get a channel by name from a data buffer.

        :param buffer: The buffer to load the radiation field from.
        :param channel_name: The name of the channel.
        :return: The channel.
        """
        ...

    def access_channel(self, file: str, channel_name: str) -> VoxelGridBuffer:
        """
        Get a channel by name from a file

        :param file: The file path to the stored radiation field.
        :param channel_name: The name of the channel.
        :return: The channel.
        """
        ...

    def access_layer_from_buffer(self, buffer: bytes, channel_name: str, layer_name: str) -> VoxelGrid:
        """
        Get a layer by name from a data buffer.

        :param buffer: The buffer to load the radiation field from.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :return: The layer.
        """
        ...

    def access_layer(self, file: str, channel_name: str, layer_name: str) -> VoxelGrid:
        """
        Get a layer by name from a file.

        :param file: The file path to the stored radiation field.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :return: The layer.
        """
        ...

    def access_layer_across_channels_from_buffer(self, buffer: bytes, layer_name: str) -> dict[str, VoxelGrid]:
        """
        Get a layer by name from a data buffer across all channels.

        :param buffer: The buffer to load the radiation field from.
        :param layer_name: The name of the layer.
        :return: All layers with the specified name across all channels with the channel name as key.
        """
        ...

    def access_layer_across_channels(self, file: str, layer_name: str) -> dict[str, VoxelGrid]:
        """
        Get a layer by name from a file across all channels.

        :param file: The file path to the stored radiation field.
        :param layer_name: The name of the layer.
        :return: All layers with the specified name across all channels with the channel name as key.
        """
        ...

    def access_voxel_from_buffer(self, buffer: bytes, channel_name: str, layer_name: str, idx: uvec3) -> Voxel:
        """
        Get a voxel at a specific quantized index from a data buffer.

        :param buffer: The buffer to load the radiation field from.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :param idx: The index of the voxel.
        :return: The voxel at the specified index.
        """
        ...
    
    def access_voxel(self, file: str, channel_name: str, layer_name: str, idx: uvec3) -> Voxel:
        """
        Get a voxel at a specific quantized index from a file.

        :param file: The file path to the stored radiation field.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :param idx: The index of the voxel.
        :return: The voxel at the specified index.
        """
        ...

    def access_voxel_by_coord_from_buffer(self, buffer: bytes, channel_name: str, layer_name: str, coord: vec3) -> Voxel:
        """
        Get a voxel at specific continuous coordinates from a data buffer.

        :param buffer: The buffer to load the radiation field from.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :param coord: The coordinates of the voxel.
        :return: The voxel at the specified coordinates.
        """
        ...

    def access_voxel_by_coord(self, file: str, channel_name: str, layer_name: str, coord: vec3) -> Voxel:
        """
        Get a voxel at specific continuous coordinates from a file.

        :param file: The file path to the stored radiation field.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :param coord: The coordinates of the voxel.
        :return: The voxel at the specified coordinates.
        """
        ...


class PolarFieldAccessor(FieldAccessor):
    def access_layer_from_buffer(self, buffer: bytes, channel_name: str, layer_name: str) -> PolarSegments:
        """
        Get a layer by name from a data buffer.

        :param buffer: The buffer to load the radiation field from.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :return: The layer.
        """
        ...

    def access_layer(self, file: str, channel_name: str, layer_name: str) -> PolarSegments:
        """
        Get a layer by name from a file.

        :param file: The file path to the stored radiation field.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :return: The layer.
        """
        ...

    def access_voxel_from_buffer(self, buffer: bytes, channel_name: str, layer_name: str, idx: uvec2) -> Voxel:
        """
        Get a voxel at a specific quantized index from a data buffer.

        :param buffer: The buffer to load the radiation field from.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :param idx: The index of the voxel.
        :return: The voxel at the specified index.
        """
        ...
    
    def access_voxel(self, file: str, channel_name: str, layer_name: str, idx: uvec2) -> Voxel:
        """
        Get a voxel at a specific quantized index from a file.

        :param file: The file path to the stored radiation field.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :param idx: The index of the voxel.
        :return: The voxel at the specified index.
        """
        ...

    def access_voxel_by_coord_from_buffer(self, buffer: bytes, channel_name: str, layer_name: str, coord: vec2) -> Voxel:
        """
        Get a voxel at specific continuous coordinates from a data buffer.

        :param buffer: The buffer to load the radiation field from.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :param coord: The coordinates of the voxel.
        :return: The voxel at the specified coordinates.
        """
        ...

    def access_voxel_by_coord(self, file: str, channel_name: str, layer_name: str, coord: vec2) -> Voxel:
        """
        Get a voxel at specific continuous coordinates from a file.

        :param file: The file path to the stored radiation field.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        :param coord: The coordinates of the voxel.
        :return: The voxel at the specified coordinates.
        """
        ...


class FieldStore:
    @staticmethod
    def enable_file_lock_syncronization(enable: bool) -> None:
        """
        Enable or disable file transaction synchronization. This will make sure, that only one process can perform transactions such as joining on a file at a time and that other processes are queued.
        
        :param enable: Enable or disable file transaction synchronization.
        """
        ...
    

    @staticmethod
    def init_store_instance(version: StoreVersion) -> None:
        """
        Initialize the store instance with a specific version.

        :param version: The version to initialize the store instance with.
        """
        ...

    @staticmethod
    def load_metadata(file: str) -> RadiationFieldMetadata:
        """
        Get the metadata of a stored radiation field.

        :param file: The file path to the stored radiation field.
        """
        ...

    @staticmethod
    def peek_metadata(file: str) -> RadiationFieldMetadata:
        """
        Quickly peeks at the mandatory metadata header of the radiation field from a file

        :param file: The file path to the stored radiation field.
        """
        ...

    @staticmethod
    def load_metadata_from_buffer(buffer: bytes) -> RadiationFieldMetadata:
        """
        Get the metadata of a stored radiation field from a buffer.

        :param buffer: The buffer to load the metadata from.
        """
        ...

    @staticmethod
    def peek_metadata_from_buffer(buffer: bytes) -> RadiationFieldMetadataHeaderV1:
        """
        Quickly peeks at the mandatory metadata header of the radiation field from a buffer

        :param buffer: The buffer to load the metadata from.
        """
        ...

    @staticmethod
    def load(file: str) -> RadiationField:
        """
        Load a stored radiation field.

        :param file: The file path to the stored radiation field.
        """
        ...

    @staticmethod
    def load_from_buffer(buffer: bytes) -> RadiationField:
        """
        Load a stored radiation field from a buffer.

        :param buffer: The buffer to load the radiation field from.
        """
        ...

    @staticmethod
    def load_single_grid_layer_from_buffer(buffer: bytes, channel_name: str, layer_name: str) -> VoxelGrid:
        """
        Load a single layer from a stored radiation field from a buffer.
        Returns a VoxelGrid.

        :param buffer: The buffer to load the radiation field from.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        """
        ...

    @staticmethod
    def load_single_polar_layer_from_buffer(buffer: bytes, channel_name: str, layer_name: str) -> PolarSegments:
        """
        Load a single layer from a stored radiation field from a buffer.
        Returns a PolarSegments.

        :param buffer: The buffer to load the radiation field from.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        """
        ...

    @staticmethod
    def store(field: RadiationField, metadata: RadiationFieldMetadata, file: str, version: StoreVersion = StoreVersion.V1) -> None:
        """
        Store a radiation field to a file.

        :param field: The radiation field to store.
        :param metadata: The metadata of the radiation field.
        :param file: The file path to store the radiation field to.
        :param version: The version to store the radiation field with.
        """
        ...

    @staticmethod
    def get_store_version(file: str) -> StoreVersion:
        """
        Get the store version of a stored radiation field.

        :param file: The file path to the stored radiation field.
        """
        ...

    @staticmethod
    def join(field: RadiationField, metadata: RadiationFieldMetadata, file: str, join_mode: FieldJoinMode, check_mode: FieldJoinCheckMode, fallback_version: StoreVersion = StoreVersion.V1) -> None:
        """
        Join a radiation field to an existing stored radiation field.
        Creates a new stored radiation field if no radiation field was present at the file path.

        :param field: The radiation field to join.
        :param metadata: The metadata of the radiation field.
        :param file: The file path to the stored radiation field.
        :param join_mode: The mode to join the radiation fields with.
        :param check_mode: The mode to check the radiation fields with.
        :param fallback_version: The version to fallback to if there wasn't already a radiation field present whose version could be used.
        """
        ...

    @staticmethod
    def load_single_grid_layer(file: str, channel_name: str, layer_name: str) -> VoxelGrid:
        """
        Load a single layer from a stored radiation field.
        Returns a VoxelGrid or PolarSegments depending on the field type.

        :param file: The file path to the stored radiation field.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        """
        ...

    @staticmethod
    def load_single_polar_layer(file: str, channel_name: str, layer_name: str) -> PolarSegments:
        """
        Load a single layer from a stored radiation field.
        Returns a PolarSegments.

        :param file: The file path to the stored radiation field.
        :param channel_name: The name of the channel.
        :param layer_name: The name of the layer.
        """
        ...

    @staticmethod
    def peek_field_type(file: str) -> FieldType:
        """
        Quickly peeks at the type of the radiation field from a file

        :param file: The file path to the stored radiation field.
        """
        ...

    @staticmethod
    def construct_field_accessor(file: str) -> FieldAccessor:
        """
        Construct a radiation field accessor from a file for a set of radiation fields that share the same metadata size and overall field structure.
        This includes channels, layers, and voxel dimensions/counts.

        :param file: The file path to the stored radiation field.
        :return: The radiation field accessor.
        """
        ...

    @staticmethod
    def construct_field_accessor_from_buffer(buffer: bytes) -> FieldAccessor:
        """
        Construct a radiation field accessor from a buffer for a set of radiation fields that share the same metadata size and overall field structure.
        This includes channels, layers, and voxel dimensions/counts.

        :param buffer: The buffer to load the radiation field from.
        :return: The radiation field accessor.
        """
        ...


class GridTracer:
    def trace(self, p1: vec3, p2: vec3) -> list[int]:
        """
        Trace a line between two points in the grid.

        :param p1: The start point of the line.
        :param p2: The end point of the line.
        :return: The indices of the voxels intersected by the line.
        """
        ...


class GridTracerFactory:
    @staticmethod
    def construct(field: CartesianRadiationField, algorithm: GridTracerAlgorithm = GridTracerAlgorithm.SAMPLING) -> GridTracer:
        """
        Construct a grid tracer for a Cartesian radiation field.

        :param field: The Cartesian radiation field to trace in.
        :param algorithm: The algorithm to use for tracing.
        :return: The grid tracer.
        """
        ...


class VoxelCollectionRequest(object):
    file_path: str
    voxel_indices: list[int]

    def __init__(self, file_path: str, voxel_indices: list[int]) -> None:
        """
        Initialize a voxel collection request.

        :param file_path: The file path to the stored radiation field.
        :param voxel_indices: The indices of the voxels to collect.
        """
        ...


class VoxelCollection(object):
    def get_as_ndarray(self, channel: str, layer: str) -> np.ndarray:
        """
        Get the collected voxels as a numpy ndarray.

        :param channel: The name of the channel.
        :param layer: The name of the layer.
        :return: The collected voxels as a numpy ndarray.
        """
        ...


class VoxelCollectionAccessor(object):
    def __init__(self, accessor: FieldAccessor, channels: list[str], layers: list[str]) -> None:
        """
        Initialize a voxel collection accessor.

        :param accessor: The field accessor to use for accessing the radiation field.
        :param channels: The names of the channels to collect.
        :param layers: The names of the layers to collect.
        """
        ...

    def access(self, requests: list[VoxelCollectionRequest]) -> VoxelCollection:
        """
        Load the voxels from the radiation field based on the requests.

        :param requests: The requests for voxel collections.
        :return: The collected voxels as a VoxelCollection.
        """
        ...
