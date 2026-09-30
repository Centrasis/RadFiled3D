"""RadFiled3D: storage and tooling for 3D radiation fields.

The C++ core is exposed by the private extension module ``radfiled3d._core`` and
re-exported here, so the data model is imported straight from the package::

    from radfiled3d import CartesianRadiationField, DType

Two areas live in their own modules to keep this namespace to the data model:

* ``radfiled3d.glm``   -- the vector types (``vec3``, ``uvec3``, ...)
* ``radfiled3d.store`` -- reading and writing ``.rf3`` files (``FieldStore``,
  ``StoreVersion``, the field accessors and the join modes)

High-level helpers live in ``radfiled3d.metadata`` and ``radfiled3d.pytorch``.
"""
from ._core import (
    # --- errors -------------------------------------------------------------
    InvalidArgument,
    OutOfRange,
    RadFiled3DError,
    RadiationFieldStoreException,
    VoxelBufferException,
    # --- enums --------------------------------------------------------------
    DType,
    FieldShape,
    FieldType,
    GridTracerAlgorithm,
    # --- build configuration ------------------------------------------------
    HAS_FLOAT16,
    # --- voxels -------------------------------------------------------------
    Voxel,
    AngularResolvedVoxel,
    ByteVoxel,
    Float32Voxel,
    Float64Voxel,
    HistogramVoxel,
    Int32Voxel,
    Int64Voxel,
    SCharVoxel,
    UInt32Voxel,
    UInt64Voxel,
    VMFMixtureVoxel,
    Vec2Voxel,
    Vec3Voxel,
    Vec4Voxel,
    OwningAngularResolvedVoxel,
    OwningByteVoxel,
    OwningFloat32Voxel,
    OwningFloat64Voxel,
    OwningHistogramVoxel,
    OwningInt32Voxel,
    OwningInt64Voxel,
    OwningSCharVoxel,
    OwningUInt32Voxel,
    OwningUInt64Voxel,
    OwningVMFMixtureVoxel,
    OwningVec2Voxel,
    OwningVec3Voxel,
    OwningVec4Voxel,
    # --- layers, channels and fields ----------------------------------------
    VoxelLayer,
    VoxelBuffer,
    VoxelGrid,
    VoxelGridBuffer,
    PolarSegments,
    PolarSegmentsBuffer,
    RadiationField,
    CartesianRadiationField,
    PolarRadiationField,
    # --- raw file metadata (see radfiled3d.metadata.v1 for the friendly API) --
    RadiationFieldMetadata,
    RadiationFieldMetadataV1,
    RadiationFieldMetadataHeaderV1,
    RadiationFieldSimulationMetadataV1,
    RadiationFieldSoftwareMetadataV1,
    RadiationFieldXRayTubeMetadataV1,
    # --- grid tracing -------------------------------------------------------
    GridTracer,
    GridTracerFactory,
    BresenhamGridTracer,
    LinetracingGridTracer,
    SamplingGridTracer,
    # --- voxel collections (dataset access) ---------------------------------
    VoxelCollection,
    VoxelCollectionAccessor,
    VoxelCollectionRequest,
)

# Float16 needs a compiler providing _Float16 (see HAS_FLOAT16), so these two classes
# are absent from builds without it -- importing them unconditionally would make the
# whole package fail to import on such a build (the windows and glibc 2.17 wheels).
if HAS_FLOAT16:
    from ._core import Float16Voxel, OwningFloat16Voxel

# Derived rather than repeated: every name imported above is public.
__all__ = [name for name in globals() if not name.startswith("_")]
