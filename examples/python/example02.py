from RadFiled3D.RadFiled3D import vec3, GridTracerFactory, GridTracerAlgorithm, CartesianRadiationField, DType
from plotly import graph_objects as go

# This file demonstrates how to accumulate the voxels intersected by a line segment
# directly into a radiation field layer, as a simulation would do during particle tracking.

field = CartesianRadiationField(vec3(1.0, 1.0, 1.0), vec3(0.01, 0.01, 0.01))
field.add_channel("test").add_layer("flux", "counts", DType.INT32)

tracer = GridTracerFactory.construct(field, GridTracerAlgorithm.SAMPLING)

# The tracer returns flat voxel indices of the voxels the segment passes through.
indices = tracer.trace(vec3(0.5, 0.5, 0.0), vec3(0.5, 0.85, 1.0))

# get_layer_as_ndarray returns a zero-copy view of shape (x, y, z, elements_per_voxel)
# in which x varies fastest -- the same order the flat indices above are numbered in.
# Flattening with order="F" therefore keeps the view alive, so the increments end up
# in the field. Using .flatten() (or reshape without order="F") would copy the data
# in C order, which both loses the writes and maps the indices to the wrong voxels.
hits_counts = field.get_channel("test").get_layer_as_ndarray("flux")
hits_counts.reshape(-1, order="F")[indices] += 1

# drop the trailing per-voxel element axis to get the (x, y, z) grid
hits_per_voxel = hits_counts[..., 0]
print(f"traced {len(indices)} voxels, {int((hits_per_voxel > 0).sum())} of them set in the field")

# display the hits projected along z
fig = go.Figure(data=go.Heatmap(z=hits_per_voxel.sum(axis=2)))
fig.show()
