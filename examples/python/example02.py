from RadFiled3D.RadFiled3D import vec3, GridTracerFactory, GridTracerAlgorithm, CartesianRadiationField, DType
from plotly import graph_objects as go

field = CartesianRadiationField(vec3(1.0, 1.0, 1.0), vec3(0.01, 0.01, 0.01))
field.add_channel("test").add_layer("flux", "counts", DType.INT32)
hits_counts = field.get_channel("test").get_layer_as_ndarray("flux")
hits_counts_shape = hits_counts.shape
hits_counts = hits_counts.flatten()
hits_counts.fill(0)

tracer = GridTracerFactory.construct(field, GridTracerAlgorithm.SAMPLING)

indices = tracer.trace(vec3(0.5, 0.5, 0.0), vec3(0.5, 0.85, 1.0))
hits_counts[indices] += 1

hits_counts = hits_counts.reshape(hits_counts_shape)

# display the hits from xyz
fig = go.Figure(data=go.Heatmap(z=hits_counts.sum(axis=2)))
fig.show()
