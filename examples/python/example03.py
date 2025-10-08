from RadFiled3D.RadFiled3D import FieldShape
from plotly import graph_objects as go
import numpy
import sys
from RadFiled3D.metadata.v1 import Metadata
from RadFiled3D.utils import FieldStore


# This file demonstrates how to load a radiation field files metadata and which information can be extracted from it.

radiation_file = sys.argv[1]

metadata: Metadata = FieldStore.peek_metadata(radiation_file)
metadata = FieldStore.load_metadata(radiation_file)
energy_eV = metadata.simulation.tube.max_energy_eV
print(f"Energy: {energy_eV / 1e3} keV")
direction = metadata.simulation.tube.radiation_direction
print(f"Direction: {direction}")
origin = metadata.simulation.tube.radiation_origin
print(f"Origin: {origin}")
geometry = metadata.simulation.geometry
print(f"Geometry: {geometry}")
particles = metadata.simulation.primary_particle_count
print(f"Particles: {particles}")
dyn_data_keys = metadata.get_dynamic_metadata_keys()
print(f"Dynamic metadata keys: {dyn_data_keys}")
simulation_duration_s = metadata.get_dynamic_metadata("simulation_duration_s").get_data()
print(f"Simulation duration: {simulation_duration_s} s")
tube_spectrum = metadata.simulation.tube.spectrum
print(f"Tube spectrum: {tube_spectrum}")
shape_type = metadata.simulation.tube.field_shape
print(f"Field shape type: {shape_type}")
if shape_type == FieldShape.CONE:
    opening_angle_deg = metadata.simulation.tube.opening_angle_deg
    print(f"Opening angle: {opening_angle_deg} deg")
elif shape_type == FieldShape.RECTANGLE:
    extends_at_origin = metadata.simulation.tube.field_rect_dimensions_m
    print(f"Field extends at origin: {extends_at_origin} m")
elif shape_type == FieldShape.ELLIPSIS:
    ellipsis_opening_angles = metadata.simulation.tube.field_ellipsis_opening_angles_deg
    print(f"Field ellipsis opening angles: {ellipsis_opening_angles} deg")
else:
    print("Unknown field shape type")


# plot tube spectrum
fig = go.Figure(data=go.Scatter(x=numpy.arange(0, (tube_spectrum.get_bins() - 1) * tube_spectrum.get_histogram_bin_width(), tube_spectrum.get_histogram_bin_width()) - tube_spectrum.get_histogram_bin_width(), y=tube_spectrum.get_histogram()))
fig.show()
