from RadFiled3D.utils import FieldStore
from RadFiled3D.RadFiled3D import CartesianRadiationField, HistogramVoxel
from argparse import ArgumentParser
import os
from rich import print
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import trimesh


def write_histogram2file(bins: np.ndarray, counts: np.ndarray, filename: str):
    with open(filename, "w") as f:
        f.write("Energy bin center in keV;Relative Counts\n")
        for bin_center, count in zip(bins, counts):
            f.write(f"{bin_center};{count}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Join multiple field files into one.")
    parser.add_argument("--input", nargs=1, help="List of input field files to join.")
    parser.add_argument("--render_spectra_of", nargs='+', type=str, help="List of locations which spectra to render.", default=[])
    parser.add_argument("--overlay_mesh", type=str, help="Path to a mesh file to overlay on the volume rendering.", default=None)
    parser.add_argument("--overlay_phantom_name", type=str, help="Name of the phantom mesh in the overlay mesh file.", default="phantom")

    hide_axes = False

    args = parser.parse_args()
    input_file = args.input
    spectra_locations = args.render_spectra_of

    field_bytes = open(input_file[0], "rb").read()
    field: CartesianRadiationField = FieldStore.load_from_buffer(field_bytes)

    meta = FieldStore.load_metadata(input_file[0])
    print(f"Primary particle count: {meta.get_header().simulation.primary_particle_count:.2e}")

    voxel_size = (
        field.get_voxel_dimensions().x,
        field.get_voxel_dimensions().y,
        field.get_voxel_dimensions().z,
    )
    field_dimensions = (
        field.get_field_dimensions().x,
        field.get_field_dimensions().y,
        field.get_field_dimensions().z,
    )

    print(field.get_channel("xray_beam").get_layers())

    volume = field.get_channel("xray_beam").get_layer_as_ndarray("hits") + field.get_channel("scatter_field").get_layer_as_ndarray("hits")
    # flip y and z for correct orientation
    print(f"Volume shape before transpose: {volume.shape}")
    volume = np.transpose(volume, (0, 2, 1, 3))
    #volume =  np.log1p(volume)
    sc_volume = volume.copy()
    sc_volume[volume > volume.max()*0.02] = -1
    sc_volume /= sc_volume.max()
    volume /= volume.max()
    xs=[i * voxel_size[0] for i in range(volume.shape[0]) for j in range(volume.shape[1]) for k in range(volume.shape[2])] 
    ys=[j * voxel_size[1] for i in range(volume.shape[0]) for j in range(volume.shape[1]) for k in range(volume.shape[2])]
    zs=[k * voxel_size[2] for i in range(volume.shape[0]) for j in range(volume.shape[1]) for k in range(volume.shape[2])]
    fig = go.Figure()
    fig.add_trace(
        go.Volume(
            x=xs,
            y=ys,
            z=zs,
            value=sc_volume.flatten(),
            isomin=0.03,
            isomax=1,
            opacity=0.1,
            surface_count=20,
            colorscale='Viridis',
            colorbar=dict(title='Doserate Intensity'),
            name='Scatter Field'
        )
    )
    fig.add_trace(
        go.Volume(
            x=xs,
            y=ys,
            z=zs,
            value=volume.flatten(),
            isomin=0.03,
            isomax=1,
            opacity=0.1,
            surface_count=20,
            colorscale='Viridis',
            colorbar=None,
            name='Full Field'
        )
    )
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [True, True]}],
                        label="Both",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, False]}],
                        label="Scatter Only",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True]}],
                        label="Full Field Only",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, False]}],
                        label="None",
                        method="update"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'  # Transparent background
        ) if hide_axes else None,
    )

    if args.overlay_mesh is not None and os.path.isfile(args.overlay_mesh):
        phantom_name = args.overlay_phantom_name
        scene  = trimesh.load(args.overlay_mesh, process=False, force='scene', split_objects=True, group_material=False)

        mesh_names = list(scene.geometry.keys())
        print(f"Loaded overlay mesh with geometries: {mesh_names}")
        for mesh_name in mesh_names:
            mesh = scene.geometry[mesh_name]
            mesh.apply_transform(trimesh.transformations.rotation_matrix(
                np.radians(90),
                (1, 0, 0)
            ))
            mesh.apply_translation((
                field_dimensions[0] / 2,
                field_dimensions[1] / 2,
                field_dimensions[2] / 2,
            ))

            mesh_vertices = mesh.vertices
            mesh_faces = mesh.faces

            fig.add_trace(
                go.Mesh3d(
                    x=mesh_vertices[:, 0],
                    y=mesh_vertices[:, 1],
                    z=mesh_vertices[:, 2],
                    i=mesh_faces[:, 0],
                    j=mesh_faces[:, 1],
                    k=mesh_faces[:, 2],
                    color='lightpink' if mesh_name == phantom_name else 'lightblue',
                    opacity=1.0,
                    name='Overlay Mesh'
                )
            )
    
    fig.update_layout(
        font=dict(size=16),
        scene=dict(
            xaxis_title="X in m",
            yaxis_title="Y in m",
            zaxis_title="Z in m"
        )
    )
    fig.show()

    print(f"Field size: {field_dimensions}")
    for loc_str in spectra_locations:
        loc = tuple(map(float, loc_str.split(",")))
        loc = (
            loc[0] + (field_dimensions[0] / 2),
            loc[1] + (field_dimensions[1] / 2),
            loc[2] + (field_dimensions[2] / 2),
        )
        print(f"Rendering spectrum at location: {loc}")
        sc_weight: float = field.get_channel("scatter_field").get_voxel_by_coord("hits", loc[0], loc[1], loc[2]).get_data() / (field.get_channel("xray_beam").get_voxel_by_coord("hits", loc[0], loc[1], loc[2]).get_data() + field.get_channel("scatter_field").get_voxel_by_coord("hits", loc[0], loc[1], loc[2]).get_data())
        xb_weight: float = field.get_channel("xray_beam").get_voxel_by_coord("hits", loc[0], loc[1], loc[2]).get_data() / (field.get_channel("xray_beam").get_voxel_by_coord("hits", loc[0], loc[1], loc[2]).get_data() + field.get_channel("scatter_field").get_voxel_by_coord("hits", loc[0], loc[1], loc[2]).get_data())

        smoothed_spectrum = None
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    scatter_spectrum = field.get_channel("scatter_field").get_voxel_by_coord("spectrum", loc[0] + i*voxel_size[0], loc[1] + j*voxel_size[1], loc[2] + k*voxel_size[2]).get_data()
                    xray_beam_spectrum = field.get_channel("xray_beam").get_voxel_by_coord("spectrum", loc[0] + i*voxel_size[0], loc[1] + j*voxel_size[1], loc[2] + k*voxel_size[2]).get_data()
                    hist: HistogramVoxel = field.get_channel("xray_beam").get_voxel_by_coord("spectrum", loc[0] + i*voxel_size[0], loc[1] + j*voxel_size[1], loc[2] + k*voxel_size[2])

                    combined_spectrum = scatter_spectrum * sc_weight + xray_beam_spectrum * xb_weight
                    combined_spectrum /= combined_spectrum.sum()  # Normalize
                    if smoothed_spectrum is None:
                        smoothed_spectrum = combined_spectrum
                    else:
                        smoothed_spectrum += combined_spectrum
        smoothed_spectrum /= 27  # Average over the 3x3x3 neighborhood
        energy_bins = np.arange(0, scatter_spectrum.shape[0] * hist.get_histogram_bin_width(), hist.get_histogram_bin_width()) + hist.get_histogram_bin_width() / 2.0 # in MeV

        # New line plot for combined_spectrum
        line_fig = px.line(x=energy_bins * 1000, y=smoothed_spectrum, title=f"Combined Spectrum at location {loc}")
        line_fig.update_layout(
            xaxis_title="Energy (keV)",
            yaxis_title="Weighted Counts"
        )
        line_fig.show()

        write_histogram2file(energy_bins * 1000, smoothed_spectrum, f"spectrum_{loc_str.replace(',', '_')}.csv")

    slice_2d = volume[volume.shape[1] // 2, :, :, :].reshape(volume.shape[0], volume.shape[-2])
    heatmap_slice = px.imshow(slice_2d.T, title=f"volume slice at center")
    heatmap_slice.update_layout(
        xaxis_title="Energy (keV)",
        yaxis_title="Counts"
    )
    heatmap_slice.show()
