from .cartesian import CartesianFieldDataset
from RadFiled3D.RadFiled3D import CartesianRadiationField, VoxelCollectionRequest, VoxelCollection, VoxelCollectionAccessor, FieldShape, vec2
from RadFiled3D.metadata.v1 import Metadata
from .base import MetadataLoadMode
from RadFiled3D.pytorch.types import RadiationField, TrainingInputData, DirectionalInput, RadiationFieldChannel, PositionalInput
from RadFiled3D.pytorch.helpers import RadiationFieldHelper
import torch
from torch import Tensor
from typing import Union
from .processing import DataProcessing


class RadField3DDataset(CartesianFieldDataset):
    """
    A dataset for radiation fields generated with the RadField3D generator.
    This dataset is a subclass of CartesianFieldDataset and provides methods to access radiation fields.
    It is designed to work with the RadField3D format, which includes metadata and radiation fields.
    The dataset can be initialized with a list of file paths or a zip file containing the radiation fields.
    It supports loading radiation fields and their metadata, and provides a method to access the radiation field data as PyTorch tensors.
    The dataset returns instances of TrainingInputData, which contains the input as a DirectionalInput as well as the ground truth as a RadiationField.
    The shape of the ground truth tensors is (c, x, y, z) c is the number of channels (typically 32 for spectra and 1 for all other layers), and (x, y, z) are the dimensions of the radiation field.
    The shape of the input tensor is (3,) for the tube direction and (n,) for the tube spectrum, where n is the number of bins in the tube spectrum.
    The dataset can apply data processing techniques to the input data.
    The dataset respects beam shape parameters, if available in the metadata.
    """
    def __init__(self, file_paths: list[str] = None, zip_file: str = None, data_processings: list[DataProcessing] = None):
        super().__init__(file_paths=file_paths, zip_file=zip_file, metadata_load_mode=MetadataLoadMode.FULL)
        self.data_processings = data_processings if data_processings is not None else []

    def __len__(self):
        dataset_size = super().__len__()
        multiplicator = 1.0
        if self.data_processings is not None:
            for aug in self.data_processings:
                multiplicator *= aug.dataset_multiplier()
        return int(dataset_size * multiplicator)

    def __getitem__(self, idx: int) -> TrainingInputData:
        field, metadata = super().__getitem__(idx)
        assert isinstance(field, CartesianRadiationField), "Dataset must contain CartesianRadiationFields."
        assert isinstance(metadata, Metadata), "Metadata must be of type RadiationFieldMetadataV1."
        return self.transform2training_input(field, metadata)

    def transform2training_input(self, field: CartesianRadiationField, metadata: Metadata) -> TrainingInputData:
        with torch.no_grad():
            rad_field = RadiationField(
                scatter_field=RadiationFieldChannel(
                    spectrum=RadiationFieldHelper.load_tensor_from_field(field, "scatter_field", "spectrum"),
                    fluence=RadiationFieldHelper.load_tensor_from_field(field, "scatter_field", "hits"),
                    error=RadiationFieldHelper.load_tensor_from_field(field, "scatter_field", "error")
                ),
                xray_beam= RadiationFieldChannel(
                    spectrum=RadiationFieldHelper.load_tensor_from_field(field, "xray_beam", "spectrum"),
                    fluence=RadiationFieldHelper.load_tensor_from_field(field, "xray_beam", "hits"),
                    error=RadiationFieldHelper.load_tensor_from_field(field, "xray_beam", "error")
                )
            )

            metadata_header = metadata.get_header()
            abc = (
                metadata_header.simulation.tube.radiation_direction.x,
                metadata_header.simulation.tube.radiation_direction.y,
                metadata_header.simulation.tube.radiation_direction.z
            )
            tube_origin = (
                metadata_header.simulation.tube.radiation_origin.x,
                metadata_header.simulation.tube.radiation_origin.y,
                metadata_header.simulation.tube.radiation_origin.z
            )
            tube_direction = torch.tensor([abc[0], abc[1], abc[2]], dtype=torch.float32)
            tube_origin = torch.tensor([tube_origin[0], tube_origin[1], tube_origin[2]], dtype=torch.float32)
            field_dimensions = (
                field.get_voxel_counts().x * field.get_voxel_dimensions().x,
                field.get_voxel_counts().y * field.get_voxel_dimensions().y,
                field.get_voxel_counts().z * field.get_voxel_dimensions().z
            )
            field_dimensions = torch.tensor([field_dimensions[0], field_dimensions[1], field_dimensions[2]], dtype=torch.float32)
            tube_origin = (tube_origin + (0.5 * field_dimensions)) / field_dimensions # Normalize to [0, 1], assuming the origin is at the center of the field
            tube_spectrum = metadata.simulation.tube.spectrum
            if tube_spectrum is not None:
                tube_spectrum = torch.tensor(tube_spectrum[:, 1], dtype=torch.float32)  # Only take the counts, not the bin edges

            field_shape = metadata.simulation.tube.field_shape
            field_shape_params = None
            if field_shape is not None:
                if field_shape == FieldShape.CONE:
                    field_shape_params = torch.tensor([metadata.simulation.tube.opening_angle_deg], dtype=torch.float32)
                elif field_shape == FieldShape.RECTANGLE:
                    rect: vec2 = metadata.simulation.tube.field_rect_dimensions_m
                    field_shape_params = torch.tensor([rect.x, rect.y], dtype=torch.float32)
                elif field_shape == FieldShape.ELLIPSIS:
                    angles: vec2 = metadata.simulation.tube.field_ellipsis_opening_angles_deg
                    field_shape_params = torch.tensor([angles.x, angles.y], dtype=torch.float32)
                else:
                    raise RuntimeError(f"Unsupported field shape: {field_shape}")
                field_shape = torch.tensor([int(field_shape)], dtype=torch.float32)

            input = DirectionalInput(
                direction=tube_direction,
                origin=tube_origin,
                spectrum=tube_spectrum,
                beam_shape_parameters=field_shape_params,
                beam_shape_type=field_shape
            )

            return TrainingInputData(
                input=input,
                ground_truth=rad_field
            )
        
    def apply_processings(self, input: TrainingInputData) -> TrainingInputData:
        """
        Apply all data processing modules to the input data.
        :param input: The input data to process.
        :return: The processed input data.
        """
        for processing in self.data_processings:
            input = processing(input)
        return input


class RadField3DVoxelwiseDataset(RadField3DDataset):
    """
    A dataset for radiation fields generated with the RadField3D generator that loads single voxels.
    This dataset is a subclass of RadField3DDataset and interates a RadField3D datasets per voxel.
    It provides methods to access radiation fields and their metadata, and provides a method to access the radiation field data as PyTorch tensors.
    The dataset returns instances of TrainingInputData, which contains the input as a PositionalInput as well as the ground truth as a RadiationField.
    The shape of the ground truth tensors is (c, x, y, z) c is the number of channels (typically 32 for spectra and 1 for all other layers), and (x, y, z) are the dimensions of the radiation field.
    The shape of the input tensor is (3,) for the voxel position in normalized world space [0..1] as well as the tube direction and (n,) for the tube spectrum, where n is the number of bins in the tube spectrum.
    The dataset respects beam shape parameters, if available in the metadata.
    The dataset respects geometry information, if available in the .rf3 files.
    """

    def __init__(self, file_paths: list[str] = None, zip_file: str = None, data_processings: list[DataProcessing] = None):
        super().__init__(file_paths=file_paths, zip_file=zip_file, data_processings=data_processings)
        field = self._get_field(0)
        self.field_voxel_counts = field.get_voxel_counts()
        self.field_dimensions = (
            field.get_voxel_counts().x * field.get_voxel_dimensions().x,
            field.get_voxel_counts().y * field.get_voxel_dimensions().y,
            field.get_voxel_counts().z * field.get_voxel_dimensions().z
        )
        self.voxels_per_field = self.field_voxel_counts.x * self.field_voxel_counts.y * self.field_voxel_counts.z
        self.cached_metadata: DirectionalInput = None
        self.cached_fields: RadiationField = None
        self._has_geometry = field.has_channel("geometry")

    def fetch_data2cache(self, files: list[str], external_fields_cache: RadiationField = None, external_metadata_cache: DirectionalInput = None) -> tuple[DirectionalInput, RadiationField]:
        """
        Fetches the data for a single voxel and returns it as a TrainingInputData instance.
        This method is used to load the data for a single voxel from the dataset.
        :param files: List of file paths to the radiation fields.
        :param external_fields_cache: Optional external cache for the radiation fields. Default: None, which means the internal cache will be used.
        :param external_metadata_cache: Optional external cache for the metadata. Default: None, which means the internal cache will be used.
        :return: A tuple containing the metadata and the radiation fields as RadiationField and DirectionalInput instances.
        """
        fields_cache = self.cached_fields if external_fields_cache is None else external_fields_cache
        metadata_cache = self.cached_metadata if external_metadata_cache is None else external_metadata_cache
        for i, file in enumerate(files):
            field = self._get_field_by_path(file)
            metadata = self._get_metadata_by_path(file)
            data = self.transform2training_input(field, metadata)
            metadata_cache.direction[i] = data.input.direction.detach()
            metadata_cache.spectrum[i] = data.input.spectrum.detach()
            fields_cache.scatter_field.spectrum[i] = data.ground_truth.scatter_field.spectrum.detach()
            fields_cache.scatter_field.fluence[i] = data.ground_truth.scatter_field.fluence.detach()
            fields_cache.scatter_field.error[i] = data.ground_truth.scatter_field.error.detach()
            fields_cache.xray_beam.spectrum[i] = data.ground_truth.xray_beam.spectrum.detach()
            fields_cache.xray_beam.fluence[i] = data.ground_truth.xray_beam.fluence.detach()
            fields_cache.xray_beam.error[i] = data.ground_truth.xray_beam.error.detach()
        metadata_cache.direction.requires_grad_(False)
        metadata_cache.spectrum.requires_grad_(False)
        fields_cache.scatter_field.spectrum.requires_grad_(False)
        fields_cache.scatter_field.fluence.requires_grad_(False)
        fields_cache.scatter_field.error.requires_grad_(False)
        fields_cache.xray_beam.spectrum.requires_grad_(False)
        fields_cache.xray_beam.fluence.requires_grad_(False)
        fields_cache.xray_beam.error.requires_grad_(False)
        
        return metadata_cache, fields_cache

    def prefetch_data(self):
        """
        Prefetches all data in the dataset to speed up training.
        This method loads all radiation fields and their metadata into memory.
        """
        if self.cached_metadata is not None and self.cached_fields is not None:
            raise RuntimeError("Data has already been prefetched. Please create a new dataset instance to reload the data.")
        if len(self.file_paths) == 0:
            raise RuntimeError("No files found in the dataset. Please check the file paths or zip file.")
        first_data = super().__getitem__(0)
        with torch.no_grad():
            field_voxel_counts = first_data.ground_truth.scatter_field.spectrum.shape[1:4]
            self.cached_metadata = DirectionalInput(
                direction=torch.empty((len(self.file_paths), 3), dtype=torch.float32).share_memory_(),
                origin=torch.empty((len(self.file_paths), 3), dtype=torch.float32).share_memory_(),
                spectrum=torch.empty((len(self.file_paths), first_data.input.spectrum.shape[0]), dtype=torch.float32).share_memory_(),
                geometry=torch.empty((len(self.file_paths), *first_data.input.geometry.shape), dtype=torch.float32).share_memory_() if first_data.input.geometry is not None else None,
                beam_shape_parameters=torch.empty((len(self.file_paths), first_data.input.beam_shape_parameters.shape[0]), dtype=torch.float32).share_memory_() if first_data.input.beam_shape_parameters is not None else None,
                beam_shape_type=torch.empty((len(self.file_paths), 1), dtype=torch.float32).share_memory_() if first_data.input.beam_shape_type is not None else None
            )

            self.cached_fields = RadiationField(
                scatter_field=RadiationFieldChannel(
                    spectrum=torch.empty((len(self.file_paths), first_data.input.spectrum.shape[0], *field_voxel_counts), dtype=torch.float32).share_memory_(),
                    fluence=torch.empty((len(self.file_paths), 1, *field_voxel_counts), dtype=torch.float32).share_memory_(),
                    error=torch.empty((len(self.file_paths), 1, *field_voxel_counts), dtype=torch.float32).share_memory_()
                ),
                xray_beam=RadiationFieldChannel(
                    spectrum=torch.empty((len(self.file_paths), first_data.input.spectrum.shape[0], *field_voxel_counts), dtype=torch.float32).share_memory_(),
                    fluence=torch.empty((len(self.file_paths), 1, *field_voxel_counts), dtype=torch.float32).share_memory_(),
                    error=torch.empty((len(self.file_paths), 1, *field_voxel_counts), dtype=torch.float32).share_memory_()
                )
            )

            self.cached_metadata, self.cached_fields = self.fetch_data2cache(self.file_paths)
    
    def load_voxel_training_data_from_cache(self, idx: Union[int, Tensor], xyz: Union[Tensor, tuple[int, int, int]], external_fields_cache: RadiationField = None, external_metadata_cache: DirectionalInput = None) -> TrainingInputData:
        cached_fields = self.cached_fields if external_fields_cache is None else external_fields_cache
        cached_metadata = self.cached_metadata if external_metadata_cache is None else external_metadata_cache
        xyz = torch.tensor(xyz, dtype=torch.float32, device=cached_fields.scatter_field.fluence.device, requires_grad=False) if not isinstance(xyz, Tensor) else xyz
        xyz_idx = xyz.long()

        field = RadiationField(
            scatter_field=RadiationFieldChannel(
                spectrum=cached_fields.scatter_field.spectrum[idx, :, xyz_idx[0], xyz_idx[1], xyz_idx[2]].clone(),
                fluence=cached_fields.scatter_field.fluence[idx, :, xyz_idx[0], xyz_idx[1], xyz_idx[2]].clone(),
                error=cached_fields.scatter_field.error[idx, :, xyz_idx[0], xyz_idx[1], xyz_idx[2]].clone()
            ),
            xray_beam=RadiationFieldChannel(
                spectrum=cached_fields.xray_beam.spectrum[idx, :, xyz_idx[0], xyz_idx[1], xyz_idx[2]].clone(),
                fluence=cached_fields.xray_beam.fluence[idx, :, xyz_idx[0], xyz_idx[1], xyz_idx[2]].clone(),
                error=cached_fields.xray_beam.error[idx, :, xyz_idx[0], xyz_idx[1], xyz_idx[2]].clone()
            )
        ) if cached_fields is not None else None
        # normalize xyz to 0 to 1
        field_voxel_counts = torch.tensor([self.field_voxel_counts.x, self.field_voxel_counts.y, self.field_voxel_counts.z], dtype=torch.float32, device=xyz.device, requires_grad=False)
        xyz = xyz / (field_voxel_counts - 1.0) # Normalize xyz to [0, 1]

        input = PositionalInput(
            position=xyz,
            direction=cached_metadata.direction[idx].clone(),
            origin=cached_metadata.origin[idx].clone(),
            spectrum=cached_metadata.spectrum[idx].clone(),
            geometry=cached_metadata.geometry[idx].clone() if cached_metadata.geometry is not None else None,
            beam_shape_parameters=cached_metadata.beam_shape_parameters[idx].clone() if cached_metadata.beam_shape_parameters is not None else None,
            beam_shape_type=cached_metadata.beam_shape_type[idx].clone() if cached_metadata.beam_shape_type is not None else None
        ) if cached_metadata is not None else None

        return TrainingInputData(
            input=input,
            ground_truth=field
        )

    def __getitem__(self, idx: int) -> TrainingInputData:
        voxel_idx = idx % self.voxels_per_field
        file_idx = idx // self.voxels_per_field
        xyz = (
            voxel_idx % self.field_voxel_counts.x,
            (voxel_idx // self.field_voxel_counts.x) % self.field_voxel_counts.y,
            voxel_idx // (self.field_voxel_counts.x * self.field_voxel_counts.y)
        )
        xyz = torch.tensor([xyz[0], xyz[1], xyz[2]], dtype=torch.float32, requires_grad=False)

        if self.cached_metadata is not None:
            return self.load_voxel_training_data_from_cache(file_idx, xyz)
        else:
            scatter_spectrum = self._get_voxel_flat(file_idx=file_idx, vx_idx=voxel_idx, channel_name="scatter_field", layer_name="spectrum").get_histogram()
            scatter_fluence = self._get_voxel_flat(file_idx=file_idx, vx_idx=voxel_idx, channel_name="scatter_field", layer_name="hits").get_data()
            scatter_error = self._get_voxel_flat(file_idx=file_idx, vx_idx=voxel_idx, channel_name="scatter_field", layer_name="error").get_data()
            xray_spectrum = self._get_voxel_flat(file_idx=file_idx, vx_idx=voxel_idx, channel_name="xray_beam", layer_name="spectrum").get_histogram()
            xray_fluence = self._get_voxel_flat(file_idx=file_idx, vx_idx=voxel_idx, channel_name="xray_beam", layer_name="hits").get_data()
            xray_error = self._get_voxel_flat(file_idx=file_idx, vx_idx=voxel_idx, channel_name="xray_beam", layer_name="error").get_data()
            geometry = self._get_voxel_flat(file_idx=file_idx, vx_idx=voxel_idx, channel_name="geometry", layer_name="density").get_data() if self._has_geometry else None
            field = RadiationField(
                scatter_field=RadiationFieldChannel(
                    spectrum=torch.tensor(scatter_spectrum, dtype=torch.float32, device=xyz.device, requires_grad=False),
                    fluence=torch.tensor(scatter_fluence, dtype=torch.float32, device=xyz.device, requires_grad=False),
                    error=torch.tensor(scatter_error, dtype=torch.float32, device=xyz.device, requires_grad=False)
                ),
                xray_beam=RadiationFieldChannel(
                    spectrum=torch.tensor(xray_spectrum, dtype=torch.float32, device=xyz.device, requires_grad=False),
                    fluence=torch.tensor(xray_fluence, dtype=torch.float32, device=xyz.device, requires_grad=False),
                    error=torch.tensor(xray_error, dtype=torch.float32, device=xyz.device, requires_grad=False)
                )
            )
            metadata: Metadata = self._get_metadata(file_idx)
            metadata_header = metadata.get_header()
            abc = (
                metadata_header.simulation.tube.radiation_direction.x,
                metadata_header.simulation.tube.radiation_direction.y,
                metadata_header.simulation.tube.radiation_direction.z
            )
            tube_origin = (
                metadata_header.simulation.tube.radiation_origin.x,
                metadata_header.simulation.tube.radiation_origin.y,
                metadata_header.simulation.tube.radiation_origin.z
            )
            tube_direction = torch.tensor([abc[0], abc[1], abc[2]], dtype=torch.float32, device=xyz.device, requires_grad=False)
            tube_origin = torch.tensor([tube_origin[0], tube_origin[1], tube_origin[2]], dtype=torch.float32, device=xyz.device, requires_grad=False)
            field_dimensions = torch.tensor([self.field_dimensions[0], self.field_dimensions[1], self.field_dimensions[2]], dtype=torch.float32)
            tube_origin = (tube_origin + (0.5 * field_dimensions)) / field_dimensions # Normalize to [0, 1], assuming the origin is at the center of the field
            tube_spectrum = metadata.simulation.tube.spectrum
            if tube_spectrum is not None:
                tube_spectrum = torch.tensor(tube_spectrum[:, 1], dtype=torch.float32, device=xyz.device, requires_grad=False)

            field_shape = metadata.simulation.tube.field_shape
            beam_shape_params = None
            if field_shape is not None:
                if field_shape == FieldShape.CONE:
                    beam_shape_params = torch.tensor([metadata.simulation.tube.opening_angle_deg], dtype=torch.float32)
                elif field_shape == FieldShape.RECTANGLE:
                    rect: vec2 = metadata.simulation.tube.field_rect_dimensions_m
                    beam_shape_params = torch.tensor([rect.x, rect.y], dtype=torch.float32)
                elif field_shape == FieldShape.ELLIPSIS:
                    angles: vec2 = metadata.simulation.tube.field_ellipsis_opening_angles_deg
                    beam_shape_params = torch.tensor([angles.x, angles.y], dtype=torch.float32)
                else:
                    raise RuntimeError(f"Unsupported field shape: {field_shape}")
                field_shape = torch.tensor([int(field_shape)], dtype=torch.float32)

            field_voxel_counts = torch.tensor([self.field_voxel_counts.x, self.field_voxel_counts.y, self.field_voxel_counts.z], dtype=torch.float32, device=xyz.device, requires_grad=False)
            xyz = xyz / (field_voxel_counts - 1.0) # Normalize xyz to [0, 1]

            input = PositionalInput(
                position=xyz,
                origin=tube_origin,
                direction=tube_direction,
                spectrum=tube_spectrum,
                geometry=geometry,
                beam_shape_parameters=beam_shape_params,
                beam_shape_type=field_shape
            )
            return TrainingInputData(
                input=input,
                ground_truth=field
            )

    def __len__(self):
        return super().__len__() * self.voxels_per_field

    def __getitems__(self, indices) -> Union[TrainingInputData, list[TrainingInputData]]:
        indices = torch.tensor(
            indices,
            dtype=torch.int64,
            device=self.cached_fields.scatter_field.fluence.device if self.cached_fields is not None else None,
            requires_grad=False
        ) if not isinstance(indices, Tensor) else indices
        file_indices = indices // self.voxels_per_field
        voxel_indices = indices % self.voxels_per_field
        xyz = torch.empty((len(indices), 3), dtype=torch.float32, requires_grad=False, device=indices.device)
        xyz[:, 0] = voxel_indices % self.field_voxel_counts.x
        xyz[:, 1] = (voxel_indices // self.field_voxel_counts.x) % self.field_voxel_counts.y
        xyz[:, 2] = voxel_indices // (self.field_voxel_counts.x * self.field_voxel_counts.y)

        if self.cached_metadata is not None and self.cached_fields is not None:
            return self.load_voxel_training_data_from_cache(file_indices, xyz)
        else:
            accessor = VoxelCollectionAccessor(
                self.field_accessor,
                [
                    "scatter_field",
                    "xray_beam"
                ],
                [
                    "spectrum",
                    "hits",
                    "error"
                ]
            )
            geom_accessor = VoxelCollectionAccessor(self.field_accessor, ["geometry"], ["density"]) if self._has_geometry else None
            field_voxel_counts = torch.tensor([self.field_voxel_counts.x, self.field_voxel_counts.y, self.field_voxel_counts.z], dtype=torch.float32, device=xyz.device, requires_grad=False)

            requests = []
            indices = torch.tensor(indices, dtype=torch.int64, device=self.cached_fields.scatter_field.fluence.device, requires_grad=False) if not isinstance(indices, Tensor) else indices
            file_indices = indices // self.voxels_per_field
            unique_file_indices = torch.unique(file_indices)
            voxel_indices = indices % self.voxels_per_field
            metadata: PositionalInput = None
            vx_count = 0
            for file_idx in unique_file_indices:
                voxel_request = VoxelCollectionRequest(
                    self.file_paths[file_idx.item()],
                    voxel_indices=voxel_indices[file_indices == file_idx].cpu().numpy()
                )
                requests.append(voxel_request)
                raw_metadata: Metadata = self._get_metadata(file_idx.item())
                metadata_header = raw_metadata.get_header()
                abc = (
                    metadata_header.simulation.tube.radiation_direction.x,
                    metadata_header.simulation.tube.radiation_direction.y,
                    metadata_header.simulation.tube.radiation_direction.z
                )
                tube_origin = (
                    metadata_header.simulation.tube.radiation_origin.x,
                    metadata_header.simulation.tube.radiation_origin.y,
                    metadata_header.simulation.tube.radiation_origin.z
                )
                tube_spectrum = torch.tensor(raw_metadata.simulation.tube.spectrum[:, 1], dtype=torch.float32, device=xyz.device, requires_grad=False)
                tube_origin = torch.tensor([tube_origin[0], tube_origin[1], tube_origin[2]], dtype=torch.float32, device=xyz.device, requires_grad=False)
                field_dimensions = torch.tensor([self.field_dimensions[0], self.field_dimensions[1], self.field_dimensions[2]], dtype=torch.float32)
                tube_origin = (tube_origin + (0.5 * field_dimensions)) / field_dimensions # Normalize to [0, 1], assuming the origin is at the center of the field
                field_shape = raw_metadata.simulation.tube.field_shape
                beam_shape_params = None
                if field_shape is not None:
                    if field_shape == FieldShape.CONE:
                        beam_shape_params = torch.tensor([raw_metadata.simulation.tube.opening_angle_deg], dtype=torch.float32)
                    elif field_shape == FieldShape.RECTANGLE:
                        rect: vec2 = raw_metadata.simulation.tube.field_rect_dimensions_m
                        beam_shape_params = torch.tensor([rect.x, rect.y], dtype=torch.float32)
                    elif field_shape == FieldShape.ELLIPSIS:
                        angles: vec2 = raw_metadata.simulation.tube.field_ellipsis_opening_angles_deg
                        beam_shape_params = torch.tensor([angles.x, angles.y], dtype=torch.float32)
                    else:
                        raise RuntimeError(f"Unsupported field shape: {field_shape}")
                    field_shape = torch.tensor([int(field_shape)], dtype=torch.float32)

                if metadata is None:
                    metadata: PositionalInput = PositionalInput(
                        position=xyz / (field_voxel_counts - 1.0),  # Normalize xyz to [0, 1]
                        direction=torch.empty((len(indices), 3), dtype=torch.float32, device=xyz.device, requires_grad=False),
                        origin=torch.empty((len(indices), 3), dtype=torch.float32, device=xyz.device, requires_grad=False),
                        spectrum=torch.empty((len(indices), tube_spectrum.shape[-1]), dtype=torch.float32, device=xyz.device, requires_grad=False),
                        geometry=None,
                        beam_shape_parameters=torch.empty((len(indices), 1), dtype=torch.float32, device=xyz.device, requires_grad=False) if raw_metadata.simulation.tube.field_shape is not None else None,
                        beam_shape_type=torch.empty((len(indices), 1), dtype=torch.float32, device=xyz.device, requires_grad=False) if raw_metadata.simulation.tube.field_shape is not None else None
                    )

                metadata.direction[vx_count:vx_count + len(voxel_request.voxel_indices), 0] = abc[0]
                metadata.direction[vx_count:vx_count + len(voxel_request.voxel_indices), 1] = abc[1]
                metadata.direction[vx_count:vx_count + len(voxel_request.voxel_indices), 2] = abc[2]
                
                metadata.spectrum[vx_count:vx_count + len(voxel_request.voxel_indices), :] = tube_spectrum
                if field_shape is not None:
                    metadata.beam_shape_parameters[vx_count:vx_count + len(voxel_request.voxel_indices), :] = beam_shape_params
                    metadata.beam_shape_type[vx_count:vx_count + len(voxel_request.voxel_indices), :] = field_shape
                metadata.origin[vx_count:vx_count + len(voxel_request.voxel_indices), :] = tube_origin
                vx_count += len(voxel_request.voxel_indices)

            collection: VoxelCollection = accessor.access(requests)
            geom_collection: VoxelCollection = geom_accessor.access(requests) if geom_accessor is not None else None

            return TrainingInputData(
                input=PositionalInput(
                    position=metadata.position,
                    direction=metadata.direction,
                    origin=metadata.origin,
                    spectrum=metadata.spectrum,
                    beam_shape_parameters=metadata.beam_shape_parameters,
                    beam_shape_type=metadata.beam_shape_type,
                    geometry=torch.tensor(geom_collection.get_as_ndarray("geometry", "density"), device=xyz.device, requires_grad=False).unsqueeze(-1)
                ) if geom_collection is not None else metadata,
                ground_truth=RadiationField(
                    scatter_field=RadiationFieldChannel(
                        spectrum=torch.tensor(collection.get_as_ndarray("scatter_field", "spectrum"), device=xyz.device, requires_grad=False),
                        fluence=torch.tensor(collection.get_as_ndarray("scatter_field", "hits"), device=xyz.device, requires_grad=False).unsqueeze(-1),
                        error=torch.tensor(collection.get_as_ndarray("scatter_field", "error"), device=xyz.device, requires_grad=False).unsqueeze(-1)
                    ),
                    xray_beam=RadiationFieldChannel(
                        spectrum=torch.tensor(collection.get_as_ndarray("xray_beam", "spectrum"), device=xyz.device, requires_grad=False),
                        fluence=torch.tensor(collection.get_as_ndarray("xray_beam", "hits"), device=xyz.device, requires_grad=False).unsqueeze(-1),
                        error=torch.tensor(collection.get_as_ndarray("xray_beam", "error"), device=xyz.device, requires_grad=False).unsqueeze(-1)
                    )
                )
            )


class RadField3DDatasetWithGeometry(RadField3DDataset):
    """
    A 3D dataset class for RadField3D with additional geometric information.
    If an .rf file contains a channel named "geometry", its "density" layer will be loaded as a tensor of shape (1, D, W, H).
    """

    def __init__(self, file_paths: list[str] = None, zip_file: str = None, data_processings: list[DataProcessing] = None, create_binary_geometry_mask: bool = False, normalize_geometry: bool = False):
        """
        Initialize the dataset with the given parameters.
        Args:
            file_paths: A list of file paths to the .rf files. Those shall be relative, if loading from a zip file.
            zip_file: A zip file containing the .rf files.
            data_processings: A list of data processing functions to apply to the dataset.
            create_binary_geometry_mask: Whether to create a binary mask for the geometry.
            normalize_geometry: Whether to normalize the geometry tensor by using z-score normalization.
        """
        super().__init__(file_paths, zip_file, data_processings)
        self.create_binary_geometry_mask = create_binary_geometry_mask
        self.should_normalize_geometry = normalize_geometry

    @staticmethod
    def normalize_geometry(geometry_tensor: torch.Tensor) -> torch.Tensor:
        return (geometry_tensor - geometry_tensor.mean()) / (geometry_tensor.std() + 1e-6)

    def transform2training_input(self, field: CartesianRadiationField, metadata: Metadata) -> TrainingInputData:
        data = super().transform2training_input(field, metadata)
        if field.has_channel("geometry"):
            geometry_tensor = RadiationFieldHelper.load_tensor_from_field(field, "geometry", "density").to(torch.float32)
            if self.create_binary_geometry_mask:
                geometry_tensor[geometry_tensor > 0.0] = 1.0  # Normalize geometry tensor to binary values
            if self.should_normalize_geometry:
                geometry_tensor = RadField3DDatasetWithGeometry.normalize_geometry(geometry_tensor)
            data = TrainingInputData(
                input=DirectionalInput(
                    direction=data.input.direction,
                    origin=data.input.origin,
                    spectrum=data.input.spectrum,
                    beam_shape_parameters=data.input.beam_shape_parameters,
                    beam_shape_type=data.input.beam_shape_type,
                    geometry=geometry_tensor
                ),
                ground_truth=data.ground_truth
            )
        else:
            data = TrainingInputData(
                input=DirectionalInput(
                    direction=data.input.direction,
                    origin=data.input.origin,
                    spectrum=data.input.spectrum,
                    beam_shape_parameters=data.input.beam_shape_parameters,
                    beam_shape_type=data.input.beam_shape_type,
                    geometry=None
                ),
                ground_truth=data.ground_truth
            )
        return data
