from RadFiled3D.RadFiled3D import vec2, vec3, FieldShape, RadiationFieldMetadataV1, HistogramVoxel, DType, Voxel, RadiationFieldSoftwareMetadataV1, RadiationFieldXRayTubeMetadataV1, RadiationFieldSimulationMetadataV1, RadiationFieldMetadataHeaderV1
import __main__
import os
import numpy as np


class Metadata(object):
    class Software(object):
        def _get_name(self) -> str:
            return self._metadata.get_header().software.name
        
        def _set_name(self, name: str) -> None:
            header = self._metadata.get_header()
            header.software.name = name + chr(0) # Ensure null-terminated string
            self._metadata.set_header(header)

        def _get_version(self) -> str:
            return self._metadata.get_header().software.version
        
        def _set_version(self, version: str) -> None:
            header = self._metadata.get_header()
            header.software.version = version + chr(0) # Ensure null-terminated string
            self._metadata.set_header(header)

        def _get_repository(self) -> str:
            return self._metadata.get_header().software.repository
        
        def _set_repository(self, repository: str) -> None:
            header = self._metadata.get_header()
            header.software.repository = repository + chr(0) # Ensure null-terminated string
            self._metadata.set_header(header)

        def _get_commit(self) -> str:
            return self._metadata.get_header().software.commit
        
        def _set_commit(self, commit: str) -> None:
            header = self._metadata.get_header()
            header.software.commit = commit + chr(0) # Ensure null-terminated string
            self._metadata.set_header(header)

        def _get_doi(self) -> str:
            return self._metadata.get_header().software.doi
        
        def _set_doi(self, doi: str) -> None:
            header = self._metadata.get_header()
            header.software.doi = doi + chr(0) # Ensure null-terminated string
            self._metadata.set_header(header)

        name: str = property(_get_name, _set_name)
        version: str = property(_get_version, _set_version)
        repository: str = property(_get_repository, _set_repository)
        commit: str = property(_get_commit, _set_commit)
        doi: str = property(_get_doi, _set_doi)

        def __init__(self, name: str, version: str, repository: str, commit: str, doi: str, metadata: RadiationFieldMetadataV1) -> None:
            self._metadata = metadata
            if metadata is not None:
                self.name = name
                self.version = version
                self.repository = repository
                self.commit = commit
                self.doi = doi


    class XRayTube(object):
        def __init__(self, radiation_direction, radiation_origin, max_energy_eV, tube_id, metadata: RadiationFieldMetadataV1):
            super().__init__()
            self._metadata = metadata
            self.dyn_metadata_keys = metadata.get_dynamic_metadata_keys() if metadata is not None else []
            self._field_shape = None
            if metadata is not None:
                self.radiation_direction = radiation_direction
                self.radiation_origin = radiation_origin
                self.max_energy_eV = max_energy_eV
                self.tube_id = tube_id

        def _get_radiation_direction(self) -> vec3:
            return self._metadata.get_header().simulation.tube.radiation_direction
        
        def _set_radiation_direction(self, direction: vec3) -> None:
            header = self._metadata.get_header()
            header.simulation.tube.radiation_direction = direction
            self._metadata.set_header(header)

        def _get_radiation_origin(self) -> vec3:
            return self._metadata.get_header().simulation.tube.radiation_origin
        
        def _set_radiation_origin(self, origin: vec3) -> None:
            header = self._metadata.get_header()
            header.simulation.tube.radiation_origin = origin
            self._metadata.set_header(header)

        def _get_max_energy_eV(self) -> float:
            return self._metadata.get_header().simulation.tube.max_energy_eV
        
        def _set_max_energy_eV(self, max_energy_eV: float) -> None:
            header = self._metadata.get_header()
            header.simulation.tube.max_energy_eV = max_energy_eV
            self._metadata.set_header(header)

        def _get_tube_id(self) -> str:
            return self._metadata.get_header().simulation.tube.tube_id
        
        def _set_tube_id(self, tube_id: str) -> None:
            header = self._metadata.get_header()
            header.simulation.tube.tube_id = tube_id + chr(0) # Ensure null-terminated string
            self._metadata.set_header(header)

        radiation_direction: vec3 = property(_get_radiation_direction, _set_radiation_direction)
        radiation_origin: vec3 = property(_get_radiation_origin, _set_radiation_origin)
        max_energy_eV: float = property(_get_max_energy_eV, _set_max_energy_eV)
        tube_id: str = property(_get_tube_id, _set_tube_id)

        def _get_spectrum(self) -> np.ndarray:
            if "tube_spectrum" not in self.dyn_metadata_keys:
                return None
            tube_spectrum_data: HistogramVoxel = self._metadata.get_dynamic_metadata("tube_spectrum")
            tube_spectrum = np.zeros((tube_spectrum_data.get_bins(), 2), dtype=np.float32)
            tube_spectrum[:, 0] = np.arange(0, tube_spectrum_data.get_bins() * tube_spectrum_data.get_histogram_bin_width(), tube_spectrum_data.get_histogram_bin_width(), dtype=np.float32)
            tube_spectrum[:, 1] = tube_spectrum_data.get_histogram()
            tube_spectrum[:, 1] = np.where(~np.isnan(tube_spectrum[:, 1]), tube_spectrum[:, 1], 0.0)
            sum = tube_spectrum[:, 1].sum()
            if sum > 0.0 and not np.isclose(sum, 1.0):
                tube_spectrum[:, 1] = tube_spectrum[:, 1] / sum
            return tube_spectrum
        
        def _set_spectrum(self, spectrum: np.ndarray) -> None:
            assert isinstance(spectrum, np.ndarray), f"Expected numpy.ndarray, got {type(spectrum)}"
            assert spectrum.ndim == 2 and spectrum.shape[1] == 2, f"Spectrum must be a 2D array with shape (N, 2), got {spectrum.shape}"
            assert np.all(spectrum[:, 1] >= 0), "Spectrum values must be non-negative"
            bins = spectrum.shape[0]
            bin_width = spectrum[1, 0] - spectrum[0, 0] if bins > 1 else self.max_energy_eV
            if "tube_spectrum" not in self.dyn_metadata_keys:
                self.dyn_metadata_keys.append("tube_spectrum")
                vx = self._metadata.add_dynamic_histogram_metadata("tube_spectrum", bins, bin_width)
            else:
                vx = self._metadata.get_dynamic_metadata("tube_spectrum")
                assert isinstance(vx, HistogramVoxel), f"Expected HistogramVoxel, got {type(vx)}"
                assert vx.get_bins() == bins, f"Bin count mismatch: {vx.get_bins()} != {bins}"
                assert np.isclose(vx.get_histogram_bin_width(), bin_width), f"Bin width mismatch: {vx.get_histogram_bin_width()} != {bin_width}"
            vx.get_histogram()[:] = spectrum[:, 1]

        spectrum: np.ndarray = property(_get_spectrum, _set_spectrum)

        def _get_field_shape(self) -> FieldShape:
            if not "xray_field_shape" in self.dyn_metadata_keys or self._field_shape is not None:
                return self._field_shape
            vx = self._metadata.get_dynamic_metadata("xray_field_shape")
            self._field_shape = FieldShape(vx.get_data())
            return self._field_shape
        
        def _set_field_shape(self, shape: FieldShape) -> None:
            assert isinstance(shape, FieldShape), f"Expected FieldShape, got {type(shape)}"
            if "xray_field_shape" not in self.dyn_metadata_keys:
                self.dyn_metadata_keys.append("xray_field_shape")
                vx = self._metadata.add_dynamic_metadata("xray_field_shape", DType.BYTE)
            else:
                vx = self._metadata.get_dynamic_metadata("xray_field_shape")
                assert isinstance(vx, Voxel), f"Expected Voxel, got {type(vx)}"
            vx.set_data(shape.value)
            self._field_shape = shape

        field_shape: FieldShape = property(_get_field_shape, _set_field_shape)
        
        def _get_opening_angle_deg(self) -> float:
            if self.field_shape != FieldShape.CONE or "xray_tube_opening_angle_deg" not in self.dyn_metadata_keys:
                return None
            vx = self._metadata.get_dynamic_metadata("xray_tube_opening_angle_deg")
            return vx.get_data()
        
        def _set_opening_angle_deg(self, angle: float) -> None:
            assert isinstance(angle, (float, int)), f"Expected float, got {type(angle)}"
            assert angle > 0 and angle < 360, "Opening angle must be between 0 and 360 degrees"
            if self.field_shape != FieldShape.CONE:
                raise ValueError("Field shape must be CONE to set opening angle")
            if "xray_tube_opening_angle_deg" not in self.dyn_metadata_keys:
                self.dyn_metadata_keys.append("xray_tube_opening_angle_deg")
                vx = self._metadata.add_dynamic_metadata("xray_tube_opening_angle_deg", DType.FLOAT32)
            else:
                vx = self._metadata.get_dynamic_metadata("xray_tube_opening_angle_deg")
            vx.set_data(np.float32(angle))

        opening_angle_deg: float = property(_get_opening_angle_deg, _set_opening_angle_deg)

        def _get_field_rect_dimensions_m(self) -> vec2:
            if self.field_shape != FieldShape.RECTANGLE or "xray_tube_field_rect_dimensions_m" not in self.dyn_metadata_keys:
                return None
            vx = self._metadata.get_dynamic_metadata("xray_tube_field_rect_dimensions_m")
            return vx.get_data()

        def _set_field_rect_dimensions_m(self, dimensions: vec2) -> None:
            assert isinstance(dimensions, vec2), f"Expected vec2, got {type(dimensions)}"
            assert dimensions.x > 0 and dimensions.y > 0, "Field rectangle dimensions must be positive"
            if self.field_shape != FieldShape.RECTANGLE:
                raise ValueError("Field shape must be RECTANGLE to set field rectangle dimensions")
            if "xray_tube_field_rect_dimensions_m" not in self.dyn_metadata_keys:
                self.dyn_metadata_keys.append("xray_tube_field_rect_dimensions_m")
                vx = self._metadata.add_dynamic_metadata("xray_tube_field_rect_dimensions_m", DType.VEC2)
            else:
                vx = self._metadata.get_dynamic_metadata("xray_tube_field_rect_dimensions_m")
            vx.set_data(dimensions)
        
        field_rect_dimensions_m: vec2 = property(_get_field_rect_dimensions_m, _set_field_rect_dimensions_m)
        
        def _get_field_ellipsis_opening_angles_deg(self) -> vec2:
            if self.field_shape != FieldShape.ELLIPSIS or "xray_tube_field_ellipsis_opening_angles_deg" not in self.dyn_metadata_keys:
                return None
            vx = self._metadata.get_dynamic_metadata("xray_tube_field_ellipsis_opening_angles_deg")
            return vx.get_data()

        def _set_field_ellipsis_opening_angles_deg(self, angles: vec2) -> None:
            assert isinstance(angles, vec2), f"Expected vec2, got {type(angles)}"
            assert angles.x > 0 and angles.x < 360, "Field ellipse opening angle x must be between 0 and 360 degrees"
            assert angles.y > 0 and angles.y < 360, "Field ellipse opening angle y must be between 0 and 360 degrees"
            if self.field_shape != FieldShape.ELLIPSIS:
                raise ValueError("Field shape must be ELLIPSIS to set field ellipse opening angles")
            if "xray_tube_field_ellipsis_opening_angles_deg" not in self.dyn_metadata_keys:
                self.dyn_metadata_keys.append("xray_tube_field_ellipsis_opening_angles_deg")
                vx = self._metadata.add_dynamic_metadata("xray_tube_field_ellipsis_opening_angles_deg", DType.VEC2)
            else:
                vx = self._metadata.get_dynamic_metadata("xray_tube_field_ellipsis_opening_angles_deg")
            vx.set_data(angles)
        
        field_ellipsis_opening_angles_deg: vec2 = property(_get_field_ellipsis_opening_angles_deg, _set_field_ellipsis_opening_angles_deg)


    class Simulation(object):
        def _get_primary_particle_count(self) -> int:
            return self._metadata.get_header().simulation.primary_particle_count
        
        def _set_primary_particle_count(self, count: int) -> None:
            header = self._metadata.get_header()
            header.simulation.primary_particle_count = count
            self._metadata.set_header(header)

        def _get_geometry(self) -> str:
            return self._metadata.get_header().simulation.geometry
        
        def _set_geometry(self, geometry: str) -> None:
            header = self._metadata.get_header()
            header.simulation.geometry = geometry + chr(0) # Ensure null-terminated string
            self._metadata.set_header(header)

        def _get_physics_list(self) -> str:
            return self._metadata.get_header().simulation.physics_list
        
        def _set_physics_list(self, physics_list: str) -> None:
            header = self._metadata.get_header()
            header.simulation.physics_list = physics_list + chr(0) # Ensure null-terminated string
            self._metadata.set_header(header)

        def _get_simulation_duration_s(self) -> int:
            vx = self._metadata.get_dynamic_metadata("simulation_duration_s")
            return vx.get_data()

        def _set_simulation_duration_s(self, duration_s: int) -> None:
            assert isinstance(duration_s, (float, int)), f"Expected int, got {type(duration_s)}"
            duration_s = int(duration_s)
            if "simulation_duration_s" not in self._metadata.get_dynamic_metadata_keys():
                vx = self._metadata.add_dynamic_metadata("simulation_duration_s", DType.UINT64)
            else:
                vx = self._metadata.get_dynamic_metadata("simulation_duration_s")
            vx.set_data(duration_s)

        primary_particle_count: int = property(_get_primary_particle_count, _set_primary_particle_count)
        geometry: str = property(_get_geometry, _set_geometry)
        physics_list: str = property(_get_physics_list, _set_physics_list)
        simulation_duration_s: int = property(_get_simulation_duration_s, _set_simulation_duration_s)
        tube: "Metadata.XRayTube"

        def __init__(self, geometry: str, primary_particle_count: int, physics_list: str, tube: "Metadata.XRayTube", metadata: RadiationFieldMetadataV1) -> None:
            self._metadata = metadata
            if metadata is not None:
                self.geometry = geometry
                self.primary_particle_count = primary_particle_count
                self.physics_list = physics_list
            self.tube = tube

    simulation: "Metadata.Simulation"
    software: "Metadata.Software"

    def __init__(self, simulation: "Metadata.Simulation", software: "Metadata.Software", metadata: RadiationFieldMetadataV1 = None) -> None:
        super().__init__()
        self.raw_metadata = metadata
        if self.raw_metadata is None:
            self.raw_metadata = RadiationFieldMetadataV1(
                simulation=RadiationFieldSimulationMetadataV1(
                    geometry="",
                    primary_particle_count=0,
                    physics_list="",
                    tube=RadiationFieldXRayTubeMetadataV1(
                        radiation_direction=vec3(0, 0, 0),
                        radiation_origin=vec3(0, 0, 0),
                        max_energy_eV=0,
                        tube_id=""
                    )
                ),
                software=RadiationFieldSoftwareMetadataV1(
                    name="",
                    version="",
                    repository="",
                    commit=""
                )
            )

        self.simulation = simulation
        if simulation is not None:
            self.simulation.metadata = self.raw_metadata
            self.simulation.tube.metadata = self.raw_metadata
        self.software = software
        if software is not None:
            self.software.metadata = self.raw_metadata

    @staticmethod
    def default():
        """
        Returns a default metadata object for storing RadiationField files.
        Metadata contains the current main script name and default values in each field.
        Users may modify the metadata to describe the simulation.
        """

        sw_name = os.path.basename(__main__.__file__) if hasattr(__main__, "__file__") else "interactive"
        md = RadiationFieldMetadataV1(
            simulation=RadiationFieldSimulationMetadataV1(
                geometry="",
                primary_particle_count=0,
                physics_list="",
                tube=RadiationFieldXRayTubeMetadataV1(
                    radiation_direction=vec3(0, 0, 0),
                    radiation_origin=vec3(0, 0, 0),
                    max_energy_eV=0,
                    tube_id=""
                )
            ),
            software=RadiationFieldSoftwareMetadataV1(
                name=sw_name,
                version="",
                repository="",
                commit=""
            )
        )
        return Metadata(
            simulation=Metadata.Simulation(
                geometry=md.get_header().simulation.geometry,
                primary_particle_count=md.get_header().simulation.primary_particle_count,
                physics_list=md.get_header().simulation.physics_list,
                tube=Metadata.XRayTube(
                    md.get_header().simulation.tube.radiation_direction,
                    md.get_header().simulation.tube.radiation_origin,
                    md.get_header().simulation.tube.max_energy_eV,
                    md.get_header().simulation.tube.tube_id,
                    metadata=md
                ),
                metadata=md
            ),
            software=Metadata.Software(
                name=md.get_header().software.name,
                version=md.get_header().software.version,
                repository=md.get_header().software.repository,
                commit=md.get_header().software.commit,
                doi=md.get_header().software.doi,
                metadata=md
            ),
            metadata=md
        )

    def set_from_raw_metadata(self, metadata: RadiationFieldMetadataV1) -> None:
        """
        Set the Metadata object from a raw RadiationFieldMetadataV1 object.

        :param metadata: The RadiationFieldMetadataV1 object to set the Metadata object from.
        """
        assert isinstance(metadata, RadiationFieldMetadataV1), f"Expected RadiationFieldMetadataV1, got {type(metadata)}"
        raw_header = metadata.get_header()
        self.simulation = Metadata.Simulation(
            geometry=raw_header.simulation.geometry,
            primary_particle_count=raw_header.simulation.primary_particle_count,
            physics_list=raw_header.simulation.physics_list,
            tube=Metadata.XRayTube(
                radiation_direction=raw_header.simulation.tube.radiation_direction,
                radiation_origin=raw_header.simulation.tube.radiation_origin,
                max_energy_eV=raw_header.simulation.tube.max_energy_eV,
                tube_id=raw_header.simulation.tube.tube_id,
                metadata=metadata
            ),
            metadata=metadata
        )
        self.software = Metadata.Software(
            name=raw_header.software.name,
            version=raw_header.software.version,
            repository=raw_header.software.repository,
            commit=raw_header.software.commit,
            doi=raw_header.software.doi,
            metadata=metadata
        )
        self.raw_metadata = metadata

    @staticmethod
    def from_raw_metadata(metadata: RadiationFieldMetadataV1) -> "Metadata":
        """
        Convert a raw RadiationFieldMetadataV1 object to a Metadata object.

        :param metadata: The RadiationFieldMetadataV1 object to convert.
        """
        if isinstance(metadata, Metadata):
            return metadata
        md = Metadata(
            simulation=None,
            software=None,
            metadata=metadata
        )
        md.set_from_raw_metadata(metadata)
        return md

    def as_raw_metadata(self) -> RadiationFieldMetadataV1:
        """
        Convert the Metadata object to a raw RadiationFieldMetadataV1 object.
        """
        metadata = RadiationFieldMetadataV1(
            simulation=RadiationFieldSimulationMetadataV1(
                geometry=self.simulation.geometry,
                primary_particle_count=self.simulation.primary_particle_count,
                physics_list=self.simulation.physics_list,
                tube=RadiationFieldXRayTubeMetadataV1(
                    radiation_direction=self.simulation.tube.radiation_direction,
                    radiation_origin=self.simulation.tube.radiation_origin,
                    max_energy_eV=self.simulation.tube.max_energy_eV,
                    tube_id=self.simulation.tube.tube_id
                )
            ),
            software=RadiationFieldSoftwareMetadataV1(
                name=self.software.name,
                version=self.software.version,
                repository=self.software.repository,
                commit=self.software.commit,
                doi=self.software.doi
            )
        ) if self.raw_metadata is None else self.raw_metadata

        if self.raw_metadata is None:
            spectrum = self.simulation.tube.spectrum
            if spectrum is not None:
                bins = spectrum.shape[0]
                bin_width = spectrum[1, 0] - spectrum[0, 0] if bins > 1 else self.simulation.tube.max_energy_eV
                vx = metadata.add_dynamic_histogram_metadata("tube_spectrum", bins, bin_width)
                vx.get_histogram()[:] = spectrum[:, 1]
            
            field_shape = self.simulation.tube.field_shape
            if field_shape is not None:
                vx = metadata.add_dynamic_metadata("xray_field_shape", DType.BYTE)
                vx.set_data(np.uint8(field_shape.value))

            if field_shape == FieldShape.CONE:
                opening_angle_deg = self.simulation.tube.opening_angle_deg
                if opening_angle_deg is not None:
                    vx = metadata.add_dynamic_metadata("xray_tube_opening_angle_deg", DType.FLOAT32)
                    vx.set_data(np.float32(opening_angle_deg))
            elif field_shape == FieldShape.RECTANGLE:
                field_rect_dimensions_m = self.simulation.tube.field_rect_dimensions_m
                if field_rect_dimensions_m is not None:
                    vx: Voxel = metadata.add_dynamic_metadata("xray_tube_field_rect_dimensions_m", DType.VEC2)
                    vx.set_data(field_rect_dimensions_m)
            elif field_shape == FieldShape.ELLIPSIS:
                field_ellipsis_opening_angles_deg = self.simulation.tube.field_ellipsis_opening_angles_deg
                if field_ellipsis_opening_angles_deg is not None:
                    vx: Voxel = metadata.add_dynamic_metadata("xray_tube_field_ellipsis_opening_angles_deg", DType.VEC2)
                    vx.set_data(field_ellipsis_opening_angles_deg)
            else:
                raise ValueError("Invalid field shape")
            self.raw_metadata = metadata
        return metadata

    def get_header(self) -> RadiationFieldMetadataHeaderV1:
        """
        Returns the mandatory header of the metadata.
        The header contains the simulation and software metadata.
        
        :return: The header of the metadata.
        """
        return self.as_raw_metadata().get_header()

    def set_header(self, header: RadiationFieldMetadataHeaderV1) -> None:
        """
        Sets the mandatory header of the metadata.
        The header contains the simulation and software metadata.
        
        :param header: The header to set.
        """
        self.raw_metadata.set_header(header)
        self.set_from_raw_metadata(self.raw_metadata)


    def get_dynamic_metadata_keys(self) -> list[str]:
        """
        Returns a list of all dynamic metadata keys.
        Dynamic metadata keys are used to store additional information about the radiation field.
        The keys are used to store additional information about the radiation field.
        The values of the keys are Voxel objects that can be used to store any data type.

        :return: A list of all dynamic metadata keys.
        """
        return self.raw_metadata.get_dynamic_metadata_keys()

    def get_dynamic_metadata(self, key: str) -> Voxel:
        """
        Returns the dynamic metadata for a given key.
        The key is used to store additional information about the radiation field.
        The value of the key is a Voxel that can be used to store any data type.

        :param key: The key of the metadata.
        :return: The Voxel that can be used to store the value.
        """
        return self.raw_metadata.get_dynamic_metadata(key)

    def add_dynamic_metadata(self, key: str, dtype: DType) -> Voxel:
        """
        Adds a dynamic metadata key to the radiation field metadata.
        The key is used to store additional information about the radiation field.
        The value of the key is a Voxel that can be used to store any data type.

        :param key: The key to add.
        :param dtype: The data type of the value.
        :return: The Voxel that can be used to store the value.
        """
        return self.raw_metadata.add_dynamic_metadata(key, dtype)

    def add_dynamic_histogram_metadata(self, key: str, bins: int, bin_width: float) -> HistogramVoxel:
        """
        Adds a dynamic histogram metadata to the radiation field metadata.

        :param key: The key of the metadata.
        :param bins: The number of bins in the histogram.
        :param bin_width: The width of each bin in the histogram.
        :return: The histogram voxel representing the dynamic metadata.
        """
        return self.raw_metadata.add_dynamic_histogram_metadata(key, bins, bin_width)
