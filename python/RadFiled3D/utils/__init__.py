from typing import Union
from RadFiled3D.RadFiled3D import FieldStore as FS, StoreVersion, FieldJoinMode, RadiationFieldMetadata, FieldAccessor as FA, RadiationField, FieldJoinCheckMode
from RadFiled3D.metadata.v1 import Metadata as MetadataV1


class FieldStore(FS):
    @staticmethod
    def load_metadata(file: str) -> Union[RadiationFieldMetadata, MetadataV1]:
        """
        Get the metadata of a stored radiation field.

        :param file: The file path to the stored radiation field.
        """
        metadata = FS.load_metadata(file)
        if FS.get_store_version(file) == StoreVersion.V1:
            metadata = MetadataV1.from_raw_metadata(metadata)
        return metadata
    
    @staticmethod
    def load_metadata_v1(file: str) -> MetadataV1:
        """
        Get the metadata of a stored radiation field in version 1 format.

        :param file: The file path to the stored radiation field.
        """
        metadata = FS.load_metadata(file)
        return MetadataV1.from_raw_metadata(metadata)

    @staticmethod
    def load_metadata_from_buffer_v1(buffer: bytes) -> MetadataV1:
        """
        Get the metadata of a stored radiation field from a buffer.

        :param buffer: The buffer to load the metadata from.
        """
        metadata = FS.load_metadata_from_buffer(buffer)
        return MetadataV1.from_raw_metadata(metadata)

    @staticmethod
    def store(field: RadiationField, metadata: Union[RadiationFieldMetadata, MetadataV1], file: str, version: StoreVersion = StoreVersion.V1) -> None:
        """
        Store a radiation field to a file.

        :param field: The radiation field to store.
        :param metadata: The metadata of the radiation field.
        :param file: The file path to store the radiation field to.
        :param version: The version to store the radiation field with.
        """
        assert isinstance(metadata, RadiationFieldMetadata) or (isinstance(metadata, MetadataV1) and version == StoreVersion.V1), "Metadata must be of type RadiationFieldMetadata or MetadataV1 when using version V1"
        if isinstance(metadata, MetadataV1):
            metadata: RadiationFieldMetadata = metadata.as_raw_metadata()
        FS.store(field, metadata, file, version)

    @staticmethod
    def join(field: RadiationField, metadata: Union[RadiationFieldMetadata, MetadataV1], file: str, join_mode: FieldJoinMode, check_mode: FieldJoinCheckMode, fallback_version: StoreVersion = StoreVersion.V1) -> None:
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
        assert isinstance(metadata, RadiationFieldMetadata) or (isinstance(metadata, MetadataV1) and fallback_version == StoreVersion.V1), "Metadata must be of type RadiationFieldMetadata or MetadataV1 when using version V1"
        if isinstance(metadata, MetadataV1):
            metadata: RadiationFieldMetadata = metadata.as_raw_metadata()
        FS.join(field, metadata, file, join_mode, check_mode, fallback_version)
