#include "radfiled3d/storage/metadata_serializer.hpp"
#include <ostream>

void radfiled3d::storage::v1::MetadataSerializer::serializeMetadata(std::ostream& buffer, std::shared_ptr<radfiled3d::storage::RadiationFieldMetadata> metadata) const
{
	if (metadata->get_version() != StoreVersion::V1)
		throw std::runtime_error("Metadata version mismatch");

	metadata->serialize(buffer);
}