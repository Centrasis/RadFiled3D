#include "radfiled3d/storage/metadata_accessor.hpp"
#include "radfiled3d/storage/types.hpp"
#include <istream>

using namespace radfiled3d;
using namespace radfiled3d::storage;
using namespace radfiled3d::storage::filed_types;

std::shared_ptr<radfiled3d::storage::RadiationFieldMetadata> radfiled3d::storage::v1::MetadataAccessor::accessMetadata(std::istream& buffer, bool quick_peek_only) const
{
	buffer.seekg(sizeof(VersionHeader), std::ios::beg);

	auto metadata = std::make_shared<radfiled3d::storage::v1::RadiationFieldMetadata>();
	metadata->deserialize(buffer, quick_peek_only);

	return metadata;
}

size_t radfiled3d::storage::v1::MetadataAccessor::get_metadata_size(std::istream& stream) const
{
	return this->meta_template.get_metadata_size(stream);
}