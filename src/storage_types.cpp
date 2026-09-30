#include "radfiled3d/storage/types.hpp"
#include "radfiled3d/storage/field_serializer.hpp"


using namespace radfiled3d;
using namespace radfiled3d::storage;

radfiled3d::storage::v1::RadiationFieldMetadata::RadiationFieldMetadata(const filed_types::v1::RadiationFieldMetadataHeader::Simulation& simulation, const filed_types::v1::RadiationFieldMetadataHeader::Software& software)
	: header(filed_types::v1::RadiationFieldMetadataHeader(simulation, software)),
	  dynamic_metadata(std::make_shared<VoxelBuffer>(1)),
	  radfiled3d::storage::RadiationFieldMetadata(StoreVersion::V1),
	  serializer(new v1::BinaryFieldBlockHandler())
{};

radfiled3d::storage::v1::RadiationFieldMetadata::RadiationFieldMetadata()
	: header(
		filed_types::v1::RadiationFieldMetadataHeader(
			filed_types::v1::RadiationFieldMetadataHeader::Simulation(
				0,
				"",
				"",
				filed_types::v1::RadiationFieldMetadataHeader::Simulation::XRayTube()
			),
			filed_types::v1::RadiationFieldMetadataHeader::Software(
				"Unknown",
				"0.0",
				"",
				""
			)
		)
	),
	dynamic_metadata(std::make_shared<VoxelBuffer>(1)),
	radfiled3d::storage::RadiationFieldMetadata(StoreVersion::V1),
	serializer(new v1::BinaryFieldBlockHandler())
{};

radfiled3d::storage::v1::RadiationFieldMetadata::~RadiationFieldMetadata()
{
	delete this->serializer;
}

void radfiled3d::storage::v1::RadiationFieldMetadata::serialize(std::ostream& stream) const
{
	filed_types::v1::RadiationFieldMetadataHeaderBlock mHeader;
	if (this->dynamic_metadata->get_layers().size() > 0) {
		auto oss = this->serializer->serializeChannel(this->dynamic_metadata);
		mHeader.dynamic_metadata_size = oss->str().length();
		stream.write((const char*)&mHeader, sizeof(filed_types::v1::RadiationFieldMetadataHeaderBlock));
		stream.write((const char*)&this->header, sizeof(filed_types::v1::RadiationFieldMetadataHeader));
		stream.write(oss->str().c_str(), mHeader.dynamic_metadata_size);
	}
	else {
		mHeader.dynamic_metadata_size = 0;
		stream.write((const char*)&mHeader, sizeof(filed_types::v1::RadiationFieldMetadataHeaderBlock));
		stream.write((const char*)&this->header, sizeof(filed_types::v1::RadiationFieldMetadataHeader));
	}
}

void radfiled3d::storage::v1::RadiationFieldMetadata::deserialize(std::istream& stream, bool quick_peek_only)
{
	filed_types::v1::RadiationFieldMetadataHeaderBlock mHeader;
	stream.read((char*)&mHeader, sizeof(filed_types::v1::RadiationFieldMetadataHeaderBlock));
	stream.read((char*)&this->header, sizeof(filed_types::v1::RadiationFieldMetadataHeader));
	if (mHeader.dynamic_metadata_size > 0 && !quick_peek_only) {
		char* metadata_data = new char[mHeader.dynamic_metadata_size];
		stream.read(metadata_data, mHeader.dynamic_metadata_size);
		this->dynamic_metadata = this->serializer->deserializeChannel(this->dynamic_metadata, metadata_data, mHeader.dynamic_metadata_size);
		delete[] metadata_data;
	}
}

size_t radfiled3d::storage::v1::RadiationFieldMetadata::get_metadata_size(std::istream& stream) const
{
	filed_types::v1::RadiationFieldMetadataHeaderBlock mHeader;

	size_t current_pos = stream.tellg();

	stream.seekg(sizeof(filed_types::VersionHeader), std::ios::beg);
	stream.read((char*)&mHeader, sizeof(filed_types::v1::RadiationFieldMetadataHeaderBlock));
	size_t metadata_size = mHeader.dynamic_metadata_size + sizeof(filed_types::v1::RadiationFieldMetadataHeaderBlock) + sizeof(filed_types::v1::RadiationFieldMetadataHeader);

	stream.seekg(current_pos, std::ios::beg);

	return metadata_size;
}