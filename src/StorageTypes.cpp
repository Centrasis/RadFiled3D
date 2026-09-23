#include "RadFiled3D/storage/Types.hpp"
#include "RadFiled3D/storage/FieldSerializer.hpp"


using namespace RadFiled3D;
using namespace RadFiled3D::Storage;

RadFiled3D::Storage::V1::RadiationFieldMetadata::RadiationFieldMetadata(const FiledTypes::V1::RadiationFieldMetadataHeader::Simulation& simulation, const FiledTypes::V1::RadiationFieldMetadataHeader::Software& software)
	: header(FiledTypes::V1::RadiationFieldMetadataHeader(simulation, software)),
	  dynamic_metadata(std::make_shared<VoxelBuffer>(1)),
	  RadFiled3D::Storage::RadiationFieldMetadata(StoreVersion::V1),
	  serializer(new V1::BinaryFieldBlockHandler())
{};

RadFiled3D::Storage::V1::RadiationFieldMetadata::RadiationFieldMetadata()
	: header(
		FiledTypes::V1::RadiationFieldMetadataHeader(
			FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				0,
				"",
				"",
				FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube()
			),
			FiledTypes::V1::RadiationFieldMetadataHeader::Software(
				"Unknown",
				"0.0",
				"",
				""
			)
		)
	),
	dynamic_metadata(std::make_shared<VoxelBuffer>(1)),
	RadFiled3D::Storage::RadiationFieldMetadata(StoreVersion::V1),
	serializer(new V1::BinaryFieldBlockHandler())
{};

RadFiled3D::Storage::V1::RadiationFieldMetadata::~RadiationFieldMetadata()
{
	delete this->serializer;
}

void RadFiled3D::Storage::V1::RadiationFieldMetadata::serialize(std::ostream& stream) const
{
	FiledTypes::V1::RadiationFieldMetadataHeaderBlock mHeader;
	if (this->dynamic_metadata->get_layers().size() > 0) {
		auto oss = this->serializer->serializeChannel(this->dynamic_metadata);
		mHeader.dynamic_metadata_size = oss->str().length();
		stream.write((const char*)&mHeader, sizeof(FiledTypes::V1::RadiationFieldMetadataHeaderBlock));
		stream.write((const char*)&this->header, sizeof(FiledTypes::V1::RadiationFieldMetadataHeader));
		stream.write(oss->str().c_str(), mHeader.dynamic_metadata_size);
	}
	else {
		mHeader.dynamic_metadata_size = 0;
		stream.write((const char*)&mHeader, sizeof(FiledTypes::V1::RadiationFieldMetadataHeaderBlock));
		stream.write((const char*)&this->header, sizeof(FiledTypes::V1::RadiationFieldMetadataHeader));
	}
}

void RadFiled3D::Storage::V1::RadiationFieldMetadata::deserialize(std::istream& stream, bool quick_peek_only)
{
	FiledTypes::V1::RadiationFieldMetadataHeaderBlock mHeader;
	stream.read((char*)&mHeader, sizeof(FiledTypes::V1::RadiationFieldMetadataHeaderBlock));
	stream.read((char*)&this->header, sizeof(FiledTypes::V1::RadiationFieldMetadataHeader));
	if (mHeader.dynamic_metadata_size > 0 && !quick_peek_only) {
		char* metadata_data = new char[mHeader.dynamic_metadata_size];
		stream.read(metadata_data, mHeader.dynamic_metadata_size);
		this->dynamic_metadata = this->serializer->deserializeChannel(this->dynamic_metadata, metadata_data, mHeader.dynamic_metadata_size);
		delete[] metadata_data;
	}
}

size_t RadFiled3D::Storage::V1::RadiationFieldMetadata::get_metadata_size(std::istream& stream) const
{
	FiledTypes::V1::RadiationFieldMetadataHeaderBlock mHeader;

	size_t current_pos = stream.tellg();

	stream.seekg(sizeof(FiledTypes::VersionHeader), std::ios::beg);
	stream.read((char*)&mHeader, sizeof(FiledTypes::V1::RadiationFieldMetadataHeaderBlock));
	size_t metadata_size = mHeader.dynamic_metadata_size + sizeof(FiledTypes::V1::RadiationFieldMetadataHeaderBlock) + sizeof(FiledTypes::V1::RadiationFieldMetadataHeader);

	stream.seekg(current_pos, std::ios::beg);

	return metadata_size;
}