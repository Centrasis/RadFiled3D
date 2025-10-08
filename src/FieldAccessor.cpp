#include "RadFiled3D/storage/FieldAccessor.hpp"
#include "RadFiled3D/RadiationField.hpp"
#include "RadFiled3D/Voxel.hpp"
#include "RadFiled3D/VoxelGrid.hpp"
#include "RadFiled3D/PolarSegments.hpp"
#include "RadFiled3D/storage/FieldSerializer.hpp"
#include "RadFiled3D/storage/MetadataAccessor.hpp"
#include <istream>
#include <fstream>
#include <memory>


using namespace RadFiled3D;
using namespace RadFiled3D::Storage;
using namespace RadFiled3D::Storage::FiledTypes;


std::vector<char> RadFiled3D::Storage::V1::FileParser::SerializeChannelsLayersOffsets(const std::map<std::string, AccessorTypes::ChannelStructure>& channels_layers_offsets)
{
	std::vector<char> channel_layer_offsets_data;

	for (auto& channel : channels_layers_offsets) {
		channel_layer_offsets_data.insert(channel_layer_offsets_data.end(), channel.first.begin(), channel.first.end());
		channel_layer_offsets_data.push_back('\0');
		channel_layer_offsets_data.insert(channel_layer_offsets_data.end(), (char*)&channel.second.channel_block.offset, (char*)&channel.second.channel_block.offset + sizeof(size_t));
		channel_layer_offsets_data.insert(channel_layer_offsets_data.end(), (char*)&channel.second.channel_block.size, (char*)&channel.second.channel_block.size + sizeof(size_t));

		size_t layer_count = channel.second.layers.size();
		channel_layer_offsets_data.insert(channel_layer_offsets_data.end(), (char*)&layer_count, (char*)&layer_count + sizeof(size_t));

		for (auto& layer : channel.second.layers) {
			size_t voxel_header_data_size = layer.second.get_voxel_header_data_size();
			channel_layer_offsets_data.insert(channel_layer_offsets_data.end(), layer.first.begin(), layer.first.end());
			channel_layer_offsets_data.push_back('\0');
			channel_layer_offsets_data.insert(channel_layer_offsets_data.end(), (char*)&layer.second.offset, (char*)&layer.second.offset + sizeof(size_t));
			channel_layer_offsets_data.insert(channel_layer_offsets_data.end(), (char*)&layer.second.size, (char*)&layer.second.size + sizeof(size_t));
			channel_layer_offsets_data.insert(channel_layer_offsets_data.end(), (char*)&layer.second.dtype, (char*)&layer.second.dtype + sizeof(Typing::DType));
			channel_layer_offsets_data.insert(channel_layer_offsets_data.end(), (char*)&layer.second.elements_per_voxel, (char*)&layer.second.elements_per_voxel + sizeof(size_t));
			channel_layer_offsets_data.insert(channel_layer_offsets_data.end(), (char*)&voxel_header_data_size, (char*)&voxel_header_data_size + sizeof(size_t));
			if (voxel_header_data_size > 0)
				channel_layer_offsets_data.insert(channel_layer_offsets_data.end(), (char*)layer.second.get_voxel_header_data(), (char*)layer.second.get_voxel_header_data() + voxel_header_data_size);
		}
	}
	return channel_layer_offsets_data;
}

std::map<std::string, AccessorTypes::ChannelStructure> RadFiled3D::Storage::V1::FileParser::DeserializeChannelsLayersOffsets(const std::vector<char>& data) {
	std::map<std::string, AccessorTypes::ChannelStructure> channels_layers_offsets = std::map<std::string, AccessorTypes::ChannelStructure>();
	size_t offset = 0;
	const size_t min_bytes_per_channel = sizeof(size_t) * 3 + sizeof(char);
	const size_t min_remaining_size = (data.size() >= min_bytes_per_channel) ? data.size() - min_bytes_per_channel : 0;
	while (offset < min_remaining_size) {
		std::string channel_name = std::string(data.data() + offset);
		if (offset + channel_name.size() + 1 + sizeof(size_t) * 3 > min_remaining_size)
			break;
		offset += channel_name.size() + 1;
		size_t channel_offset = *(size_t*)(data.data() + offset);
		offset += sizeof(size_t);
		size_t channel_size = *(size_t*)(data.data() + offset);
		offset += sizeof(size_t);
		size_t layer_count = *(size_t*)(data.data() + offset);
		offset += sizeof(size_t);

		std::map<std::string, AccessorTypes::TypedMemoryBlockDefinition> layers;
		while (offset < data.size() && layers.size() < layer_count) {
			std::string layer_name = std::string(data.data() + offset);
			offset += layer_name.size() + 1;
			size_t layer_offset = *(size_t*)(data.data() + offset);
			offset += sizeof(size_t);
			size_t layer_size = *(size_t*)(data.data() + offset);
			offset += sizeof(size_t);
			Typing::DType dtype = *(Typing::DType*)(data.data() + offset);
			offset += sizeof(Typing::DType);
			size_t elements_per_voxel = *(size_t*)(data.data() + offset);
			offset += sizeof(size_t);
			size_t voxel_header_data_size = *(size_t*)(data.data() + offset);
			offset += sizeof(size_t);

			AccessorTypes::TypedMemoryBlockDefinition layer_block(layer_offset, layer_size, dtype, elements_per_voxel);
			if (voxel_header_data_size > 0)
				layer_block.set_voxel_header_data((char*)(data.data() + offset), voxel_header_data_size);
			offset += voxel_header_data_size;
			layers[layer_name] = layer_block;
		}

		AccessorTypes::ChannelStructure channel_block(
			AccessorTypes::MemoryBlockDefinition(channel_offset, channel_size),
			layers
		);
		channels_layers_offsets[channel_name] = channel_block;
	}
	return channels_layers_offsets;
}

std::vector<char> RadFiled3D::Storage::V1::CartesianFieldAccessor::SerializationData::serialize_additional_data() const {
	return FileParser::SerializeChannelsLayersOffsets(this->channels_layers_offsets);
}

void RadFiled3D::Storage::V1::CartesianFieldAccessor::SerializationData::deserialize_additional_data(const std::vector<char>& data) {
	this->channels_layers_offsets = FileParser::DeserializeChannelsLayersOffsets(data);
}

std::vector<char> RadFiled3D::Storage::FieldAccessor::Serialize(const FieldAccessor* accessor)
{
	FieldAccessor::SerializationData* sdata = accessor->generateSerializationBuffer();
	std::vector<char> additional_data = sdata->serialize_additional_data();
	std::vector<char> serialized_accessor;

	if (sdata->field_type == FieldType::Cartesian) {
		serialized_accessor.resize(sizeof(V1::CartesianFieldAccessor::SerializationData) + additional_data.size());
		memcpy((char*)serialized_accessor.data(), (void*)sdata, sizeof(V1::CartesianFieldAccessor::SerializationData));
	}
	else if (sdata->field_type == FieldType::Polar) {
		serialized_accessor.resize(sizeof(V1::PolarFieldAccessor::SerializationData) + additional_data.size());
		memcpy((char*)serialized_accessor.data(), (void*)sdata, sizeof(V1::PolarFieldAccessor::SerializationData));
	}
	else {
		throw std::runtime_error("Unsupported field type");
	}

	memcpy((char*)serialized_accessor.data() + (serialized_accessor.size() - additional_data.size()), additional_data.data(), additional_data.size());
	delete sdata;
	return serialized_accessor;
}

std::shared_ptr<FieldAccessor> RadFiled3D::Storage::FieldAccessor::Deserialize(const std::vector<char>& buffer)
{
	FieldAccessor::SerializationData sdata_header;
	memcpy((char*)&sdata_header, buffer.data(), sizeof(FieldAccessor::SerializationData));
	std::shared_ptr<FieldAccessor> accessor;
	if (sdata_header.store_version == StoreVersion::V1) {
		if (sdata_header.field_type == FieldType::Cartesian) {
			V1::CartesianFieldAccessor::SerializationData sdata;
			memcpy(((char*)&sdata), buffer.data(), sizeof(V1::CartesianFieldAccessor::SerializationData) - sizeof(std::map<std::string, AccessorTypes::ChannelStructure>));

			size_t remaining_size = buffer.size() - sizeof(V1::CartesianFieldAccessor::SerializationData);
			std::vector<char> additional_data(remaining_size);
			memcpy(additional_data.data(), buffer.data() + sizeof(V1::CartesianFieldAccessor::SerializationData), remaining_size);
			sdata.deserialize_additional_data(additional_data);
			return std::static_pointer_cast<FieldAccessor>(std::make_shared<V1::CartesianFieldAccessor>(sdata));
		}
		else if (sdata_header.field_type == FieldType::Polar) {
			V1::PolarFieldAccessor::SerializationData sdata;
			memcpy(((char*)&sdata), buffer.data(), sizeof(V1::PolarFieldAccessor::SerializationData) - sizeof(std::map<std::string, AccessorTypes::ChannelStructure>));

			size_t remaining_size = buffer.size() - sizeof(V1::PolarFieldAccessor::SerializationData);
			std::vector<char> additional_data(remaining_size);
			memcpy(additional_data.data(), buffer.data() + sizeof(V1::PolarFieldAccessor::SerializationData), remaining_size);
			sdata.deserialize_additional_data(additional_data);
			return std::static_pointer_cast<FieldAccessor>(std::make_shared<V1::PolarFieldAccessor>(sdata));
		}
		else {
			throw std::runtime_error("Unsupported field type");
		}
	}
	else {
		throw std::runtime_error("Unsupported store version");
	}

	throw std::runtime_error("Unsupported field type");
}


StoreVersion RadFiled3D::Storage::FieldAccessor::getStoreVersion(std::istream& buffer)
{
	VersionHeader version;
	buffer.seekg(0, std::ios::beg);
	buffer.read((char*)&version, sizeof(VersionHeader));

	if (strcmp(version.version, "1.0") == 0)
		return StoreVersion::V1;

	throw RadiationFieldStoreException(std::string("Unsupported file version: ") + std::string(version.version));
}

void RadFiled3D::Storage::FieldAccessor::verifyBuffer(std::istream& buffer) const
{
	/*buffer.seekg(this->metadata_fieldheader_verification_position.first, std::ios::beg);
	char verifyByte = 0;
	buffer.read(&verifyByte, 1);
	if (verifyByte != this->metadata_fieldheader_verification_position.second)
		throw RadiationFieldStoreException("Field header verification failed");
	*/

	buffer.seekg(this->metadata_fileheader_size, std::ios::beg);
}

RadFiled3D::Storage::FieldAccessor::FieldAccessor(FieldType field_type)
	: field_type(field_type), metadata_fileheader_size(0), store_version(StoreVersion::V1), voxel_count(0)
{
}

void RadFiled3D::Storage::V1::FileParser::initialize(std::istream& buffer)
{
	if (this->voxel_count == 0) {
		throw RadiationFieldStoreException("Invalid voxel count");
	}

	this->channels_layers_offsets.clear();
	buffer.seekg(this->getFieldDataOffset(), std::ios::beg);

	this->serializer = std::make_unique<BinayFieldBlockHandler>();

	size_t channel_pos = 0;
	const size_t start_pos = buffer.tellg();
	buffer.seekg(0, std::ios::end);
	const size_t max_bytes = buffer.tellg();
	buffer.seekg(start_pos);
	while (!buffer.eof() && channel_pos < max_bytes) {
		FiledTypes::V1::ChannelHeader channel_header;
		buffer.read((char*)&channel_header, sizeof(FiledTypes::V1::ChannelHeader));

		if (std::string(channel_header.name).empty()) {
			channel_pos += sizeof(FiledTypes::V1::ChannelHeader);
			buffer.seekg(this->getFieldDataOffset() + channel_pos, std::ios::beg);
			continue;
		}

		AccessorTypes::MemoryBlockDefinition channel_block(channel_pos, channel_header.channel_bytes);
		std::map<std::string, AccessorTypes::TypedMemoryBlockDefinition> layers_blocks;

		size_t layer_pos = 0;
		while (layer_pos + sizeof(FiledTypes::V1::VoxelGridLayerHeader) < channel_header.channel_bytes) {
			FiledTypes::V1::VoxelGridLayerHeader layer_header;
			buffer.seekg(this->getFieldDataOffset() + sizeof(FiledTypes::V1::ChannelHeader) + layer_pos + channel_pos, std::ios::beg);
			buffer.read((char*)&layer_header, sizeof(FiledTypes::V1::VoxelGridLayerHeader));

			const size_t layer_size = sizeof(FiledTypes::V1::VoxelGridLayerHeader) + this->voxel_count * layer_header.bytes_per_element + layer_header.header_block_size;
			Typing::DType dtype = Typing::Helper::get_dtype(std::string(layer_header.dtype));
			size_t elements_per_voxel = layer_header.bytes_per_element / Typing::Helper::get_bytes_of_dtype(dtype);
			layers_blocks[layer_header.name] = AccessorTypes::TypedMemoryBlockDefinition(layer_pos, layer_size, dtype, elements_per_voxel);

			if (layer_header.header_block_size > 0) {
				char* header_data = new char[layer_header.header_block_size];
				buffer.read(header_data, layer_header.header_block_size);
				layers_blocks[layer_header.name].set_voxel_header_data(header_data, layer_header.header_block_size);
				delete[] header_data;
			}

			layer_pos += layer_size;
		}
		this->channels_layers_offsets[channel_header.name] = AccessorTypes::ChannelStructure(channel_block, layers_blocks);
		channel_pos += channel_header.channel_bytes + sizeof(FiledTypes::V1::ChannelHeader);
		buffer.seekg(this->getFieldDataOffset() + channel_pos, std::ios::beg);
	}
}

IVoxel* RadFiled3D::Storage::V1::FileParser::createVoxelFromBuffer(char* data_buffer, Typing::DType dtype, const char* voxel_header_data) const
{
	IVoxel* voxel = nullptr;
	switch (dtype) {
	case Typing::DType::Float:
		voxel = new OwningScalarVoxel<float>((float*)data_buffer);
		break;
	case Typing::DType::Double:
#if defined(__x86_64__) || defined(_M_X64)
		voxel = new OwningScalarVoxel<double>((double*)data_buffer);
#else
		throw RadiationFieldStoreException("Can't load 64-bit file in 32-bit system!");
#endif
		break;
	case Typing::DType::Int:
		voxel = new OwningScalarVoxel<int>((int*)data_buffer);
		break;
	case Typing::DType::UInt32:
		voxel = new OwningScalarVoxel<uint32_t>((uint32_t*)data_buffer);
		break;
	case Typing::DType::UInt64:
#if defined(__x86_64__) || defined(_M_X64)
		voxel = new OwningScalarVoxel<uint64_t>((uint64_t*)data_buffer);
#else
		throw RadiationFieldStoreException("Can't load 64-bit file in 32-bit system!");
#endif
		break;
	case Typing::DType::Char:
		voxel = new OwningScalarVoxel<char>((char*)data_buffer);
		break;
	case Typing::DType::Byte:
		voxel = new OwningScalarVoxel<uint8_t>((uint8_t*)data_buffer);
		break;
	case Typing::DType::Vec2:
		voxel = new OwningScalarVoxel<glm::vec2>((glm::vec2*)data_buffer);
		break;
	case Typing::DType::Vec3:
		voxel = new OwningScalarVoxel<glm::vec3>((glm::vec3*)data_buffer);
		break;
	case Typing::DType::Vec4:
		voxel = new OwningScalarVoxel<glm::vec4>((glm::vec4*)data_buffer);
		break;
	case Typing::DType::Hist:
		break;
	}

	if (voxel == nullptr && dtype == Typing::DType::Hist) {
		OwningHistogramVoxel* vx = new OwningHistogramVoxel();
		if (voxel_header_data != nullptr) {
			vx->init_from_header(voxel_header_data);
		}
		vx->set_data(data_buffer);
		voxel = vx;
	}

	if (voxel == nullptr)
		throw RadiationFieldStoreException("Unsupported data type");
	return voxel;
}

IVoxel* RadFiled3D::Storage::V1::FileParser::accessVoxelRawFlat(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, size_t voxel_idx) const
{
	auto channel_block_itr = this->channels_layers_offsets.find(channel_name);
	if (channel_block_itr == this->channels_layers_offsets.end())
		throw RadiationFieldStoreException("Channel not found");

	auto layer_block_itr = channel_block_itr->second.layers.find(layer_name);
	if (layer_block_itr == channel_block_itr->second.layers.end())
		throw RadiationFieldStoreException("Layer not found");

	auto& channel_block = channel_block_itr->second.channel_block;
	auto& layer_block = layer_block_itr->second;

	const size_t element_size = Typing::Helper::get_bytes_of_dtype(layer_block.dtype);

	if (voxel_idx >= this->voxel_count)
		throw RadiationFieldStoreException("Voxel index out of bounds");

	// Add layer_block.get_voxel_header_data_size() to read position as the creation of the voxel does not revist the header data
	buffer.seekg(this->getFieldDataOffset() + channel_block.offset + layer_block.offset + layer_block.get_voxel_header_data_size() + sizeof(FiledTypes::V1::VoxelGridLayerHeader) + sizeof(FiledTypes::V1::ChannelHeader) + voxel_idx * layer_block.elements_per_voxel * element_size, std::ios::beg);
	char* data_buffer = new char[layer_block.elements_per_voxel * element_size];
	buffer.read(data_buffer, layer_block.elements_per_voxel * element_size);

	IVoxel* voxel = this->createVoxelFromBuffer(data_buffer, layer_block.dtype, (layer_block.get_voxel_header_data_size() > 0) ? layer_block.get_voxel_header_data() : nullptr);
	delete[] data_buffer;
	return voxel;
}

std::vector<IVoxel*> RadFiled3D::Storage::V1::FileParser::accessVoxelsRawFlat(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const std::vector<size_t>& voxel_indices) const
{
	auto voxels = std::vector<IVoxel*>(voxel_indices.size());

	auto channel_block_itr = this->channels_layers_offsets.find(channel_name);
	if (channel_block_itr == this->channels_layers_offsets.end())
		throw RadiationFieldStoreException("Channel not found");

	auto layer_block_itr = channel_block_itr->second.layers.find(layer_name);
	if (layer_block_itr == channel_block_itr->second.layers.end())
		throw RadiationFieldStoreException("Layer not found");

	auto& channel_block = channel_block_itr->second.channel_block;
	auto& layer_block = layer_block_itr->second;

	const size_t element_size = Typing::Helper::get_bytes_of_dtype(layer_block.dtype);

	size_t vx_parsed = 0;
	for (size_t voxel_idx : voxel_indices) {
		if (voxel_idx >= this->voxel_count)
			throw RadiationFieldStoreException("Voxel index out of bounds");

		// Add layer_block.get_voxel_header_data_size() to read position as the creation of the voxel does not revist the header data
		buffer.seekg(this->getFieldDataOffset() + channel_block.offset + layer_block.offset + layer_block.get_voxel_header_data_size() + sizeof(FiledTypes::V1::VoxelGridLayerHeader) + sizeof(FiledTypes::V1::ChannelHeader) + voxel_idx * layer_block.elements_per_voxel * element_size, std::ios::beg);
		char* data_buffer = new char[layer_block.elements_per_voxel * element_size];
		buffer.read(data_buffer, layer_block.elements_per_voxel * element_size);
		voxels[vx_parsed++] = this->createVoxelFromBuffer(data_buffer, layer_block.dtype, (layer_block.get_voxel_header_data_size() > 0) ? layer_block.get_voxel_header_data() : nullptr);
		delete[] data_buffer;
	}

	return voxels;
}

size_t RadFiled3D::Storage::V1::CartesianFieldAccessor::getFieldDataOffset() const
{
	return this->getMetadataFileheaderOffset() + sizeof(FiledTypes::V1::RadiationFieldHeader) + sizeof(FiledTypes::V1::CartesianHeader);
}

size_t RadFiled3D::Storage::V1::PolarFieldAccessor::getFieldDataOffset() const
{
	return this->getMetadataFileheaderOffset() + sizeof(FiledTypes::V1::RadiationFieldHeader) + sizeof(FiledTypes::V1::PolarHeader);
}

void RadFiled3D::Storage::V1::CartesianFieldAccessor::initialize(std::istream& buffer)
{
	this->metadata_fileheader_size = RadFiled3D::Storage::V1::MetadataAccessor().get_metadata_size(buffer) + sizeof(Storage::FiledTypes::VersionHeader);

	buffer.seekg(this->getMetadataFileheaderOffset() + sizeof(FiledTypes::V1::RadiationFieldHeader), std::ios::beg);
	FiledTypes::V1::CartesianHeader ch;
	buffer.read((char*)&ch, sizeof(FiledTypes::V1::CartesianHeader));

	this->field_dimensions = glm::vec3(ch.voxel_counts) * ch.voxel_dimensions;
	this->voxel_dimensions = ch.voxel_dimensions;
	this->voxel_count = ch.voxel_counts.x * ch.voxel_counts.y * ch.voxel_counts.z;
	this->default_grid = std::make_unique<VoxelGrid>(this->field_dimensions, ch.voxel_dimensions);

	FileParser::initialize(buffer);
}

RadFiled3D::Storage::V1::CartesianFieldAccessor::CartesianFieldAccessor()
	: RadFiled3D::Storage::FieldAccessor(FieldType::Cartesian),
	  RadFiled3D::Storage::CartesianFieldAccessor(FieldType::Cartesian),
	  RadFiled3D::Storage::V1::FileParser(FieldType::Cartesian)
{
	this->field_dimensions = glm::vec3(0);
	this->voxel_dimensions = glm::vec3(0.f);
	this->store_version = StoreVersion::V1;
}

RadFiled3D::Storage::V1::CartesianFieldAccessor::CartesianFieldAccessor(const SerializationData& data)
	: RadFiled3D::Storage::FieldAccessor(FieldType::Cartesian),
	  RadFiled3D::Storage::CartesianFieldAccessor(FieldType::Cartesian),
	  RadFiled3D::Storage::V1::FileParser(FieldType::Cartesian)
{
	if (data.store_version != StoreVersion::V1)
		throw RadiationFieldStoreException("Invalid store version");

	this->metadata_fileheader_size = data.metadata_fileheader_size;
	this->voxel_count = data.voxel_count;

	this->field_dimensions = data.field_dimensions;
	this->voxel_dimensions = data.voxel_dimensions;
	this->store_version = StoreVersion::V1;
	this->channels_layers_offsets = data.channels_layers_offsets;
	this->default_grid = std::make_unique<VoxelGrid>(this->field_dimensions, this->voxel_dimensions);
	this->serializer = std::make_unique<BinayFieldBlockHandler>();
}

std::shared_ptr<IRadiationField> RadFiled3D::Storage::V1::CartesianFieldAccessor::accessField(std::istream& buffer) const
{
	size_t metadata_size = this->getMetadataFileheaderOffset();
	buffer.seekg(metadata_size, std::ios::beg);
	return this->serializer->deserializeField(buffer);
}

std::shared_ptr<VoxelGridBuffer> RadFiled3D::Storage::V1::CartesianFieldAccessor::accessChannel(std::istream& buffer, const std::string& channel_name) const
{
	std::ofstream out("access_channel_accessor.log", std::ios::app);
	out << "From CartesianFieldAccessorV1" << std::endl;
	out << "Accessing channel '" << channel_name << "'" << std::endl;
	auto channel_block_itr = this->channels_layers_offsets.find(channel_name);
	if (channel_block_itr == this->channels_layers_offsets.end()) {
		out << "Channel not found" << std::endl;
		throw RadiationFieldStoreException("Channel not found");
	}

	auto& channel_block = channel_block_itr->second.channel_block;

	buffer.seekg(this->getFieldDataOffset() + channel_block.offset + sizeof(FiledTypes::V1::ChannelHeader), std::ios::beg);

	auto grid_buffer = std::make_shared<VoxelGridBuffer>(this->field_dimensions, this->voxel_dimensions);
	char* data_buffer = new char[channel_block.size];
	buffer.read(data_buffer, channel_block.size);
	this->serializer->deserializeChannel(grid_buffer, data_buffer, channel_block.size);
	delete[] data_buffer;

	out << "Channel voxels loaded: " << grid_buffer->get_voxel_count() << std::endl;

	return grid_buffer;
}

std::shared_ptr<VoxelGrid> RadFiled3D::Storage::V1::CartesianFieldAccessor::accessLayer(std::istream& buffer, const std::string& channel_name, const std::string& layer_name) const
{
	auto channel_block_itr = this->channels_layers_offsets.find(channel_name);
	if (channel_block_itr == this->channels_layers_offsets.end())
		throw RadiationFieldStoreException("Channel not found");

	auto layer_block_itr = channel_block_itr->second.layers.find(layer_name);
	if (layer_block_itr == channel_block_itr->second.layers.end())
		throw RadiationFieldStoreException("Layer not found");

	auto& channel_block = channel_block_itr->second.channel_block;
	auto& layer_block = layer_block_itr->second;

	buffer.seekg(this->getFieldDataOffset() + channel_block.offset + layer_block.offset + sizeof(FiledTypes::V1::ChannelHeader), std::ios::beg);

	char* data_buffer = new char[layer_block.size];
	buffer.read(data_buffer, layer_block.size);
	VoxelLayer* layer = this->serializer->deserializeLayer(data_buffer, layer_block.size);
	delete[] data_buffer;

	return std::make_shared<VoxelGrid>(this->field_dimensions, this->voxel_dimensions, std::shared_ptr<VoxelLayer>(layer));
}

std::map<std::string, std::shared_ptr<VoxelGrid>> RadFiled3D::Storage::V1::CartesianFieldAccessor::accessLayerAcrossChannels(std::istream& buffer, const std::string& layer_name) const
{
	std::map<std::string, std::shared_ptr<VoxelGrid>> layers = std::map<std::string, std::shared_ptr<VoxelGrid>>();

	for (auto& channel : this->channels_layers_offsets) {
		auto layer_block_itr = channel.second.layers.find(layer_name);
		if (layer_block_itr == channel.second.layers.end())
			continue;
		auto& channel_block = channel.second.channel_block;
		auto& layer_block = layer_block_itr->second;
		buffer.seekg(this->getFieldDataOffset() + channel_block.offset + layer_block.offset + sizeof(FiledTypes::V1::ChannelHeader), std::ios::beg);
		char* data_buffer = new char[layer_block.size];
		buffer.read(data_buffer, layer_block.size);
		VoxelLayer* layer = this->serializer->deserializeLayer(data_buffer, layer_block.size);
		delete[] data_buffer;
		layers.insert(std::make_pair(channel.first, std::make_shared<VoxelGrid>(this->field_dimensions, this->voxel_dimensions, std::shared_ptr<VoxelLayer>(layer))));
	}

	return layers;
}

IVoxel* RadFiled3D::Storage::V1::CartesianFieldAccessor::accessVoxelRaw(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec3& voxel_idx) const
{
	const size_t idx = this->default_grid->get_voxel_idx(voxel_idx.x, voxel_idx.y, voxel_idx.z);
	return this->accessVoxelRawFlat(buffer, channel_name, layer_name, idx);
}

IVoxel* RadFiled3D::Storage::V1::CartesianFieldAccessor::accessVoxelRawByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec3& voxel_pos) const
{
	const size_t idx = this->default_grid->get_voxel_idx_by_coord(voxel_pos.x, voxel_pos.y, voxel_pos.z);
	return this->accessVoxelRawFlat(buffer, channel_name, layer_name, idx);
}

void RadFiled3D::Storage::V1::PolarFieldAccessor::initialize(std::istream& buffer)
{
	this->metadata_fileheader_size = RadFiled3D::Storage::V1::MetadataAccessor().get_metadata_size(buffer) + sizeof(Storage::FiledTypes::VersionHeader);
	
	buffer.seekg(this->getMetadataFileheaderOffset() + sizeof(FiledTypes::V1::RadiationFieldHeader), std::ios::beg);
	FiledTypes::V1::PolarHeader ph;
	buffer.read((char*)&ph, sizeof(FiledTypes::V1::PolarHeader));

	this->segments_counts = ph.segments_counts;
	this->voxel_count = ph.segments_counts.x * ph.segments_counts.y;
	this->default_segments = std::make_unique<PolarSegments>(ph.segments_counts);

	FileParser::initialize(buffer);
}

RadFiled3D::Storage::V1::PolarFieldAccessor::PolarFieldAccessor()
	: RadFiled3D::Storage::FieldAccessor(FieldType::Polar),
	  RadFiled3D::Storage::PolarFieldAccessor(FieldType::Polar),
	  RadFiled3D::Storage::V1::FileParser(FieldType::Polar)
{
	this->segments_counts = glm::uvec2(0);
	this->store_version = StoreVersion::V1;
}

RadFiled3D::Storage::V1::PolarFieldAccessor::PolarFieldAccessor(const SerializationData& data)
	: RadFiled3D::Storage::FieldAccessor(FieldType::Polar),
	  RadFiled3D::Storage::PolarFieldAccessor(FieldType::Polar),
	  RadFiled3D::Storage::V1::FileParser(FieldType::Polar)
{
	if (data.store_version != StoreVersion::V1)
		throw RadiationFieldStoreException("Invalid store version");

	this->metadata_fileheader_size = data.metadata_fileheader_size;
	this->voxel_count = data.voxel_count;

	this->segments_counts = data.segments_counts;
	this->store_version = StoreVersion::V1;
	this->channels_layers_offsets = data.channels_layers_offsets;
	this->default_segments = std::make_unique<PolarSegments>(this->segments_counts);
	this->serializer = std::make_unique<BinayFieldBlockHandler>();
}

IVoxel* RadFiled3D::Storage::V1::PolarFieldAccessor::accessVoxelRaw(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec2& voxel_idx) const
{
	size_t idx = this->default_segments->get_segment_idx(voxel_idx.x, voxel_idx.y);
	return this->accessVoxelRawFlat(buffer, channel_name, layer_name, idx);
}

IVoxel* RadFiled3D::Storage::V1::PolarFieldAccessor::accessVoxelRawByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec2& voxel_pos) const
{
	size_t idx = this->default_segments->get_segment_idx_by_coord(voxel_pos.x, voxel_pos.y);
	return this->accessVoxelRawFlat(buffer, channel_name, layer_name, idx);
}

std::shared_ptr<IRadiationField> RadFiled3D::Storage::V1::PolarFieldAccessor::accessField(std::istream& buffer) const
{
	buffer.seekg(this->getMetadataFileheaderOffset(), std::ios::beg);
	return this->serializer->deserializeField(buffer);
}

std::shared_ptr<PolarSegments> RadFiled3D::Storage::V1::PolarFieldAccessor::accessLayer(std::istream& buffer, const std::string& channel_name, const std::string& layer_name) const
{
	auto channel_block_itr = this->channels_layers_offsets.find(channel_name);
	if (channel_block_itr == this->channels_layers_offsets.end())
		throw RadiationFieldStoreException("Channel not found");

	auto layer_block_itr = channel_block_itr->second.layers.find(layer_name);
	if (layer_block_itr == channel_block_itr->second.layers.end())
		throw RadiationFieldStoreException("Layer not found");

	auto& channel_block = channel_block_itr->second.channel_block;
	auto& layer_block = layer_block_itr->second;

	buffer.seekg(this->getFieldDataOffset() + channel_block.offset + layer_block.offset + sizeof(FiledTypes::V1::ChannelHeader), std::ios::beg);

	char* data_buffer = new char[layer_block.size];
	buffer.read(data_buffer, layer_block.size);
	PolarSegments* layer = (PolarSegments*)this->serializer->deserializeLayer(data_buffer, layer_block.size);
	delete[] data_buffer;

	return std::shared_ptr<PolarSegments>(layer);
}

std::shared_ptr<FieldAccessor> FieldAccessorBuilder::Construct(std::istream& buffer)
{
	StoreVersion version = FieldAccessor::getStoreVersion(buffer);

	std::shared_ptr<FieldAccessor> accessor;

	switch (version)
	{
	case StoreVersion::V1:
		buffer.seekg(RadFiled3D::Storage::V1::MetadataAccessor().get_metadata_size(buffer) + sizeof(Storage::FiledTypes::VersionHeader), std::ios::beg);
		switch (RadFiled3D::Storage::V1::BinayFieldBlockHandler().getFieldType(buffer)) {
		case FieldType::Cartesian:
			accessor = std::static_pointer_cast<FieldAccessor>(std::shared_ptr<Storage::V1::CartesianFieldAccessor>(new Storage::V1::CartesianFieldAccessor()));
			break;
		case FieldType::Polar:
			accessor = std::static_pointer_cast<FieldAccessor>(std::shared_ptr<Storage::V1::PolarFieldAccessor>(new Storage::V1::PolarFieldAccessor()));
			break;
		default:
			throw RadiationFieldStoreException("Unsupported field type");
		}
		break;
	default:
		throw RadiationFieldStoreException("Unsupported file version");
	}

	accessor->initialize(buffer);

	return accessor;
}
