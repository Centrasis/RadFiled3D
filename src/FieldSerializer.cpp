#pragma comment(lib, "Dbghelp.lib")
#include "RadFiled3D/storage/FieldSerializer.hpp"
#include "RadFiled3D/VoxelBuffer.hpp"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <string.h>
#include <RadFiled3D/helpers/Typing.hpp>
#include <RadFiled3D/RadiationField.hpp>


using namespace RadFiled3D;
using namespace RadFiled3D::Storage;
using namespace RadFiled3D::Storage::FiledTypes;

void Storage::V1::BinayFieldBlockHandler::serializeField(std::shared_ptr<IRadiationField> field, std::ostream& buffer) const
{
	FiledTypes::V1::RadiationFieldHeader desc;

	const std::string field_type = field->get_typename();
	std::strncpy(desc.field_type, field_type.c_str(), std::min<size_t>(64, field_type.length()));
	buffer.write((const char*)&desc, sizeof(FiledTypes::V1::RadiationFieldHeader));

	if (field_type == "CartesianRadiationField") {
		auto field_cartesian = std::dynamic_pointer_cast<CartesianRadiationField>(field);
		FiledTypes::V1::CartesianHeader ch;
		ch.voxel_counts = field_cartesian->get_voxel_counts();
		ch.voxel_dimensions = field_cartesian->get_voxel_dimensions();
		buffer.write((const char*)&ch, sizeof(FiledTypes::V1::CartesianHeader));
	}
	else if (field_type == "PolarRadiationField") {
		auto field_polar = std::dynamic_pointer_cast<PolarRadiationField>(field);
		FiledTypes::V1::PolarHeader ph;
		ph.segments_counts = field_polar->get_segments_count();
		buffer.write((const char*)&ph, sizeof(FiledTypes::V1::PolarHeader));
	}
	else {
		std::string msg = "Field type " + field_type + " is not supported!";
		throw RadiationFieldStoreException(msg.c_str());
	}

	auto channels = field->get_channels();
	for (auto& channel : channels) {
		FiledTypes::V1::ChannelHeader ch;
		std::strncpy(ch.name, channel.first.c_str(), std::min<size_t>(64, channel.first.length()));
		auto serialized_field = this->serializeChannel(channel.second);
		ch.channel_bytes = serialized_field->str().length();
		buffer.write((const char*)&ch, sizeof(FiledTypes::V1::ChannelHeader));
		buffer.write(serialized_field->str().c_str(), ch.channel_bytes);
	}
}

std::unique_ptr<std::ostringstream> Storage::V1::BinayFieldBlockHandler::serializeChannel(std::shared_ptr<VoxelBuffer> voxel_buffer) const
{
	auto layers = voxel_buffer->get_layers();

	std::unique_ptr<std::ostringstream> oss = std::make_unique<std::ostringstream>();
	for (auto& layer_name : layers) {
		FiledTypes::V1::VoxelGridLayerHeader layer_desc;
		const IVoxel& voxel = voxel_buffer->get_voxel_flat(layer_name, 0);
		layer_desc.bytes_per_element = voxel.get_bytes();
		std::strncpy(layer_desc.dtype, voxel.get_type().c_str(), std::min<size_t>(32, voxel.get_type().length()));
		std::strncpy(layer_desc.name, layer_name.c_str(), std::min<size_t>(64, layer_name.length()));
		const std::string layer_unit = voxel_buffer->get_layer_unit(layer_name);
		std::strncpy(layer_desc.unit, layer_unit.c_str(), std::min<size_t>(32, layer_unit.length()));
		layer_desc.statistical_error = voxel_buffer->get_statistical_error(layer_name);
		if (voxel.get_header().header_bytes > 0)
			layer_desc.header_block_size = voxel.get_header().header_bytes;
		oss->write((const char*)&layer_desc, sizeof(FiledTypes::V1::VoxelGridLayerHeader));
		if (layer_desc.header_block_size > 0) {
			const char* header_data = (char*)voxel.get_header().header;
			oss->write(header_data, layer_desc.header_block_size);
		}
		const char* data_buffer = voxel_buffer->get_layer<char>(layer_name);
		oss->write(data_buffer, voxel_buffer->get_voxel_count() * layer_desc.bytes_per_element);
	}
	return oss;
}

VoxelLayer* Storage::V1::BinayFieldBlockHandler::deserializeLayer(char* data, size_t size) const
{
	if (size < sizeof(FiledTypes::V1::VoxelGridLayerHeader))
		throw std::runtime_error("Data is too small to contain a valid layer header");

	size_t mem_pos = 0;
	FiledTypes::V1::VoxelGridLayerHeader layer_desc = *(FiledTypes::V1::VoxelGridLayerHeader*)(data);
	mem_pos += sizeof(FiledTypes::V1::VoxelGridLayerHeader);
	void* header_data = nullptr;

	if (layer_desc.header_block_size > 0) {
		header_data = (void*)(data + mem_pos);
		mem_pos += layer_desc.header_block_size;
	}

	if (mem_pos >= size)
		throw std::runtime_error("Data is too small to contain a valid layer data");

	size_t remaining_bytes = size - mem_pos;
	size_t voxel_count = remaining_bytes / layer_desc.bytes_per_element;

	Typing::DType dtype = Typing::Helper::get_dtype(std::string(layer_desc.dtype));
	VoxelLayer* layer = nullptr;
	HistogramVoxel hist_template;
	
	switch (dtype)
	{
	case Typing::DType::Float:
		layer = VoxelLayer::ConstructFromBufferRaw<float>(std::string(layer_desc.unit), voxel_count, layer_desc.statistical_error, data + mem_pos, true);
		break;
	case Typing::DType::Double:
#if defined(__x86_64__) || defined(_M_X64)
		layer = VoxelLayer::ConstructFromBufferRaw<double>(std::string(layer_desc.unit), voxel_count, layer_desc.statistical_error, data + mem_pos, true);
#else
		throw std::runtime_error("Can't load 64-bit file in 32-bit system!");
#endif
		break;
	case Typing::DType::Int:
		layer = VoxelLayer::ConstructFromBufferRaw<int>(std::string(layer_desc.unit), voxel_count, layer_desc.statistical_error, data + mem_pos, true);
		break;
	case Typing::DType::Char:
		layer = VoxelLayer::ConstructFromBufferRaw<char>(std::string(layer_desc.unit), voxel_count, layer_desc.statistical_error, data + mem_pos, true);
		break;
	case Typing::DType::Byte:
		layer = VoxelLayer::ConstructFromBufferRaw<uint8_t>(std::string(layer_desc.unit), voxel_count, layer_desc.statistical_error, data + mem_pos, true);
		break;
	case Typing::DType::Vec2:
		layer = VoxelLayer::ConstructFromBufferRaw<glm::vec2>(std::string(layer_desc.unit), voxel_count, layer_desc.statistical_error, data + mem_pos, true);
		break;
	case Typing::DType::Vec3:
		layer = VoxelLayer::ConstructFromBufferRaw<glm::vec3>(std::string(layer_desc.unit), voxel_count, layer_desc.statistical_error, data + mem_pos, true);
		break;
	case Typing::DType::Vec4:
		layer = VoxelLayer::ConstructFromBufferRaw<glm::vec4>(std::string(layer_desc.unit), voxel_count, layer_desc.statistical_error, data + mem_pos, true);
		break;
	case Typing::DType::Hist:
		if (header_data != nullptr)
			hist_template.init_from_header(header_data);
		layer = VoxelLayer::ConstructFromBufferRaw<float, HistogramVoxel>(std::string(layer_desc.unit), voxel_count, layer_desc.statistical_error, data + mem_pos, true, hist_template);
		break;
	case Typing::DType::UInt64:
#if defined(__x86_64__) || defined(_M_X64)
		layer = VoxelLayer::ConstructFromBufferRaw<uint64_t>(std::string(layer_desc.unit), voxel_count, layer_desc.statistical_error, data + mem_pos, true);
#else
		throw std::runtime_error("Can't load 64-bit file in 32-bit system!");
#endif
		break;
	case Typing::DType::UInt32:
		layer = VoxelLayer::ConstructFromBufferRaw<uint32_t>(std::string(layer_desc.unit), voxel_count, layer_desc.statistical_error, data + mem_pos, true);
		break;
	default:
		throw std::runtime_error("Failed to find data-type for layer! Data-type was: " + std::string(layer_desc.dtype));
	}
	return layer;
}

std::shared_ptr<VoxelBuffer> Storage::V1::BinayFieldBlockHandler::deserializeChannel(std::shared_ptr<VoxelBuffer> destination, char* data, size_t size) const
{
	size_t mem_pos = 0;
	while (mem_pos < size) {
		const FiledTypes::V1::VoxelGridLayerHeader& layer_desc = *(FiledTypes::V1::VoxelGridLayerHeader*)(data + mem_pos);
		mem_pos += sizeof(FiledTypes::V1::VoxelGridLayerHeader);
		void* header_data = nullptr;

		if (layer_desc.header_block_size > 0) {
			header_data = (void*)(data + mem_pos);
			mem_pos += layer_desc.header_block_size;
		}

		Typing::DType dtype = Typing::Helper::get_dtype(std::string(layer_desc.dtype));

		switch (dtype) {
			case Typing::DType::Float:
				destination->add_layer<float>(std::string(layer_desc.name), 0.f, layer_desc.unit);
				break;
			case Typing::DType::Double:
#if defined(__x86_64__) || defined(_M_X64)
				destination->add_layer<double>(std::string(layer_desc.name), 0.0, layer_desc.unit);
#else
				throw std::runtime_error("Can't load 64-bit file in 32-bit system!");
#endif
				break;
			case Typing::DType::Int:
				destination->add_layer<int>(std::string(layer_desc.name), 0, layer_desc.unit);
				break;
			case Typing::DType::Char:
				destination->add_layer<char>(std::string(layer_desc.name), 0, layer_desc.unit);
				break;
			case Typing::DType::Byte:
				destination->add_layer<uint8_t>(std::string(layer_desc.name), 0, layer_desc.unit);
				break;
			case Typing::DType::Vec3:
				destination->add_layer<glm::vec3>(std::string(layer_desc.name), glm::vec3(0.f), layer_desc.unit);
				break;
			case Typing::DType::Vec2:
				destination->add_layer<glm::vec2>(std::string(layer_desc.name), glm::vec2(0.f), layer_desc.unit);
				break;
			case Typing::DType::Vec4:
				destination->add_layer<glm::vec4>(std::string(layer_desc.name), glm::vec4(0.f), layer_desc.unit);
				break;
			case Typing::DType::Hist:
				Storage::V1::BinayFieldBlockHandler::add_hist_layer(destination, std::string(layer_desc.name), layer_desc.bytes_per_element, 0, layer_desc.unit, header_data);
				break;
			case Typing::DType::UInt64:
#if defined(__x86_64__) || defined(_M_X64)
				destination->add_layer<uint64_t>(std::string(layer_desc.name), 0, layer_desc.unit);
#else
				throw std::runtime_error("Can't load 64-bit file in 32-bit system!");
#endif
				break;
			case Typing::DType::UInt32:
				destination->add_layer<uint32_t>(std::string(layer_desc.name), 0, layer_desc.unit);
				break;
			default:
				std::string msg = "Failed to find data-type for layer: '" + std::string(layer_desc.name) + "' and dtype: '" + std::string(layer_desc.dtype) + "'";
				throw std::runtime_error(msg.c_str());
		}

		destination->set_statistical_error(std::string(layer_desc.name), layer_desc.statistical_error);
		char* data_buffer = destination->get_layer<char>(std::string(layer_desc.name));
		memcpy(
			data_buffer,
			data + mem_pos,
			destination->get_voxel_count() * layer_desc.bytes_per_element
		);
		mem_pos += destination->get_voxel_count() * layer_desc.bytes_per_element;
	}

	return destination;
}

void Storage::V1::BinayFieldBlockHandler::add_hist_layer(std::shared_ptr<VoxelBuffer> field, const std::string& layer, size_t bytes_per_element, float max_energy_eV, const std::string& unit, void* header_data)
{
	HistogramVoxel hist;
	if (header_data != nullptr)
		hist.init_from_header(header_data);
	field->add_custom_layer<HistogramVoxel, float>(layer, hist, 0.f, unit);
}

std::shared_ptr<IRadiationField> RadFiled3D::Storage::V1::BinayFieldBlockHandler::deserializeField(std::istream& buffer) const
{
	FiledTypes::V1::RadiationFieldHeader desc;

	buffer.read((char*)&desc, sizeof(FiledTypes::V1::RadiationFieldHeader));

	std::shared_ptr<IRadiationField> field;
	if (strcmp(desc.field_type, "CartesianRadiationField") == 0) {
		FiledTypes::V1::CartesianHeader ch;
		buffer.read((char*)&ch, sizeof(FiledTypes::V1::CartesianHeader));
		field = std::make_shared<CartesianRadiationField>(glm::vec3(ch.voxel_counts) * ch.voxel_dimensions, ch.voxel_dimensions);
	}
	else if (strcmp(desc.field_type, "PolarRadiationField") == 0) {
		FiledTypes::V1::PolarHeader ph;
		buffer.read((char*)&ph, sizeof(FiledTypes::V1::PolarHeader));
		field = std::make_shared<PolarRadiationField>(ph.segments_counts);
	}
	else {
		std::string msg = "Field type " + std::string(desc.field_type) + " is not supported!";
		throw RadiationFieldStoreException(msg.c_str());
	}

	while (!buffer.eof()) {
		FiledTypes::V1::ChannelHeader ch;
		buffer.read((char*)&ch, sizeof(FiledTypes::V1::ChannelHeader));

		if (buffer.eof())
			break;

		char* channel_data = new char[ch.channel_bytes];
		buffer.read(channel_data, ch.channel_bytes);

		BinayFieldBlockHandler::deserializeChannel(field->add_channel(std::string(ch.name)), channel_data, ch.channel_bytes);
		delete[] channel_data;
	}

	return field;
}

RadFiled3D::FieldType RadFiled3D::Storage::V1::BinayFieldBlockHandler::getFieldType(std::istream& buffer) const
{
	FiledTypes::V1::RadiationFieldHeader desc;
	buffer.read((char*)&desc, sizeof(FiledTypes::V1::RadiationFieldHeader));

	if (strcmp(desc.field_type, "CartesianRadiationField") == 0) {
		return FieldType::Cartesian;
	}
	else if (strcmp(desc.field_type, "PolarRadiationField") == 0) {
		return FieldType::Polar;
	}
	else {
		std::string msg = "Field type " + std::string(desc.field_type) + " is not supported!";
		throw RadiationFieldStoreException(msg.c_str());
	}
}