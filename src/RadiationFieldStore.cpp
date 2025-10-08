#include "RadFiled3D/storage/RadiationFieldStore.hpp"
#include "RadFiled3D/VoxelBuffer.hpp"
#include "RadFiled3D/VoxelGrid.hpp"
#include "RadFiled3D/PolarSegments.hpp"
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>
#include "RadFiled3D/RadiationField.hpp"
#include <ios>
#include <iostream>
#include <fstream>
#if defined _WIN32 || defined _WIN64
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif
#include <stdexcept>
#include "RadFiled3D/storage/FieldSerializer.hpp"
#include <RadFiled3D/helpers/FileLock.hpp>


using namespace RadFiled3D;
using namespace RadFiled3D::Storage::FiledTypes;
using namespace RadFiled3D::Storage;

std::shared_ptr<BasicFieldStore> FieldStore::store_instance = std::shared_ptr<BasicFieldStore>(nullptr);
StoreVersion FieldStore::store_version = StoreVersion::V1;
bool FieldStore::file_lock_syncronization = false;


void IRadiationFieldExporter::store(std::shared_ptr<IRadiationField> field, std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata, const std::string& file) const
{
	std::ofstream stream(file.c_str(), std::ios::out | std::ios::binary);

	this->serialize(stream, field, metadata);
}

void RadFiled3D::Storage::BasicFieldStore::serialize(std::ostream& stream, std::shared_ptr<IRadiationField> field, std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata) const
{
	stream.seekp(0, std::ios::beg);

	VersionHeader vh;
	memcpy(&vh.version, this->file_version.c_str(), std::min<size_t>(12, this->file_version.length()));
	stream.write((const char*)&vh, sizeof(VersionHeader));

	this->metadata_serializer->serializeMetadata(stream, metadata);

	this->field_serializer->serializeField(field, stream);
}

std::shared_ptr<IRadiationField> IRadiationFieldImporter::load(const std::string& file) const
{
	if (!fs::exists(file)) {
		std::string msg = "File " + file + " does not exist!";
		throw new RadiationFieldStoreException(msg.c_str());
	}

	std::ifstream stream(file.c_str(), std::ios::in | std::ios::binary);
	stream.seekg(0, std::ios::beg);

	return this->load(stream);
}

std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> IRadiationFieldImporter::load_metadata(const std::string& file) const
{
	if (!fs::exists(file)) {
		std::string msg = "File " + file + " does not exist!";
		throw new RadiationFieldStoreException(msg.c_str());
	}

	std::ifstream stream(file.c_str(), std::ios::in | std::ios::binary);
	return this->load_metadata(stream);
}

std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> IRadiationFieldImporter::peek_metadata(const std::string& file) const
{
	if (!fs::exists(file)) {
		std::string msg = "File " + file + " does not exist!";
		throw new RadiationFieldStoreException(msg.c_str());
	}

	std::ifstream stream(file.c_str(), std::ios::in | std::ios::binary);
	return this->peek_metadata(stream);
}

void RadFiled3D::Storage::BasicFieldStore::valdiate_file_version(std::istream& stream) const
{
	stream.seekg(0, std::ios::beg);
	Storage::FiledTypes::VersionHeader version;
	stream.read((char*)&version, sizeof(Storage::FiledTypes::VersionHeader));

	if (strcmp(version.version, this->file_version.c_str()) != 0) {
		std::string msg = "File version mismatch! This loader supports version: " + this->file_version + ", but target file was written with version: " + std::string(version.version);
		throw RadiationFieldStoreException(msg.c_str());
	}
}

FieldType Storage::BasicFieldStore::peek_field_type(std::istream& file_stream) const
{
	this->valdiate_file_version(file_stream);

	size_t metadata_size = this->metadata_accessor->get_metadata_size(file_stream);
	file_stream.seekg(metadata_size, std::ios::cur);

	return this->field_serializer->getFieldType(file_stream);
}

FieldType RadFiled3D::Storage::FieldStore::peek_field_type(std::istream& file_stream)
{
	if (FieldStore::store_instance.get() == nullptr) {
		StoreVersion version = FieldAccessor::getStoreVersion(file_stream);
		FieldStore::init_store_instance(version);
	}

	return FieldStore::store_instance->peek_field_type(file_stream);
}

std::shared_ptr<FieldAccessor> RadFiled3D::Storage::FieldStore::construct_accessor(const std::string& file)
{
	std::ifstream buffer(file, std::ios::in | std::ios::binary);
	return RadFiled3D::Storage::FieldStore::construct_accessor(buffer);
}

std::shared_ptr<FieldAccessor> RadFiled3D::Storage::FieldStore::construct_accessor(std::istream& buffer)
{
	return FieldAccessorBuilder::Construct(buffer);
}

std::shared_ptr<IRadiationField> Storage::BasicFieldStore::load(std::istream& buffer) const
{
	this->valdiate_file_version(buffer);

	size_t metadata_size = this->metadata_accessor->get_metadata_size(buffer);
	buffer.seekg(metadata_size, std::ios::cur);

	std::shared_ptr<IRadiationField> field = this->field_serializer->deserializeField(buffer);
	return field;
}

std::shared_ptr<VoxelLayer> Storage::V1::FieldStore::load_single_layer(std::istream& buffer, const std::string& channel, const std::string& layer_name) const
{
	this->valdiate_file_version(buffer);
	FiledTypes::V1::RadiationFieldHeader desc;

	size_t metadata_size = this->get_metadata_accessor().get_metadata_size(buffer);
	buffer.seekg(metadata_size, std::ios::cur);

	FieldType field_type = this->get_field_serializer().getFieldType(buffer);

	size_t voxel_count = 0;

	if (field_type == FieldType::Cartesian) {
		FiledTypes::V1::CartesianHeader ch;
		buffer.read((char*)&ch, sizeof(FiledTypes::V1::CartesianHeader));
		voxel_count = ch.voxel_counts.x * ch.voxel_counts.y * ch.voxel_counts.z;
	}
	else if (field_type == FieldType::Polar) {
		FiledTypes::V1::PolarHeader ph;
		buffer.read((char*)&ph, sizeof(FiledTypes::V1::PolarHeader));
		voxel_count = ph.segments_counts.x * ph.segments_counts.y;
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

		if (std::string(ch.name) != channel) {
			buffer.seekg(ch.channel_bytes, std::ios::cur);
			continue;
		}

		char* byte_buffer = new char[ch.channel_bytes];
		buffer.read(byte_buffer, ch.channel_bytes);
		size_t layer_offset = 0;
		while (layer_offset < ch.channel_bytes) {
			FiledTypes::V1::VoxelGridLayerHeader* layer_desc = (FiledTypes::V1::VoxelGridLayerHeader*)(byte_buffer + layer_offset);
			if (layer_desc->bytes_per_element == 0) {
				delete[] byte_buffer;
				throw RadiationFieldStoreException("Layer: '" + layer_name + "' is incomplete in channel: " + channel);
			}
			size_t needed_size = layer_desc->bytes_per_element * voxel_count + sizeof(FiledTypes::V1::VoxelGridLayerHeader) + layer_desc->header_block_size;
			if (std::string(layer_desc->name) == layer_name) {
				size_t available_size = ch.channel_bytes - layer_offset;
				if (needed_size > available_size) {
					delete[] byte_buffer;
					throw RadiationFieldStoreException("Layer: '" + layer_name + "' is incomplete in channel: " + channel);
				}
				auto layer = std::shared_ptr<VoxelLayer>(this->get_field_serializer().deserializeLayer(byte_buffer + layer_offset, needed_size));
				delete[] byte_buffer;
				return layer;
			}
			layer_offset += needed_size;
		}
		delete[] byte_buffer;
		throw RadiationFieldStoreException("Layer: '" + layer_name + "' not found in channel: " + channel);
	}

	std::string msg = "Layer: '" + layer_name + "' not found in channel: " + channel;
	throw RadiationFieldStoreException(msg.c_str());
}

std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> Storage::BasicFieldStore::peek_metadata(std::istream& buffer) const
{
	this->valdiate_file_version(buffer);
	return this->get_metadata_accessor().accessMetadata(buffer, true);
}

std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> RadFiled3D::Storage::BasicFieldStore::load_metadata(std::istream& buffer) const
{
	this->valdiate_file_version(buffer);
	return this->metadata_accessor->accessMetadata(buffer, false);

}

void Storage::V1::FieldStore::join(std::shared_ptr<IRadiationField> target, std::shared_ptr<IRadiationField> additional_source, FieldJoinMode join_mode, FieldJoinCheckMode check_mode, float ratio) const
{
	if (target->get_typename() != additional_source->get_typename()) {
		std::string msg = "Field type mismatch! Existing field is of type: " + target->get_typename() + ", but target field is of type: " + additional_source->get_typename();
		throw RadiationFieldStoreException(msg.c_str());
	}

	for (auto& channel : additional_source->get_channels()) {
		if (!target->has_channel(channel.first)) {
			if (check_mode <= FieldJoinCheckMode::FieldStructureOnly)
				throw RadiationFieldStoreException("Channel: '" + channel.first + "' not found in target field");
			target->add_channel(channel.first);
		}
		auto target_channel = target->get_generic_channel(channel.first);

		for (auto& layer_name : channel.second->get_layers()) {
			if (!target_channel->has_layer(layer_name)) {
				if (check_mode <= FieldJoinCheckMode::FieldStructureOnly)
					throw RadiationFieldStoreException("Layer: '" + layer_name + "' not found in target field");
				target_channel->add_custom_layer_unsafe(layer_name, &channel.second->get_voxel_flat(layer_name, 0), channel.second->get_layer_unit(layer_name));
			}

			const Typing::DType dtype1 = Typing::Helper::get_dtype(channel.second->get_voxel_flat<IVoxel>(layer_name, 0).get_type());
			const Typing::DType dtype2 = Typing::Helper::get_dtype(target_channel->get_voxel_flat<IVoxel>(layer_name, 0).get_type());

			if (dtype1 != dtype2)
				throw RadiationFieldStoreException("Data type mismatch for layer: '" + layer_name + "' in channel: " + channel.first);
			if (target_channel->get_voxel_count() != channel.second->get_voxel_count())
				throw RadiationFieldStoreException("Voxel count mismatch for layer: '" + layer_name + "' in channel: " + channel.first);
			if (check_mode <= FieldJoinCheckMode::FieldUnitsOnly && target_channel->get_layer_unit(layer_name) != channel.second->get_layer_unit(layer_name))
				throw RadiationFieldStoreException("Unit mismatch for layer: '" + layer_name + "' in channel: " + channel.first + ". Existing unit: " + target_channel->get_layer_unit(layer_name) + ", but target unit: " + channel.second->get_layer_unit(layer_name));

			switch (dtype1) {
				case Typing::DType::Float:
					target_channel->merge_data_buffer<float>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<float>(join_mode, ratio));
					break;
				case Typing::DType::Double:
#if defined(__x86_64__) || defined(_M_X64)
					target_channel->merge_data_buffer<double>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<double>(join_mode, ratio));
#else
					throw RadiationFieldStoreException("Can't use 64-bit data type in 32-bit system!");
#endif
					break;
				case Typing::DType::Char:
					throw RadiationFieldStoreException("Unsupported data type 'char' for merging of layer: '" + layer_name + "' in channel: " + channel.first);
					//target_channel->merge_data_buffer<char>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<char>(join_mode, ratio));
					break;
				case Typing::DType::Byte:
					throw RadiationFieldStoreException("Unsupported data type 'byte' for merging of layer: '" + layer_name + "' in channel: " + channel.first);
					//target_channel->merge_data_buffer<uint8_t>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<uint8_t>(join_mode, ratio));
					break;
				case Typing::DType::Int:
					target_channel->merge_data_buffer<int>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<int>(join_mode, ratio));
					break;
				case Typing::DType::UInt64:
#if defined(__x86_64__) || defined(_M_X64)
					target_channel->merge_data_buffer<uint64_t>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<uint64_t>(join_mode, ratio));
#else
					throw RadiationFieldStoreException("Can't use 64-bit data type in 32-bit system!");
#endif
					break;
				case Typing::DType::Vec2:
					target_channel->merge_data_buffer<glm::vec2>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<glm::vec2>(join_mode, ratio));
					break;
				case Typing::DType::Vec3:
					target_channel->merge_data_buffer<glm::vec3>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<glm::vec3>(join_mode, ratio));
					break;
				case Typing::DType::Vec4:
					target_channel->merge_data_buffer<glm::vec4>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<glm::vec4>(join_mode, ratio));
					break;
				case Typing::DType::Hist:
					target_channel->merge_voxel_buffer<HistogramVoxel>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<HistogramVoxel, float>(join_mode, ratio));
					break;
			}
		}
	}
}

StoreVersion RadFiled3D::Storage::FieldStore::get_store_version(const std::string& file)
{
	std::ifstream buffer(file, std::ios::in | std::ios::binary);
	return FieldAccessor::getStoreVersion(buffer);
}

StoreVersion RadFiled3D::Storage::FieldStore::get_store_version(std::istream& buffer)
{
	return FieldAccessor::getStoreVersion(buffer);
}

void FieldStore::init_store_instance(StoreVersion version)
{
	switch (version) {
		case StoreVersion::V1:
			FieldStore::store_instance = std::make_shared<Storage::V1::FieldStore>();
			FieldStore::store_version = version;
			break;
		default:
			throw RadiationFieldStoreException("Unimplemented file version!");
	}
}

void FieldStore::store(std::shared_ptr<IRadiationField> field, std::shared_ptr<RadiationFieldMetadata> metadata, const std::string& file, StoreVersion version)
{
	if (FieldStore::store_instance.get() == nullptr || version != FieldStore::store_version) {
		FieldStore::init_store_instance(version);
	}

	FieldStore::store_instance->store(field, metadata, file);
}

void FieldStore::serialize(std::ostream& stream, std::shared_ptr<IRadiationField> field, std::shared_ptr<RadiationFieldMetadata> metadata, StoreVersion version)
{
	if (FieldStore::store_instance.get() == nullptr || version != FieldStore::store_version) {
		FieldStore::init_store_instance(version);
	}

	FieldStore::store_instance->serialize(stream, field, metadata);
}

std::shared_ptr<IRadiationField> FieldStore::load(const std::string& file)
{
	StoreVersion version = FieldStore::get_store_version(file);
	if (FieldStore::store_instance.get() == nullptr || version != FieldStore::store_version) {
		FieldStore::init_store_instance(version);
	}
	
	std::ifstream buffer(file, std::ios::in | std::ios::binary);
	return FieldStore::store_instance->load(buffer);
}

std::shared_ptr<IRadiationField> FieldStore::load(std::istream& buffer)
{
	StoreVersion version = FieldStore::get_store_version(buffer);
	if (FieldStore::store_instance.get() == nullptr || version != FieldStore::store_version) {
		FieldStore::init_store_instance(version);
	}

	return FieldStore::store_instance->load(buffer);
}

std::shared_ptr<RadiationFieldMetadata> FieldStore::load_metadata(const std::string& file)
{
	StoreVersion version = FieldStore::get_store_version(file);
	if (FieldStore::store_instance.get() == nullptr || version != FieldStore::store_version) {
		FieldStore::init_store_instance(version);
	}

	std::ifstream buffer(file, std::ios::in | std::ios::binary);
	return FieldStore::store_instance->load_metadata(buffer);
}

std::shared_ptr<RadiationFieldMetadata> FieldStore::load_metadata(std::istream& buffer)
{
	StoreVersion version = FieldStore::get_store_version(buffer);
	if (FieldStore::store_instance.get() == nullptr || version != FieldStore::store_version) {
		FieldStore::init_store_instance(version);
	}

	return FieldStore::store_instance->load_metadata(buffer);
}

std::shared_ptr<VoxelLayer> FieldStore::load_single_layer(std::istream& buffer, const std::string& channel, const std::string& layer)
{
	StoreVersion version = FieldStore::get_store_version(buffer);
	if (FieldStore::store_instance.get() == nullptr || version != FieldStore::store_version) {
		FieldStore::init_store_instance(version);
	}

	return FieldStore::store_instance->load_single_layer(buffer, channel, layer);
}

void FieldStore::join(std::shared_ptr<IRadiationField> field, std::shared_ptr<RadiationFieldMetadata> metadata, const std::string& file, FieldJoinMode join_mode, FieldJoinCheckMode check_mode, StoreVersion fallback_version)
{
	FileLock file_lock(file, FieldStore::file_lock_syncronization);

	if (!fs::exists(file)) {
		if (FieldStore::store_instance.get() == nullptr)
			FieldStore::init_store_instance(fallback_version);

		FieldStore::store(field, metadata, file, FieldStore::store_version);
		return;
	}

	Storage::V1::RadiationFieldMetadata& v1_metadata = dynamic_cast<Storage::V1::RadiationFieldMetadata&>(*metadata);

	std::shared_ptr<IRadiationField> existing_field = FieldStore::load(file);
	std::shared_ptr<RadiationFieldMetadata> _target_metadata = FieldStore::peek_metadata(file);
	FiledTypes::V1::RadiationFieldMetadataHeader target_metadata = dynamic_cast<Storage::V1::RadiationFieldMetadata&>(*_target_metadata).get_header();

	switch (check_mode) {
		case FieldJoinCheckMode::Strict:
			if (v1_metadata.get_header().simulation.primary_particle_count != target_metadata.simulation.primary_particle_count) {
				std::string msg = "Primary particle count mismatch! Existing field has: " + std::to_string(target_metadata.simulation.primary_particle_count) + ", but target field has: " + std::to_string(v1_metadata.get_header().simulation.primary_particle_count);
				throw RadiationFieldStoreException(msg.c_str());
			}
			[[fallthrough]];
		case FieldJoinCheckMode::MetadataSimulationSimilar:
			if (std::string(v1_metadata.get_header().simulation.geometry) != std::string(target_metadata.simulation.geometry)) {
				std::string msg = "Geometry mismatch! Existing field has: " + std::string(target_metadata.simulation.geometry) + ", but target field has: " + std::string(v1_metadata.get_header().simulation.geometry);
				throw RadiationFieldStoreException(msg.c_str());
			}
			if (std::string(v1_metadata.get_header().simulation.physics_list) != std::string(target_metadata.simulation.physics_list)) {
				std::string msg = "Physics list mismatch! Existing field has: " + std::string(target_metadata.simulation.physics_list) + ", but target field has: " + std::string(v1_metadata.get_header().simulation.physics_list);
				throw RadiationFieldStoreException(msg.c_str());
			}
			if (v1_metadata.get_header().simulation.tube.max_energy_eV != target_metadata.simulation.tube.max_energy_eV) {
				std::string msg = "Tube max energy mismatch! Existing field has: " + std::to_string(target_metadata.simulation.tube.max_energy_eV) + ", but target field has: " + std::to_string(v1_metadata.get_header().simulation.tube.max_energy_eV);
				throw RadiationFieldStoreException(msg.c_str());
			}
			if (v1_metadata.get_header().simulation.tube.radiation_direction != target_metadata.simulation.tube.radiation_direction) {
				throw RadiationFieldStoreException("Radiation direction mismatch!");
			}
			if (v1_metadata.get_header().simulation.tube.radiation_origin != target_metadata.simulation.tube.radiation_origin) {
				throw RadiationFieldStoreException("Radiation origin mismatch!");
			}
			if (std::string(v1_metadata.get_header().simulation.tube.tube_id) != std::string(target_metadata.simulation.tube.tube_id)) {
				throw RadiationFieldStoreException("Radiation tube_id mismatch!");
			}
			[[fallthrough]];
		case FieldJoinCheckMode::MetadataSoftwareEqual:
			if (std::string(v1_metadata.get_header().software.version) != std::string(target_metadata.software.version)) {
				throw RadiationFieldStoreException("Software version mismatch!");
			}
			if (std::string(v1_metadata.get_header().software.doi) != std::string(target_metadata.software.doi)) {
				throw RadiationFieldStoreException("Software DOI mismatch!");
			}
			if (std::string(v1_metadata.get_header().software.commit) != std::string(target_metadata.software.commit)) {
				throw RadiationFieldStoreException("Software commit mismatch!");
			}
			[[fallthrough]];
		case FieldJoinCheckMode::MetadataSoftwareSimilar:
			if (std::string(v1_metadata.get_header().software.name) != std::string(target_metadata.software.name)) {
				throw RadiationFieldStoreException("Software name mismatch!");
			}
			if (std::string(v1_metadata.get_header().software.repository) != std::string(target_metadata.software.repository)) {
				throw RadiationFieldStoreException("Software repository mismatch!");
			}
	}

	float ratio = static_cast<float>(v1_metadata.get_header().simulation.primary_particle_count) / static_cast<float>(target_metadata.simulation.primary_particle_count + v1_metadata.get_header().simulation.primary_particle_count);

	FieldStore::store_instance->join(existing_field, field, join_mode, check_mode, ratio);

	target_metadata.simulation.primary_particle_count += v1_metadata.get_header().simulation.primary_particle_count;
	v1_metadata.set_header(target_metadata);
	FieldStore::store_instance->store(existing_field, metadata, file);
}

std::shared_ptr<Storage::RadiationFieldMetadata> FieldStore::peek_metadata(const std::string& file)
{
	StoreVersion version = FieldStore::get_store_version(file);
	if (FieldStore::store_instance.get() == nullptr || version != FieldStore::store_version) {
		FieldStore::init_store_instance(version);
	}

	std::ifstream buffer(file, std::ios::in | std::ios::binary);
	return FieldStore::store_instance->peek_metadata(buffer);
}

std::shared_ptr<Storage::RadiationFieldMetadata> FieldStore::peek_metadata(std::istream& buffer)
{
	StoreVersion version = FieldStore::get_store_version(buffer);
	if (FieldStore::store_instance.get() == nullptr || version != FieldStore::store_version) {
		FieldStore::init_store_instance(version);
	}

	return FieldStore::store_instance->peek_metadata(buffer);
}
