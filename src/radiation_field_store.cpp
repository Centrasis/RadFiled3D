#include "radfiled3d/storage/radiation_field_store.hpp"
#include "radfiled3d/voxel_buffer.hpp"
#include "radfiled3d/voxel_grid.hpp"
#include "radfiled3d/polar_segments.hpp"
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>
#include "radfiled3d/radiation_field.hpp"
#include <ios>
#include <iostream>
#include <fstream>
#if defined _WIN32 || defined _WIN64
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif
#include <stdexcept>
#include <map>
#include <algorithm>
#include <cstring>
#include "radfiled3d/storage/field_serializer.hpp"
#include "radfiled3d/storage/registry.hpp"
#include <radfiled3d/helpers/file_lock.hpp>


using namespace radfiled3d;
using namespace radfiled3d::storage::filed_types;
using namespace radfiled3d::storage;

bool FieldStore::file_lock_synchronization = false;


void IRadiationFieldExporter::store(std::shared_ptr<IRadiationField> field, std::shared_ptr<radfiled3d::storage::RadiationFieldMetadata> metadata, const std::string& file) const
{
	std::ofstream stream(file.c_str(), std::ios::out | std::ios::binary);

	this->serialize(stream, field, metadata);
}

void radfiled3d::storage::BasicFieldStore::serialize(std::ostream& stream, std::shared_ptr<IRadiationField> field, std::shared_ptr<radfiled3d::storage::RadiationFieldMetadata> metadata) const
{
	stream.seekp(0, std::ios::beg);

	VersionHeader vh;
	memcpy(&vh.version, this->file_version.c_str(), std::min<size_t>(12, this->file_version.length()));
	stream.write((const char*)&vh, sizeof(VersionHeader));

	this->metadata_serializer->serializeMetadata(stream, metadata);

	this->field_serializer->serializeField(field, stream);
}

namespace {
	/** True if the header text has the shape of a RadFiled3D version, i.e. "<digits>.<digits>". */
	bool is_file_version_string(const std::string& version_str)
	{
		const size_t dot = version_str.find('.');
		if (dot == std::string::npos || dot == 0 || dot + 1 >= version_str.length())
			return false;

		for (size_t i = 0; i < version_str.length(); i++) {
			if (i == dot)
				continue;
			if (version_str[i] < '0' || version_str[i] > '9')
				return false;
		}
		return true;
	}

	/** Renders raw header bytes printably, so a message built from them stays valid UTF-8. */
	std::string describe_header_bytes(const char* data, size_t length)
	{
		static const char* hex = "0123456789abcdef";
		std::string described;
		for (size_t i = 0; i < length; i++) {
			const unsigned char byte = static_cast<unsigned char>(data[i]);
			if (byte >= 0x20 && byte < 0x7f)
				described += static_cast<char>(byte);
			else {
				described += "\\x";
				described += hex[(byte >> 4) & 0xf];
				described += hex[byte & 0xf];
			}
		}
		return described;
	}

	/** "File 'x'" when the path is known, "The given data" otherwise. */
	std::string describe_source(const std::string& file_name)
	{
		return file_name.empty() ? std::string("The given data") : "File '" + file_name + "'";
	}
}

std::string radfiled3d::storage::read_file_version_header(std::istream& stream, const std::string& file_name)
{
	static_assert(std::is_trivially_copyable_v<filed_types::VersionHeader>);

	stream.clear();
	stream.seekg(0, std::ios::beg);

	filed_types::VersionHeader version;
	stream.read((char*)&version, sizeof(filed_types::VersionHeader));
	const std::streamsize read_bytes = stream.gcount();
	stream.clear();

	if (read_bytes < static_cast<std::streamsize>(sizeof(filed_types::VersionHeader))) {
		stream.seekg(0, std::ios::beg);
		throw RadiationFieldStoreException(describe_source(file_name) + " is not a RadFiled3D (.rf3) file: it is too short to contain a file header.");
	}

	// The stored version need not be NUL-terminated when it fills the field.
	const size_t version_length = strnlen(version.version, sizeof(version.version));
	const std::string version_str(version.version, version_length);

	if (!is_file_version_string(version_str)) {
		stream.seekg(0, std::ios::beg);
		throw RadiationFieldStoreException(
			describe_source(file_name) + " is not a RadFiled3D (.rf3) file: expected a version header like '1.1', but the file starts with '"
			+ describe_header_bytes(version.version, sizeof(version.version)) + "'.");
	}

	// Leave the stream just past the header: callers continue with seeks relative to
	// this position (the metadata block follows immediately).
	stream.seekg(sizeof(filed_types::VersionHeader), std::ios::beg);
	return version_str;
}

std::ifstream radfiled3d::storage::open_file_for_reading(const std::string& file)
{
	std::error_code ec;
	if (!fs::exists(file, ec) || ec)
		throw RadiationFieldStoreException("File '" + file + "' does not exist!");

	if (fs::is_directory(file, ec))
		throw RadiationFieldStoreException("Path '" + file + "' is a directory, not a RadFiled3D file!");

	std::ifstream stream(file.c_str(), std::ios::in | std::ios::binary);
	if (!stream.is_open())
		throw RadiationFieldStoreException("File '" + file + "' exists, but could not be opened for reading!");

	// Reject an unrelated file here, where the path is still known for the message.
	storage::read_file_version_header(stream, file);

	stream.seekg(0, std::ios::beg);
	return stream;
}

std::shared_ptr<IRadiationField> IRadiationFieldImporter::load(const std::string& file) const
{
	std::ifstream stream = storage::open_file_for_reading(file);

	return this->load(stream);
}

std::shared_ptr<radfiled3d::storage::RadiationFieldMetadata> IRadiationFieldImporter::load_metadata(const std::string& file) const
{
	std::ifstream stream = storage::open_file_for_reading(file);
	return this->load_metadata(stream);
}

std::shared_ptr<radfiled3d::storage::RadiationFieldMetadata> IRadiationFieldImporter::peek_metadata(const std::string& file) const
{
	std::ifstream stream = storage::open_file_for_reading(file);
	return this->peek_metadata(stream);
}

void radfiled3d::storage::BasicFieldStore::validate_file_version(std::istream& stream) const
{
	// Throws a "not a RadFiled3D (.rf3) file" error when the header is not a version at all,
	// so only a genuine version-range mismatch reaches the check below.
	const std::string version_str = storage::read_file_version_header(stream);

	if (!this->check_version_string_validity(version_str)) {
		std::string version_range = std::to_string(this->validity_range.major.min) + "." + std::to_string(this->validity_range.minor.min) + " <= x.y <= " + std::to_string(this->validity_range.major.max) + "." + std::to_string(this->validity_range.minor.max);
		std::string msg = "File version mismatch! '" + version_str + "' was not within the supported range for this field store. Valid range would have been: " + version_range;
		throw RadiationFieldStoreException(msg.c_str());
	}
}

FieldType storage::BasicFieldStore::peek_field_type(std::istream& file_stream) const
{
	this->validate_file_version(file_stream);

	size_t metadata_size = this->metadata_accessor->get_metadata_size(file_stream);
	file_stream.seekg(metadata_size, std::ios::cur);

	return this->field_serializer->getFieldType(file_stream);
}

FieldType radfiled3d::storage::FieldStore::peek_field_type(std::istream& file_stream)
{
	return FieldStore::get_store_by(file_stream)->peek_field_type(file_stream);
}

std::shared_ptr<FieldAccessor> radfiled3d::storage::FieldStore::construct_accessor(const std::string& file)
{
	std::ifstream buffer = storage::open_file_for_reading(file);
	return radfiled3d::storage::FieldStore::construct_accessor(buffer);
}

std::shared_ptr<FieldAccessor> radfiled3d::storage::FieldStore::construct_accessor(std::istream& buffer)
{
	FieldStore::ensure_registered_stores();
	return FieldAccessorBuilder::Construct(buffer);
}

std::shared_ptr<IRadiationField> storage::BasicFieldStore::load(std::istream& buffer) const
{
	this->validate_file_version(buffer);

	size_t metadata_size = this->metadata_accessor->get_metadata_size(buffer);
	buffer.seekg(metadata_size, std::ios::cur);

	std::shared_ptr<IRadiationField> field = this->field_serializer->deserializeField(buffer);
	return field;
}

std::shared_ptr<VoxelLayer> storage::v1::FieldStore::load_single_layer(std::istream& buffer, const std::string& channel, const std::string& layer_name) const
{
	this->validate_file_version(buffer);
	filed_types::v1::RadiationFieldHeader desc;

	size_t metadata_size = this->get_metadata_accessor().get_metadata_size(buffer);
	buffer.seekg(metadata_size, std::ios::cur);

	FieldType field_type = this->get_field_serializer().getFieldType(buffer);

	size_t voxel_count = 0;

	if (field_type == FieldType::Cartesian) {
		filed_types::v1::CartesianHeader ch;
		buffer.read((char*)&ch, sizeof(filed_types::v1::CartesianHeader));
		voxel_count = ch.voxel_counts.x * ch.voxel_counts.y * ch.voxel_counts.z;
	}
	else if (field_type == FieldType::Polar) {
		filed_types::v1::PolarHeader ph;
		buffer.read((char*)&ph, sizeof(filed_types::v1::PolarHeader));
		voxel_count = ph.segments_counts.x * ph.segments_counts.y;
	}
	else {
		std::string msg = "Field type " + std::string(desc.field_type) + " is not supported!";
		throw RadiationFieldStoreException(msg.c_str());
	}

	while (!buffer.eof()) {
		filed_types::v1::ChannelHeader ch;
		buffer.read((char*)&ch, sizeof(filed_types::v1::ChannelHeader));

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
			filed_types::v1::VoxelGridLayerHeader* layer_desc = (filed_types::v1::VoxelGridLayerHeader*)(byte_buffer + layer_offset);
			if (layer_desc->bytes_per_element == 0) {
				delete[] byte_buffer;
				throw RadiationFieldStoreException("Layer: '" + layer_name + "' is incomplete in channel: " + channel);
			}
			size_t needed_size = layer_desc->bytes_per_element * voxel_count + sizeof(filed_types::v1::VoxelGridLayerHeader) + layer_desc->header_block_size;
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

std::shared_ptr<radfiled3d::storage::RadiationFieldMetadata> storage::BasicFieldStore::peek_metadata(std::istream& buffer) const
{
	this->validate_file_version(buffer);
	return this->get_metadata_accessor().accessMetadata(buffer, true);
}

std::shared_ptr<radfiled3d::storage::RadiationFieldMetadata> radfiled3d::storage::BasicFieldStore::load_metadata(std::istream& buffer) const
{
	this->validate_file_version(buffer);
	return this->metadata_accessor->accessMetadata(buffer, false);

}

void storage::v1::FieldStore::join(std::shared_ptr<IRadiationField> target, std::shared_ptr<IRadiationField> additional_source, FieldJoinMode join_mode, FieldJoinCheckMode check_mode, float ratio) const
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
				if (target_channel->get_voxel_count() != channel.second->get_voxel_count())
					throw RadiationFieldStoreException("Voxel count mismatch for layer: '" + layer_name + "' in channel: " + channel.first);
				// A layer only present in the additional source has nothing to be joined with: take it over as is.
				target_channel->add_custom_layer_unsafe(layer_name, &channel.second->get_voxel_flat(layer_name, 0), channel.second->get_layer_unit(layer_name));
				target_channel->copy_layer_data(layer_name, *channel.second.get());
				continue;
			}

			const typing::DType dtype1 = typing::Helper::get_dtype(channel.second->get_voxel_flat<IVoxel>(layer_name, 0).get_type());
			const typing::DType dtype2 = typing::Helper::get_dtype(target_channel->get_voxel_flat<IVoxel>(layer_name, 0).get_type());

			if (dtype1 != dtype2)
				throw RadiationFieldStoreException("Data type mismatch for layer: '" + layer_name + "' in channel: " + channel.first);
			if (target_channel->get_voxel_count() != channel.second->get_voxel_count())
				throw RadiationFieldStoreException("Voxel count mismatch for layer: '" + layer_name + "' in channel: " + channel.first);
			if (check_mode <= FieldJoinCheckMode::FieldUnitsOnly && target_channel->get_layer_unit(layer_name) != channel.second->get_layer_unit(layer_name))
				throw RadiationFieldStoreException("Unit mismatch for layer: '" + layer_name + "' in channel: " + channel.first + ". Existing unit: " + target_channel->get_layer_unit(layer_name) + ", but target unit: " + channel.second->get_layer_unit(layer_name));

			switch (dtype1) {
				case typing::DType::Float:
					target_channel->merge_data_buffer<float>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<float>(join_mode, ratio));
					break;
				case typing::DType::Double:
#if RADFILED3D_HAS_64BIT
					target_channel->merge_data_buffer<double>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<double>(join_mode, ratio));
#else
					throw RadiationFieldStoreException("Can't use 64-bit data type in 32-bit system!");
#endif
					break;
				case typing::DType::Char:
					target_channel->merge_data_buffer<char>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<char>(join_mode, ratio));
					break;
				case typing::DType::Byte:
					target_channel->merge_data_buffer<uint8_t>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<uint8_t>(join_mode, ratio));
					break;
				case typing::DType::Int:
					target_channel->merge_data_buffer<int>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<int>(join_mode, ratio));
					break;
				case typing::DType::UInt64:
#if RADFILED3D_HAS_64BIT
					target_channel->merge_data_buffer<uint64_t>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<uint64_t>(join_mode, ratio));
#else
					throw RadiationFieldStoreException("Can't use 64-bit data type in 32-bit system!");
#endif
					break;
				case typing::DType::Int64:
#if RADFILED3D_HAS_64BIT
					target_channel->merge_data_buffer<int64_t>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<int64_t>(join_mode, ratio));
#else
					throw RadiationFieldStoreException("Can't use 64-bit data type in 32-bit system!");
#endif
					break;
				case typing::DType::Vec2:
					target_channel->merge_data_buffer<glm::vec2>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<glm::vec2>(join_mode, ratio));
					break;
				case typing::DType::Vec3:
					target_channel->merge_data_buffer<glm::vec3>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<glm::vec3>(join_mode, ratio));
					break;
				case typing::DType::Vec4:
					target_channel->merge_data_buffer<glm::vec4>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<glm::vec4>(join_mode, ratio));
					break;
				case typing::DType::Hist:
					target_channel->merge_voxel_buffer<HistogramVoxel<float>>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<HistogramVoxel<float>, float>(join_mode, ratio));
					break;
				case typing::DType::AngularResolved:
					target_channel->merge_voxel_buffer<AngularResolvedVoxel<float>>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<AngularResolvedVoxel<float>, float>(join_mode, ratio));
					break;
				case typing::DType::VMFMixture:
					target_channel->merge_voxel_buffer<VMFMixtureVoxel<float>>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<VMFMixtureVoxel<float>, float>(join_mode, ratio));
					break;
				case typing::DType::UInt32:
					target_channel->merge_data_buffer<uint32_t>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<uint32_t>(join_mode, ratio));
					break;
				case typing::DType::Float16:
#if RADFILED3D_HAS_FLOAT16
					target_channel->merge_data_buffer<typing::float16>(layer_name, *channel.second.get(), ExporterHelpers::get_join_function<typing::float16>(join_mode, ratio));
#else
					throw RadiationFieldStoreException("RadFiled3D was built without float16 support (needs GCC >= 12 or a modern Clang).");
#endif
					break;
				default:
					throw RadiationFieldStoreException("Unsupported data type for joining layer: '" + layer_name + "' in channel: " + channel.first);
			}
		}
	}
}

StoreVersion radfiled3d::storage::FieldStore::get_store_version(const std::string& file)
{
	std::ifstream buffer = storage::open_file_for_reading(file);
	return FieldAccessor::getStoreVersion(buffer);
}

StoreVersion radfiled3d::storage::FieldStore::get_store_version(std::istream& buffer)
{
	return FieldAccessor::getStoreVersion(buffer);
}

void FieldStore::ensure_registered_stores()
{
	// exec initialization once
	static const bool initialize = []() {
		radfiled3d::storage::Registry::register_store(StoreVersion::V1, std::move(std::make_unique<storage::v1::FieldStore>()));
		return true;
	}();
}

void FieldStore::store(std::shared_ptr<IRadiationField> field, std::shared_ptr<RadiationFieldMetadata> metadata, const std::string& file, StoreVersion version)
{
	FieldStore::get_store_by(version)->store(field, metadata, file);
}

void FieldStore::serialize(std::ostream& stream, std::shared_ptr<IRadiationField> field, std::shared_ptr<RadiationFieldMetadata> metadata, StoreVersion version)
{
	FieldStore::get_store_by(version)->serialize(stream, field, metadata);
}

std::shared_ptr<IRadiationField> FieldStore::load(const std::string& file)
{
	std::ifstream buffer = storage::open_file_for_reading(file);
	return FieldStore::get_store_by(buffer)->load(buffer);
}

std::shared_ptr<IRadiationField> FieldStore::load(std::istream& buffer)
{
	return FieldStore::get_store_by(buffer)->load(buffer);
}

std::shared_ptr<RadiationFieldMetadata> FieldStore::load_metadata(const std::string& file)
{
	std::ifstream buffer = storage::open_file_for_reading(file);
	return FieldStore::get_store_by(buffer)->load_metadata(buffer);
}

std::shared_ptr<RadiationFieldMetadata> FieldStore::load_metadata(std::istream& buffer)
{
	return FieldStore::get_store_by(buffer)->load_metadata(buffer);
}

const BasicFieldStore* radfiled3d::storage::FieldStore::get_store_by(std::istream& buffer)
{
	FieldStore::ensure_registered_stores();
	return Registry::get_store_by(FieldAccessor::getStoreVersion(buffer));
}

const BasicFieldStore* radfiled3d::storage::FieldStore::get_store_by(StoreVersion version)
{
	FieldStore::ensure_registered_stores();
	return Registry::get_store_by(version);
}

std::shared_ptr<VoxelLayer> FieldStore::load_single_layer(std::istream& buffer, const std::string& channel, const std::string& layer)
{
	return FieldStore::get_store_by(buffer)->load_single_layer(buffer, channel, layer);
}

void FieldStore::join(std::shared_ptr<IRadiationField> field, std::shared_ptr<RadiationFieldMetadata> metadata, const std::string& file, FieldJoinMode join_mode, FieldJoinCheckMode check_mode, StoreVersion fallback_version)
{
	FileLock file_lock(file, FieldStore::file_lock_synchronization);

	if (!fs::exists(file)) {
		FieldStore::ensure_registered_stores();

		FieldStore::store(field, metadata, file, Registry::get_highest_registered_version());
		return;
	}

	storage::v1::RadiationFieldMetadata& v1_metadata = dynamic_cast<storage::v1::RadiationFieldMetadata&>(*metadata);

	std::shared_ptr<IRadiationField> existing_field = FieldStore::load(file);
	std::shared_ptr<RadiationFieldMetadata> _target_metadata = FieldStore::peek_metadata(file);
	filed_types::v1::RadiationFieldMetadataHeader target_metadata = dynamic_cast<storage::v1::RadiationFieldMetadata&>(*_target_metadata).get_header();

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

	auto store = FieldStore::get_store_by(FieldStore::get_store_version(file));
	store->join(existing_field, field, join_mode, check_mode, ratio);

	target_metadata.simulation.primary_particle_count += v1_metadata.get_header().simulation.primary_particle_count;
	v1_metadata.set_header(target_metadata);
	store->store(existing_field, metadata, file);
}

void FieldStore::replace(std::shared_ptr<IRadiationField> field, std::shared_ptr<RadiationFieldMetadata> metadata, const std::string& file, StoreVersion version)
{
	FileLock file_lock(file, FieldStore::file_lock_synchronization);

	if (!fs::exists(file)) {
		FieldStore::store(field, metadata, file, version);
		return;
	}

	if (FieldStore::get_store_version(file) != version)
		throw RadiationFieldStoreException("The store version of file " + file + " differs from the requested one.");
	const auto* store = dynamic_cast<const storage::v1::FieldStore*>(FieldStore::get_store_by(version));
	if (store == nullptr)
		throw RadiationFieldStoreException("Replacing channels is only supported for files of store version V1: " + file);
	store->replace(field, metadata, file);
}

void storage::v1::FieldStore::replace(std::shared_ptr<IRadiationField> field, std::shared_ptr<storage::RadiationFieldMetadata> metadata, const std::string& file) const
{
	std::ifstream existing(file, std::ios::in | std::ios::binary);
	VersionHeader version_header;
	existing.read((char*)&version_header, sizeof(VersionHeader));
	const size_t metadata_size = storage::v1::RadiationFieldMetadata().get_metadata_size(existing);
	existing.seekg(sizeof(VersionHeader) + metadata_size, std::ios::beg);

	filed_types::v1::RadiationFieldHeader field_header;
	existing.read((char*)&field_header, sizeof(filed_types::v1::RadiationFieldHeader));
	const std::string field_type = field->get_typename();
	if (std::string(field_header.field_type, strnlen(field_header.field_type, sizeof(field_header.field_type))) != field_type)
		throw RadiationFieldStoreException("Field type mismatch! File " + file + " contains a " + std::string(field_header.field_type) + ", but the field is a " + field_type);

	std::string grid_header;
	if (field_type == "CartesianRadiationField") {
		filed_types::v1::CartesianHeader existing_grid;
		existing.read((char*)&existing_grid, sizeof(filed_types::v1::CartesianHeader));
		auto cartesian = std::dynamic_pointer_cast<CartesianRadiationField>(field);
		if (existing_grid.voxel_counts != cartesian->get_voxel_counts() || existing_grid.voxel_dimensions != cartesian->get_voxel_dimensions())
			throw RadiationFieldStoreException("Voxel grid mismatch! The field's voxel grid differs from the one in file " + file);
		grid_header.assign((const char*)&existing_grid, sizeof(filed_types::v1::CartesianHeader));
	}
	else if (field_type == "PolarRadiationField") {
		filed_types::v1::PolarHeader existing_grid;
		existing.read((char*)&existing_grid, sizeof(filed_types::v1::PolarHeader));
		auto polar = std::dynamic_pointer_cast<PolarRadiationField>(field);
		if (existing_grid.segments_counts != polar->get_segments_count())
			throw RadiationFieldStoreException("Segment count mismatch! The field's segments differ from the ones in file " + file);
		grid_header.assign((const char*)&existing_grid, sizeof(filed_types::v1::PolarHeader));
	}
	else {
		throw RadiationFieldStoreException("Field type " + field_type + " is not supported!");
	}
	if (!existing)
		throw RadiationFieldStoreException("File " + file + " is truncated or corrupted.");

	// channel name -> (header, offset of its data) for the channels of the file that are kept
	std::map<std::string, std::pair<filed_types::v1::ChannelHeader, std::streamoff>> kept_channels;
	while (true) {
		filed_types::v1::ChannelHeader channel_header;
		existing.read((char*)&channel_header, sizeof(filed_types::v1::ChannelHeader));
		if (existing.gcount() == 0 && existing.eof())
			break;
		if (!existing)
			throw RadiationFieldStoreException("File " + file + " is truncated or corrupted.");
		const std::string name(channel_header.name, strnlen(channel_header.name, sizeof(channel_header.name)));
		const std::streamoff data_offset = existing.tellg();
		if (!field->has_channel(name))
			kept_channels.insert({ name, { channel_header, data_offset } });
		existing.seekg(static_cast<std::streamoff>(channel_header.channel_bytes), std::ios::cur);
	}
	existing.clear();

	std::map<std::string, std::shared_ptr<VoxelBuffer>> new_channels;
	for (auto& channel : field->get_channels())
		new_channels.insert(channel);
	std::vector<std::string> channel_names;
	for (auto& [name, channel] : new_channels)
		channel_names.push_back(name);
	for (auto& [name, kept] : kept_channels)
		channel_names.push_back(name);
	std::sort(channel_names.begin(), channel_names.end());

	const std::string temporary_file = file + ".replace.tmp";
	{
		std::ofstream out(temporary_file, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open())
			throw RadiationFieldStoreException("Could not create temporary file " + temporary_file);
		out.write((const char*)&version_header, sizeof(VersionHeader));
		this->get_metadata_serializer().serializeMetadata(out, metadata);
		out.write((const char*)&field_header, sizeof(filed_types::v1::RadiationFieldHeader));
		out.write(grid_header.data(), grid_header.size());

		std::vector<char> copy_buffer(1 << 20);
		for (const std::string& name : channel_names) {
			auto new_channel = new_channels.find(name);
			if (new_channel != new_channels.end()) {
				filed_types::v1::ChannelHeader channel_header;
				std::strncpy(channel_header.name, name.c_str(), std::min<size_t>(sizeof(channel_header.name), name.length()));
				const std::string serialized = this->get_field_serializer().serializeChannel(new_channel->second)->str();
				channel_header.channel_bytes = serialized.length();
				out.write((const char*)&channel_header, sizeof(filed_types::v1::ChannelHeader));
				out.write(serialized.data(), serialized.length());
			}
			else {
				const auto& [channel_header, data_offset] = kept_channels.at(name);
				out.write((const char*)&channel_header, sizeof(filed_types::v1::ChannelHeader));
				existing.seekg(data_offset, std::ios::beg);
				size_t remaining = channel_header.channel_bytes;
				while (remaining > 0) {
					const size_t chunk = std::min(remaining, copy_buffer.size());
					existing.read(copy_buffer.data(), chunk);
					if (static_cast<size_t>(existing.gcount()) != chunk)
						throw RadiationFieldStoreException("File " + file + " is truncated or corrupted.");
					out.write(copy_buffer.data(), chunk);
					remaining -= chunk;
				}
			}
		}
		if (!out)
			throw RadiationFieldStoreException("Could not write temporary file " + temporary_file);
	}
	existing.close();
	fs::rename(temporary_file, file);
}

std::shared_ptr<storage::RadiationFieldMetadata> FieldStore::peek_metadata(const std::string& file)
{
	std::ifstream buffer = storage::open_file_for_reading(file);
	return FieldStore::peek_metadata(buffer);
}

std::shared_ptr<storage::RadiationFieldMetadata> FieldStore::peek_metadata(std::istream& buffer)
{
	return FieldStore::get_store_by(buffer)->peek_metadata(buffer);
}
