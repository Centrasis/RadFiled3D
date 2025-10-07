#pragma once
#include <glm/glm.hpp>
#include <string>
#include "RadFiled3D/VoxelBuffer.hpp"
#include <memory>
#include <stdexcept>

#ifdef __linux__
#define strncpy_s(dest, src, count) strncpy(dest, src, count)
#endif

namespace RadFiled3D {
	class RadiationFieldStoreException : public std::runtime_error {
	public:
		RadiationFieldStoreException(const std::string& message) : std::runtime_error("RadiationFieldStoreException: " + message) {}
	};

	enum class FieldType {
		Cartesian = 0,
		Polar = 1,
	};

	namespace Storage {
		class FieldStore;

		enum class StoreVersion {
			V1 = 0
		};

		namespace FiledTypes {
#pragma pack(push, 4)
			struct VersionHeader {
				char version[12] = { 0 };
			};
#pragma pack(pop)
			namespace V1 {
#pragma pack(push, 4)
				struct VoxelGridLayerHeader {
					char name[64] = { 0 };
					char unit[32] = { 0 };
					size_t bytes_per_element = 0;
					char dtype[32] = { 0 };
					float statistical_error = 0.f;
					size_t header_block_size = 0;
				};
#pragma pack(pop)

#pragma pack(push, 4)
				struct RadiationFieldHeader {
					char field_type[64] = { 0 };
				};
#pragma pack(pop)

#pragma pack(push, 4)
				struct CartesianHeader {
					glm::uvec3 voxel_counts = glm::uvec3(0);
					glm::vec3 voxel_dimensions = glm::vec3(0.f);
				};
#pragma pack(pop)

#pragma pack(push, 4)
				struct PolarHeader {
					glm::uvec2 segments_counts = glm::uvec3(0);
				};
#pragma pack(pop)

#pragma pack(push, 4)
				struct ChannelHeader {
					char name[64] = { 0 };
					size_t channel_bytes = 0;
				};
#pragma pack(pop)

#pragma pack(push, 4)
				struct RadiationFieldMetadataHeader {
					struct Simulation {
						uint64_t primary_particle_count = 0;
						char geometry[256] = { 0 };
						char physics_list[256] = { 0 };
						struct XRayTube {
							glm::vec3 radiation_direction = glm::vec3(0.f);
							glm::vec3 radiation_origin = glm::vec3(0.f);
							float max_energy_eV = 0.f;
							char tube_id[128] = { 0 };

							XRayTube() = default;
							XRayTube(const glm::vec3& radiation_direction, const glm::vec3& radiation_origin, float max_energy_eV, const std::string& tube_id) : radiation_direction(radiation_direction), radiation_origin(radiation_origin), max_energy_eV(max_energy_eV) {
								strncpy_s(this->tube_id, tube_id.c_str(), std::min<size_t>(sizeof(this->tube_id), tube_id.length()));
							}
						} tube;
						Simulation() = default;
						Simulation(size_t primary_particle_count, const std::string& geometry, const std::string& physics_list, const XRayTube& tube) : primary_particle_count(primary_particle_count), tube(tube) {
							strncpy_s(this->geometry, geometry.c_str(), std::min<size_t>(sizeof(this->geometry), geometry.length()));
							strncpy_s(this->physics_list, physics_list.c_str(), std::min<size_t>(sizeof(this->physics_list), physics_list.length()));
						}
					} simulation;
					struct Software {
						char name[128] = { 0 };
						char version[128] = { 0 };
						char repository[256] = { 0 };
						char commit[128] = { 0 };
						char doi[256] = { 0 };

						Software() = default;
						Software(const std::string& name, const std::string& version, const std::string& repository, const std::string& commit, const std::string& doi = "") {
							strncpy_s(this->name, name.c_str(), std::min<size_t>(sizeof(this->name), name.length()));
							strncpy_s(this->version, version.c_str(), std::min<size_t>(sizeof(this->version), version.length()));
							strncpy_s(this->repository, repository.c_str(), std::min<size_t>(sizeof(this->repository), repository.length()));
							strncpy_s(this->commit, commit.c_str(), std::min<size_t>(sizeof(this->commit), commit.length()));
							strncpy_s(this->doi, doi.c_str(), std::min<size_t>(sizeof(this->doi), doi.length()));
						}
					} software;

					RadiationFieldMetadataHeader(const RadiationFieldMetadataHeader::Simulation& simulation, const RadiationFieldMetadataHeader::Software& software) : simulation(simulation), software(software) { }

					RadiationFieldMetadataHeader() = default;
				};

				struct RadiationFieldMetadataHeaderBlock {
					size_t dynamic_metadata_size = 0;
				};
#pragma pack(pop)
			};
		};

		class RadiationFieldMetadata {
		protected:
			const StoreVersion version;
		public:
			RadiationFieldMetadata(StoreVersion version) : version(version) {}

			/** Gets the serialized size of the metadata in bytes */
			virtual size_t get_metadata_size(std::istream& stream) const = 0;

			/** Gets the version of the metadata */
			inline StoreVersion get_version() const {
				return this->version;
			}

			/** Serializes the metadata to a stream */
			virtual void serialize(std::ostream& stream) const = 0;

			/** Deserializes the metadata from a stream
			* stream The stream to deserialize from
			* quick_peek_only If true, only the metadata header will be deserialized, if applicable
			*/
			virtual void deserialize(std::istream& stream, bool quick_peek_only = false) = 0;
		};

		namespace V1 {
			class BinayFieldBlockHandler;

			class RadiationFieldMetadata : public RadFiled3D::Storage::RadiationFieldMetadata {
				friend class RadFiled3D::Storage::FieldStore;
			protected:
				FiledTypes::V1::RadiationFieldMetadataHeader header;
				std::shared_ptr<VoxelBuffer> dynamic_metadata;
				Storage::V1::BinayFieldBlockHandler* serializer;
			public:
				virtual size_t get_metadata_size(std::istream& stream) const override;

				RadiationFieldMetadata(const FiledTypes::V1::RadiationFieldMetadataHeader::Simulation& simulation, const FiledTypes::V1::RadiationFieldMetadataHeader::Software& software);

				RadiationFieldMetadata();

				~RadiationFieldMetadata();

				virtual void serialize(std::ostream& stream) const override;

				virtual void deserialize(std::istream& stream, bool quick_peek_only = false) override;

				const FiledTypes::V1::RadiationFieldMetadataHeader& get_header() const {
					return this->header;
				}

				void set_header(const FiledTypes::V1::RadiationFieldMetadataHeader& header) {
					this->header = header;
				}

				template<typename dtype = float>
				void add_dynamic_metadata(const std::string& key, const dtype& value) {
					this->dynamic_metadata->add_layer<dtype>(key, value);
				}

				template<typename VoxelT = ScalarVoxel<float>, typename dtype = float>
				void add_dynamic_metadata(const std::string& key, const VoxelT& voxel, const dtype& value) {
					this->dynamic_metadata->add_custom_layer<VoxelT, dtype>(key, voxel, value);
				}

				template<typename VoxelT = HistogramVoxel, typename dtype = float>
				void set_dynamic_custom_metadata(const std::string& key, const VoxelT& value) {
					this->dynamic_metadata->add_custom_layer<VoxelT, dtype>(key, value, dtype());
				}

				std::map<std::string, IVoxel*> get_dynamic_metadata() const {
					std::map<std::string, IVoxel*> metadata;
					for (auto& key : this->dynamic_metadata->get_layers()) {
						metadata[key] = &this->dynamic_metadata->get_voxel_flat(key, 0);
					}
					return metadata;
				}

				template<typename VoxelT = ScalarVoxel<float>>
				VoxelT& get_dynamic_metadata(const std::string& key) const {
					return this->dynamic_metadata->get_voxel_flat<VoxelT>(key, 0);
				}

				std::vector<std::string> get_dynamic_metadata_keys() const {
					return this->dynamic_metadata->get_layers();
				}
			};
		};
	}
};