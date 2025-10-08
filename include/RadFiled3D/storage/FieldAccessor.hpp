#pragma once
#include <RadFiled3D/RadiationField.hpp>
#include "RadFiled3D/storage/Types.hpp"
#include "RadFiled3D/storage/FieldSerializer.hpp"
#include <stdexcept>
#include <map>


namespace RadFiled3D {
	namespace Storage {
		class FieldAccessorBuilder;

		namespace AccessorTypes {
			typedef struct MemoryBlockDefinition {
				size_t offset;
				size_t size;

				MemoryBlockDefinition(size_t offset, size_t size)
					: offset(offset), size(size) {};

				MemoryBlockDefinition() {
					this->offset = 0;
					this->size = 0;
				}
			} MemoryBlockDefinition;

			typedef struct TypedMemoryBlockDefinition : public MemoryBlockDefinition {
			protected:
				std::vector<char> voxel_header_data;
			public:
				const char* get_voxel_header_data() const {
					return this->voxel_header_data.data();
				}

				size_t get_voxel_header_data_size() const {
					return this->voxel_header_data.size();
				}

				Typing::DType dtype;
				size_t elements_per_voxel;

				TypedMemoryBlockDefinition(size_t offset, size_t size, Typing::DType dtype, size_t elements_per_voxel)
					: MemoryBlockDefinition(offset, size), dtype(dtype), elements_per_voxel(elements_per_voxel) {};

				TypedMemoryBlockDefinition() : MemoryBlockDefinition() {
					this->dtype = Typing::DType::Byte;
					this->elements_per_voxel = 0;
				}

				void set_voxel_header_data(char* data, size_t size) {
					voxel_header_data.resize(size);
					memcpy(this->voxel_header_data.data(), data, size);
				}
			} TypedMemoryBlockDefinition;

			typedef struct ChannelStructure {
				MemoryBlockDefinition channel_block;
				std::map<std::string, TypedMemoryBlockDefinition> layers;

				ChannelStructure(MemoryBlockDefinition channel_block, const std::map<std::string, TypedMemoryBlockDefinition>& layers)
					: channel_block(channel_block), layers(layers) {};

				ChannelStructure() : channel_block(), layers() {};
			} ChannelStructure;
		}

		/** A class to access fields from a buffer
		* Descadent classes should implement the accessField method and should provide all needed methods to access the fields as a whole, by channel, by layer and by voxel.
		* The accessField method should return a shared pointer to the field, and the other methods should return shared pointers to the corresponding objects.
		* The FieldAccessor will be initialized from a template buffer in order to speed up loading of subsequent fields of the same structure e.g. files from the same dataset.
		* A new FieldAccessor requires adding FieldAccessorBuilder as a friend class, as FieldAccessors are created through it.
		*/
		class FieldAccessor {
			friend class RadFiled3D::Storage::FieldAccessorBuilder;
		public:
#pragma pack(push, 4)
			struct SerializationData {
				StoreVersion store_version;
				FieldType field_type;
				size_t metadata_fileheader_size;
				size_t voxel_count;

				virtual ~SerializationData() {}

				SerializationData(StoreVersion store_version, FieldType field_type, size_t metadata_fileheader_size, size_t voxel_count)
					: store_version(store_version), field_type(field_type), metadata_fileheader_size(metadata_fileheader_size), voxel_count(voxel_count) {};

				SerializationData() {
					this->store_version = StoreVersion::V1;
					this->field_type = FieldType::Cartesian;
					this->metadata_fileheader_size = 0;
					this->voxel_count = 0;
				}

				virtual std::vector<char> serialize_additional_data() const {
					return std::vector<char>();
				}

				virtual void deserialize_additional_data(const std::vector<char>& data) { }
			};
#pragma pack(pop)
		private:
			const FieldType field_type;
		protected:
			size_t metadata_fileheader_size;
			size_t voxel_count = 0;
			StoreVersion store_version;

			/** Verify the buffer and set the read position to the beginning of the field.
			* @param buffer The buffer to verify
			* @throws RadiationFieldStoreException if the buffer is not valid
			*/
			void verifyBuffer(std::istream& buffer) const;

			/** Construct a field accessor with the given metadata size.
			* @param metadata_fileheader_size The size of the metadata in the file header
			* @param field_type The type of the field
			*/
			FieldAccessor(FieldType field_type);

			/** Initializes the accessor from a template buffer. */
			virtual void initialize(std::istream& buffer) = 0;
		public:
			virtual ~FieldAccessor() {};

			static std::vector<char> Serialize(const FieldAccessor* accessor);

			static std::shared_ptr<FieldAccessor> Deserialize(const std::vector<char>& buffer);

			/** Access a field from a buffer and return a shared pointer to it
			* @param buffer The buffer to access the field from
			* @return A shared pointer to the field
			*/
			virtual std::shared_ptr<IRadiationField> accessField(std::istream& buffer) const = 0;

			/** Get the version of the store that created a file
			* @param file The file to get the store version from
			* @return The store version of the file
			*/
			static StoreVersion getStoreVersion(std::istream& buffer);

			/** Quickly peeks at the mandatory metadata header of the radiation field from a buffer. Resets the file stream to the beginning
			* @param buffer The buffer to get the metadata from
			* @return The metadata header of the radiation field
			*/
			static FieldType peekFieldType(std::istream& file_stream);

			inline FieldType getFieldType() const {
				return this->field_type;
			}

			inline size_t getMetadataFileheaderOffset() const {
				return this->metadata_fileheader_size;
			}

			inline size_t getVoxelCount() const {
				return this->voxel_count;
			}

			/** Returns the offset from the beginning of a file to the start of the actual field data starting with the first channel block
			* @return The offset from the beginning of the file to the start of the field data
			*/
			virtual size_t getFieldDataOffset() const = 0;

			/** Accesses a voxel from a buffer and returns a pointer to it
			* @param buffer The buffer to access the voxel from
			* @param channel_name The name of the channel the voxel is in
			* @param layer_name The name of the layer the voxel is in
			* @param voxel_idx The index of the voxel in the layer
			* @return A pointer to the voxel
			*/
			virtual IVoxel* accessVoxelRawFlat(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, size_t voxel_idx) const = 0;

			/** Accesses a set of voxels from a buffer and returns a vector of pointers to them
			* @param buffer The buffer to access the voxels from
			* @param channel_name The name of the channel the voxels are in
			* @param layer_name The name of the layer the voxels are in
			* @param voxel_indices The indices of the voxels in the layer
			* @return A vector of pointers to the voxels
			*/
			virtual std::vector<IVoxel*> accessVoxelsRawFlat(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const std::vector<size_t>& voxel_indices) const = 0;

			/** Accesses a channel from a buffer and returns a shared pointer to it
			* @param buffer The buffer to access the channel from
			* @param channel_name The name of the channel to access
			* @return A shared pointer to the channel
			*/
			template<typename dtype, typename VoxelT = ScalarVoxel<dtype>>
			std::shared_ptr<VoxelT> accessVoxelFlat(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, size_t voxel_idx) const {
				IVoxel* voxel = this->accessVoxelRawFlat(buffer, channel_name, layer_name, voxel_idx);
				return std::shared_ptr<VoxelT>((VoxelT*)voxel);
			};

			template<typename dtype, typename VoxelT = ScalarVoxel<dtype>>
			std::vector<std::shared_ptr<VoxelT>> accessVoxelsFlat(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const std::vector<size_t>& voxel_indices) const {
				std::vector<IVoxel*> voxels_raw = this->accessVoxelsRawFlat(buffer, channel_name, layer_name, voxel_indices);
				std::vector<std::shared_ptr<VoxelT>> voxels(voxels_raw.size());
				for (size_t i = 0; i < voxels_raw.size(); ++i) {
					voxels[i] = std::shared_ptr<VoxelT>((VoxelT*)voxels_raw[i]);
				}
				return voxels;
			};

			virtual SerializationData* generateSerializationBuffer() const = 0;
		};

		class CartesianFieldAccessor : virtual public RadFiled3D::Storage::FieldAccessor {
		public:
#pragma pack(push, 4)
			struct SerializationData : public FieldAccessor::SerializationData {
				glm::vec3 field_dimensions;
				glm::vec3 voxel_dimensions;

				SerializationData(StoreVersion store_version, FieldType field_type, size_t metadata_fileheader_size, size_t voxel_count, glm::vec3 field_dimensions, glm::vec3 voxel_dimensions)
					: FieldAccessor::SerializationData(store_version, field_type, metadata_fileheader_size, voxel_count), field_dimensions(field_dimensions), voxel_dimensions(voxel_dimensions) {};

				SerializationData() : FieldAccessor::SerializationData() {
					this->field_dimensions = glm::vec3(0.0f);
					this->voxel_dimensions = glm::vec3(0.0f);
				}
			};
#pragma pack(pop)
		protected:
			glm::vec3 field_dimensions;
			glm::vec3 voxel_dimensions;

			CartesianFieldAccessor(FieldType field_type) : FieldAccessor(field_type), field_dimensions(glm::vec3(0.0f)), voxel_dimensions(glm::vec3(0.0f)) { };
		public:
			virtual ~CartesianFieldAccessor() {};

			virtual IVoxel* accessVoxelRaw(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec3& voxel_idx) const = 0;
			virtual IVoxel* accessVoxelRawByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec3& voxel_pos) const = 0;

			/** access a channel from a buffer and return a shared pointer to it
			* @param buffer The buffer to access the channel from
			* @param channel_name The name of the channel to access
			* @return A shared pointer to the channel container
			*/
			virtual std::shared_ptr<VoxelGridBuffer> accessChannel(std::istream& buffer, const std::string& channel_name) const = 0;

			/** access a layer from a buffer and return a shared pointer to it
			* @param buffer The buffer to access the layer from
			* @param channel_name The name of the channel the layer is in
			* @param layer_name The name of the layer to access
			* @return A shared pointer to the layer
			*/
			virtual std::shared_ptr<VoxelGrid> accessLayer(std::istream& buffer, const std::string& channel_name, const std::string& layer_name) const = 0;

			/** access all channels of a layer from a buffer and return a shared pointer to it for each channel.
			* @param buffer The buffer to access the layer from
			* @param layer_name The name of the layer to access
			* @return A shared pointer to a map containing all instances of the desired layer for each channel.
			*/
			virtual std::map<std::string, std::shared_ptr<VoxelGrid>> accessLayerAcrossChannels(std::istream& buffer, const std::string& layer_name) const = 0;

			template<typename dtype, typename VoxelT = ScalarVoxel<dtype>>
			std::shared_ptr<VoxelT> accessVoxel(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec3& voxel_idx) const {
				IVoxel* voxel = this->accessVoxelRaw(buffer, channel_name, layer_name, voxel_idx);
				return std::shared_ptr<VoxelT>((VoxelT*)voxel);
			};

			template<typename dtype, typename VoxelT = ScalarVoxel<dtype>>
			std::shared_ptr<VoxelT> accessVoxelByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec3& voxel_pos) const {
				auto voxel = this->accessVoxelRawByCoord(buffer, channel_name, layer_name, voxel_pos);
				return std::shared_ptr<VoxelT>((VoxelT*)voxel);
			};

			virtual SerializationData* generateSerializationBuffer() const override {
				return new SerializationData(this->store_version, this->getFieldType(), this->metadata_fileheader_size, this->voxel_count, this->field_dimensions, this->voxel_dimensions);
			}
		};

		class PolarFieldAccessor : virtual public RadFiled3D::Storage::FieldAccessor {
		public:
#pragma pack(push, 4)
			struct SerializationData : public FieldAccessor::SerializationData {
				glm::uvec2 segments_counts;

				SerializationData(StoreVersion store_version, FieldType field_type, size_t metadata_fileheader_size, size_t voxel_count, glm::uvec2 segments_counts)
					: FieldAccessor::SerializationData(store_version, field_type, metadata_fileheader_size, voxel_count), segments_counts(segments_counts) {};

				SerializationData() : FieldAccessor::SerializationData() {
					this->segments_counts = glm::uvec2(0);
				}
			};
#pragma pack(pop)
		protected:
			glm::uvec2 segments_counts;

			PolarFieldAccessor(FieldType field_type) : FieldAccessor(field_type), segments_counts(glm::uvec2(0)) { };

		public:
			virtual IVoxel* accessVoxelRaw(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec2& voxel_idx) const = 0;
			virtual IVoxel* accessVoxelRawByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec2& voxel_pos) const = 0;

			virtual ~PolarFieldAccessor() {};

		public:
			/** access a layer from a buffer and return a shared pointer to it
			* @param buffer The buffer to access the layer from
			* @param channel_name The name of the channel the layer is in
			* @param layer_name The name of the layer to access
			* @return A shared pointer to the layer
			*/
			virtual std::shared_ptr<PolarSegments> accessLayer(std::istream& buffer, const std::string& channel_name, const std::string& layer_name) const = 0;

			template<typename dtype, typename VoxelT = ScalarVoxel<dtype>>
			std::shared_ptr<VoxelT> accessVoxel(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec2& voxel_idx) const {
				IVoxel* voxel = this->accessVoxelRaw(buffer, channel_name, layer_name, voxel_idx);
				return std::shared_ptr<VoxelT>((VoxelT*)voxel);
			};

			template<typename dtype, typename VoxelT = ScalarVoxel<dtype>>
			std::shared_ptr<VoxelT> accessVoxelByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec2& voxel_pos) const {
				auto voxel = this->accessVoxelRawByCoord(buffer, channel_name, layer_name, voxel_pos);
				return std::shared_ptr<VoxelT>((VoxelT*)voxel);
			};

			virtual SerializationData* generateSerializationBuffer() const override {
				return new SerializationData(this->store_version, this->getFieldType(), this->metadata_fileheader_size, this->voxel_count, this->segments_counts);
			}
		};

		namespace V1 {
			class FileParser : virtual public RadFiled3D::Storage::FieldAccessor {
			public:
				static std::vector<char> SerializeChannelsLayersOffsets(const std::map<std::string, AccessorTypes::ChannelStructure>& channels_layers_offsets);
				static std::map<std::string, AccessorTypes::ChannelStructure> DeserializeChannelsLayersOffsets(const std::vector<char>& data);

			protected:
				std::unique_ptr<RadFiled3D::Storage::V1::BinayFieldBlockHandler> serializer;

				FileParser(FieldType field_type) : FieldAccessor(field_type) {}
				

				/** Initialized the channels and layers structure of version 1 files.
				* Needs to be overridden by the different field shapes and this base method needs to be called in the overridden method right after parsing the channel header.
				*/
				virtual void initialize(std::istream& buffer) override;

				std::map<std::string, AccessorTypes::ChannelStructure> channels_layers_offsets;
			public:
				virtual IVoxel* accessVoxelRawFlat(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, size_t voxel_idx) const override;
				virtual std::vector<IVoxel*> accessVoxelsRawFlat(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const std::vector<size_t>& voxel_indices) const override;

				IVoxel* createVoxelFromBuffer(char* buffer, Typing::DType dtype, const char* voxel_header_data = nullptr) const;
			};


			class CartesianFieldAccessor : public RadFiled3D::Storage::CartesianFieldAccessor, public FileParser {
			public:
#pragma pack(push, 4)
				struct SerializationData : public RadFiled3D::Storage::CartesianFieldAccessor::SerializationData {
					std::map<std::string, AccessorTypes::ChannelStructure> channels_layers_offsets;

					SerializationData(StoreVersion store_version, FieldType field_type, size_t metadata_fileheader_size, size_t voxel_count, glm::vec3 field_dimensions, glm::vec3 voxel_dimensions, std::map<std::string, AccessorTypes::ChannelStructure> channels_layers_offsets)
						: RadFiled3D::Storage::CartesianFieldAccessor::SerializationData(store_version, field_type, metadata_fileheader_size, voxel_count, field_dimensions, voxel_dimensions), channels_layers_offsets(channels_layers_offsets) {};

					SerializationData() : RadFiled3D::Storage::CartesianFieldAccessor::SerializationData() {
						this->channels_layers_offsets = std::map<std::string, AccessorTypes::ChannelStructure>();
					}

					virtual std::vector<char> serialize_additional_data() const override;

					virtual void deserialize_additional_data(const std::vector<char>& data) override;
				};
#pragma pack(pop)
				friend class RadFiled3D::Storage::FieldAccessorBuilder;
			protected:
				std::unique_ptr<VoxelGrid> default_grid;

				CartesianFieldAccessor();
				virtual void initialize(std::istream& buffer) override;
			public:
				CartesianFieldAccessor(const SerializationData& data);

				virtual IVoxel* accessVoxelRaw(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec3& voxel_idx) const override;
				virtual IVoxel* accessVoxelRawByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec3& voxel_pos) const override;

				virtual std::shared_ptr<IRadiationField> accessField(std::istream& buffer) const override;
				virtual std::shared_ptr<VoxelGridBuffer> accessChannel(std::istream& buffer, const std::string& channel_name) const override;
				virtual std::shared_ptr<VoxelGrid> accessLayer(std::istream& buffer, const std::string& channel_name, const std::string& layer_name) const override;
				virtual std::map<std::string, std::shared_ptr<VoxelGrid>> accessLayerAcrossChannels(std::istream& buffer, const std::string& layer_name) const override;

				virtual size_t getFieldDataOffset() const override;
				virtual SerializationData* generateSerializationBuffer() const override {
					return new SerializationData(this->store_version, this->getFieldType(), this->metadata_fileheader_size, this->voxel_count, this->field_dimensions, this->voxel_dimensions, this->channels_layers_offsets);
				}


				virtual ~CartesianFieldAccessor() {};
			};

			class PolarFieldAccessor : public RadFiled3D::Storage::PolarFieldAccessor, public FileParser {
			public:
#pragma pack(push, 4)
				struct SerializationData : public RadFiled3D::Storage::PolarFieldAccessor::SerializationData {
					std::map<std::string, AccessorTypes::ChannelStructure> channels_layers_offsets;

					SerializationData(StoreVersion store_version, FieldType field_type, size_t metadata_fileheader_size, size_t voxel_count, glm::uvec2 segments_counts, std::map<std::string, AccessorTypes::ChannelStructure> channels_layers_offsets)
						: RadFiled3D::Storage::PolarFieldAccessor::SerializationData(store_version, field_type, metadata_fileheader_size, voxel_count, segments_counts), channels_layers_offsets(channels_layers_offsets) {};

					SerializationData() : RadFiled3D::Storage::PolarFieldAccessor::SerializationData() {
						this->channels_layers_offsets = std::map<std::string, AccessorTypes::ChannelStructure>();
					}

					virtual std::vector<char> serialize_additional_data() const override {
						return FileParser::SerializeChannelsLayersOffsets(this->channels_layers_offsets);
					}

					virtual void deserialize_additional_data(const std::vector<char>& data) override {
						this->channels_layers_offsets = FileParser::DeserializeChannelsLayersOffsets(data);
					}
				};
#pragma pack(pop)
				friend class RadFiled3D::Storage::FieldAccessorBuilder;
			protected:
				std::unique_ptr<PolarSegments> default_segments;

				PolarFieldAccessor();
				virtual void initialize(std::istream& buffer) override;
			public:
				PolarFieldAccessor(const SerializationData& data);

				virtual IVoxel* accessVoxelRaw(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec2& voxel_idx) const override;
				virtual IVoxel* accessVoxelRawByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec2& voxel_pos) const override;

				virtual std::shared_ptr<IRadiationField> accessField(std::istream& buffer) const override;
				virtual std::shared_ptr<PolarSegments> accessLayer(std::istream& buffer, const std::string& channel_name, const std::string& layer_name) const override;

				virtual size_t getFieldDataOffset() const override;

				virtual SerializationData* generateSerializationBuffer() const override {
					return new SerializationData(this->store_version, this->getFieldType(), this->metadata_fileheader_size, this->voxel_count, this->segments_counts, this->channels_layers_offsets);
				}

				virtual ~PolarFieldAccessor() {};
			};
		};

		class FieldAccessorBuilder {
		public:
			/** Construct a field accessor from a buffer
			* @param buffer The buffer to construct the field accessor from
			* @return The field accessor
			*/
			static std::shared_ptr<FieldAccessor> Construct(std::istream& buffer);
		};
	};
}
