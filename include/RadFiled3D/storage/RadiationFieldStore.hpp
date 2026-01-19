#pragma once
#include <string>
#include <memory>
#include <glm/vec3.hpp>
#include <cstring>
#include <stdexcept>
#include "RadFiled3D/RadiationField.hpp"
#include "RadFiled3D/storage/Types.hpp"
#include <utility>
#include <RadFiled3D/storage/FieldAccessor.hpp>
#include <RadFiled3D/storage/MetadataSerializer.hpp>
#include <RadFiled3D/storage/MetadataAccessor.hpp>
#include <RadFiled3D/storage/FieldSerializer.hpp>


namespace RadFiled3D {
	namespace Storage {
		/** The mode how to join two radiation fields.
		* Identity: Use the value of the target field
		* Add: Add the values of the target and the additional source field
		* Mean: Calculate the mean of the values of the target and the additional source field
		* Subtract: Subtract the values of the additional source field from the target field
		* Devide: Devide the values of the target field by the values of the additional source field
		* Multiply: Multiply the values of the target field by the values of the additional source field
		*/
		enum class RADFILED_API FieldJoinMode {
			/* Use the value of the target field */
			Identity = 0,
			/* Add the values of the target and the additional source field */
			Add = 1,
			/* Calculate the mean of the values of the target and the additional source field */
			Mean = 2,
			/* Subtract the values of the additional source field from the target field */
			Subtract = 3,
			/* Divide the values of the target field by the values of the additional source field */
			Divide = 4,
			/* Multiply the values of the target field by the values of the additional source field */
			Multiply = 5,
			/* Add the values of the target and the additional source field with a weighting ratio based on the primary particles */
			AddWeighted = 6
		};

		/** The mode to join the fields.
		* All modes stack on each other. So Strict includes all checks of higher modes. The following mode includes all modes except for Strict and so on.
		* The modes are as follows:
		* Strict: Check if the metadata is equal. If not, throw an exception
		* MetadataSimulationSimilar: Check if the metadata is similar (e.g. Geometry, Radiation-Direction, xray-tube). If not, throw an exception
		* MetadataSoftwareEqual: Check if the software metadata is equal. If not, throw an exception
		* MetadataSoftwareSimilar: Check if the software metadata is similar (e.g. Software-Name, Software-Repository). If not, throw an exception
		* FieldStructureOnly: Check if the fields share the same channel-layer structure. If not, throw an exception
		* FieldUnitsOnly: Check if the fields layers share the same units. If not, throw an exception
		* NoChecks: Do not perform any semantic checks. Technical checks will still be performed
		*/
		enum class RADFILED_API FieldJoinCheckMode {
			/* Check if the metadata and field structure is equal. If not, throw an exception */
			Strict = 0,
			/* Check if the metadata is similar (e.g. Geometry, Radiation-Direction, xray-tube). If not, throw an exception */
			MetadataSimulationSimilar = 1,
			/* Check if the software metadata is equal. If not, throw an exception */
			MetadataSoftwareEqual = 2,
			/* Check if the software metadata is similar (e.g. Software-Name, Software-Repository). If not, throw an exception */
			MetadataSoftwareSimilar = 3,
			/* Check if the fields share the same channel-layer structure. If not, throw an exception */
			FieldStructureOnly = 4,
			/* Check if the fields layers share the same units. If not, throw an exception */
			FieldUnitsOnly = 5,
			/* Do not perform any semantic checks. Technical checks will still be performed */
			NoChecks = 6
		};

		class RADFILED_API ExporterHelpers {
		public:
			/** Perform the actual merge of the fields
			* @param target The target field
			* @param additional_source The additional source field
			* @param mode The mode to join the fields
			*/
			template<typename FieldT>
			static void ensure_channels(std::shared_ptr<FieldT> target, std::shared_ptr<FieldT> additional_source) {
				for (auto& channel : additional_source->get_channels()) {
					if (!target->has_channel(channel.first)) {
						target->add_channel(channel.first);
					}
				}
			}

			/** Generate a Voxel-level join function (a, b) -> c with a beeing the target voxel, b beeing the additional source voxel and c beeing the result voxel
			* @param mode The mode to join the fields
			* @param ratio The ratio to use for the weighted join mode. Default is 0.f meaning only the data of the target voxel is used.
			* @return The join function
			* @tparam dtype The data type of the field
			* @tparam scalarT The scalar type of the field
			*/
			template<typename dtype, typename scalarT = dtype>
			static std::function<dtype(const dtype&, const dtype&)> get_join_function(FieldJoinMode mode, float ratio = 0.f) {
				switch (mode)
				{
				case FieldJoinMode::Add:
					return [](const dtype& a, const dtype& b) { return a + b; };
				case FieldJoinMode::Mean:
					return [](const dtype& a, const dtype& b) { return (a + b) / static_cast<scalarT>(2); };
				case FieldJoinMode::Identity:
					return [](const dtype& a, const dtype& b) { return a; };
				case FieldJoinMode::Subtract:
					return [](const dtype& a, const dtype& b) { return a - b; };
				case FieldJoinMode::Divide:
					return [](const dtype& a, const dtype& b) { return a / b; };
				case FieldJoinMode::Multiply:
					return [](const dtype& a, const dtype& b) { return a * b; };
				case FieldJoinMode::AddWeighted:
					return [ratio](const dtype& a, const dtype& b) { return static_cast<dtype>((a * (1.f - ratio)) + (b * ratio)); };
				default:
					throw RadiationFieldStoreException("Unknown join mode");
				}
			}
		};

		class RADFILED_API IRadiationFieldExporter {
		public:
			/** Store the radiation field to a file
			* @param field The radiation field to store
			* @param metadata The metadata of the radiation field
			* @param file The file to store the radiation field to
			*/
			void store(std::shared_ptr<IRadiationField> field, std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata, const std::string& file) const;

			/** Serialize the radiation field to a stream
			* @param stream The stream to serialize the radiation field to
			* @param field The radiation field to serialize
			* @param metadata The metadata of the radiation field
			*/
			virtual void serialize(std::ostream& stream, std::shared_ptr<IRadiationField> field, std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata) const = 0;

			/** Merge the radiation field to the one of an existing
			* @param target The radiation field to join to
			* @param additional_source The radiation field to join from
			* @param metadata The metadata of the radiation field
			* @param join_mode The mode to join the fields
			* @param check_mode The mode to check the fields
			* @param ratio The ratio to use for the weighted join mode. Default is 0.f meaning only the data of the target field is used.
			*/
			virtual void join(std::shared_ptr<IRadiationField> target, std::shared_ptr<IRadiationField> additional_source, FieldJoinMode join_mode, FieldJoinCheckMode check_mode, float ratio = 0.f) const = 0;
		};

		class RADFILED_API IRadiationFieldImporter {
		public:
			/** Load the radiation field from a file
			* @param file The file to load the radiation field from
			* @return The radiation field
			* @param file The file to load the radiation field from
			* @return The radiation field
			* @throw RadiationFieldStoreException If the file does not exist or the file is corrupted
			*/
			std::shared_ptr<IRadiationField> load(const std::string& file) const;

			/** Fully retrieves the metadata of the radiation field from a file
			* @param file The file to get the metadata from
			* @return The metadata of the radiation field
			* @throw RadiationFieldStoreException If the file does not exist or the file is corrupted
			*/
			std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> load_metadata(const std::string& file) const;

			/** Quickly peeks at the mandatory metadata header of the radiation field from a file
			* @param file The file to get the metadata from
			* @return The metadata header of the radiation field
			* @throw RadiationFieldStoreException If the file does not exist or the file is corrupted
			*/
			std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> peek_metadata(const std::string& file) const;

			/** Load the radiation field from a buffer
			* @param buffer The buffer to load the radiation field from
			* @return The radiation field
			* @throw RadiationFieldStoreException If the buffer is corrupted
			*/
			virtual std::shared_ptr<IRadiationField> load(std::istream& buffer) const = 0;

			/** Fully retrieves the metadata of the radiation field from a buffer
			* @param buffer The buffer to get the metadata from
			* @return The metadata of the radiation field
			* @throw RadiationFieldStoreException If the buffer is corrupted
			*/
			virtual std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> load_metadata(std::istream& buffer) const = 0;

			/** Quickly peeks at the mandatory metadata header of the radiation field from a buffer
			* @param buffer The buffer to get the metadata from
			* @return The metadata header of the radiation field
			* @throw RadiationFieldStoreException If the buffer is corrupted
			*/
			virtual std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> peek_metadata(std::istream& buffer) const = 0;

			/** Load a single layer from a buffer without loding the entire radiation field
			* @param buffer The buffer to load the radiation field from
			* @return The radiation field
			* @throw RadiationFieldStoreException If the buffer is corrupted
			*/
			virtual std::shared_ptr<VoxelLayer> load_single_layer(std::istream& buffer, const std::string& channel, const std::string& layer) const = 0;

			/** Peeks at the field type of the radiation field from a buffer
			* @param buffer The buffer to get the metadata from
			* @return The field type of the radiation field
			* @throw RadiationFieldStoreException If the buffer is corrupted
			*/
			virtual FieldType peek_field_type(std::istream& file_stream) const = 0;
		};

		class RADFILED_API BasicFieldStore : public IRadiationFieldExporter, public IRadiationFieldImporter {
		private:
			std::string file_version;
			RadFiled3D::Storage::MetadataSerializer* metadata_serializer;
			RadFiled3D::Storage::BinayFieldBlockHandler* field_serializer;
			RadFiled3D::Storage::MetadataAccessor* metadata_accessor;

		protected:
			BasicFieldStore(
				const std::string& file_version,
				RadFiled3D::Storage::MetadataSerializer* metadata_serializer,
				RadFiled3D::Storage::BinayFieldBlockHandler* field_serializer,
				RadFiled3D::Storage::MetadataAccessor* metadata_accessor
			) : file_version(file_version),
				metadata_serializer(metadata_serializer),
				field_serializer(field_serializer),
				metadata_accessor(metadata_accessor)
			{}

			~BasicFieldStore() {
				delete this->metadata_serializer;
				delete this->field_serializer;
				delete this->metadata_accessor;
			}

			inline MetadataSerializer& get_metadata_serializer() const {
				return *this->metadata_serializer;
			}

			inline MetadataAccessor& get_metadata_accessor() const {
				return *this->metadata_accessor;
			}

			inline BinayFieldBlockHandler& get_field_serializer() const {
				return *this->field_serializer;
			}

		public:
			/** Serialize the radiation field to a stream
			* @param stream The stream to serialize the radiation field to
			* @param field The radiation field to serialize
			* @param metadata The metadata of the radiation field
			*/
			virtual void serialize(std::ostream& stream, std::shared_ptr<IRadiationField> field, std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata) const override;

			virtual void valdiate_file_version(std::istream& stream) const;
			virtual std::shared_ptr<IRadiationField> load(std::istream& buffer) const override;

			/** Fully retrieves the metadata of the radiation field from a file
			* @param buffer The buffer to get the metadata from
			* @return The metadata of the radiation field
			* @throw RadiationFieldStoreException If the file does not exist or the file is corrupted
			*/
			virtual std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> load_metadata(std::istream& buffer) const override;

			/** Quickly peeks at the mandatory metadata header of the radiation field from a file
			* @param file The file to get the metadata from
			* @return The metadata header of the radiation field
			* @throw RadiationFieldStoreException If the file does not exist or the file is corrupted
			*/
			virtual std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> peek_metadata(std::istream& buffer) const override;

			FieldType peek_field_type(std::istream& file_stream) const override;
		};

		namespace V1 {
			class RADFILED_API FieldStore : public BasicFieldStore {
			public:
				FieldStore() : BasicFieldStore(
					"1.0",
					new V1::MetadataSerializer(),
					(RadFiled3D::Storage::BinayFieldBlockHandler*)new RadFiled3D::Storage::V1::BinayFieldBlockHandler(),
					new V1::MetadataAccessor()
				) {}

				/** Merge the radiation field to the one of an existing
				* @param target The radiation field to join to
				* @param additional_source The radiation field to join from
				* @param metadata The metadata of the radiation field
				* @param mode The mode to join the fields
				*/
				virtual void join(std::shared_ptr<IRadiationField> target, std::shared_ptr<IRadiationField> additional_source, FieldJoinMode join_mode, FieldJoinCheckMode check_mode, float ratio = 0.f) const override;

				/** Load a single layer from a buffer without loding the entire radiation field
				* @param buffer The buffer to load the radiation field from
				* @return The radiation field
				* @throw RadiationFieldStoreException If the buffer is corrupted
				*/
				virtual std::shared_ptr<VoxelLayer> load_single_layer(std::istream& buffer, const std::string& channel, const std::string& layer) const override;
			};
		};

		/** This class should be used to accutally store and load radiation fields.
		* It will automatically detect the version of the file and use the correct store to load the radiation field.
		* When using the same versions multiple times, the class will cache the store to avoid unnecessary reinitialization.
		*/
		class RADFILED_API FieldStore {
		protected:
			static std::shared_ptr<BasicFieldStore> store_instance;
			static StoreVersion store_version;
			static bool file_lock_syncronization;
		public:
			/** Enable or disable file transaction synchronization. This will make sure, that only one process can perform transactions such as joining on a file at a time and that other processes are queued.
			* Default is disabled
			* @param enable Enable or disable the synchronization
			*/
			[[deprecated("This feature is highly experimental and not tested on platforms!")]]
			static void enable_file_lock_syncronization(bool enable) {
				FieldStore::file_lock_syncronization = enable;
			}

			/** Initialize the store instance. Optional: Will be called on load and store operations if not called manually.
			* @param version The version of the store to use
			*/
			static void init_store_instance(StoreVersion version);

			/** Get the version of the store that created a buffer
			* @param buffer The buffer to get the store version from
			* @return The store version of the buffer
			* @note The buffer will be reset to the beginning
			*/
			static StoreVersion get_store_version(const std::string& file);
			static StoreVersion get_store_version(std::istream& buffer);

			/** Store the radiation field to a file
			* @param field The radiation field to store
			* @param metadata The metadata of the radiation field
			* @param file The file to store the radiation field to
			* @param version The version of the store to use
			*/
			static void store(std::shared_ptr<IRadiationField> field, std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata, const std::string& file, StoreVersion version = StoreVersion::V1);

			/** Serialize the radiation field to a stream
			* @param stream The stream to serialize the radiation field to
			* @param field The radiation field to serialize
			* @param metadata The metadata of the radiation field
			* @param version The version of the store to use
			*/
			static void serialize(std::ostream& stream, std::shared_ptr<IRadiationField> field, std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata, StoreVersion version = StoreVersion::V1);

			/** Load the radiation field from a file
			* @param file The file to load the radiation field from
			* @return The radiation field
			*/
			static std::shared_ptr<IRadiationField> load(const std::string& file);

			/** Load the radiation field from a buffer
			* @param buffer The buffer to load the radiation field from
			* @return The radiation field
			*/
			static std::shared_ptr<IRadiationField> load(std::istream& buffer);

			/** Fully retrieves the metadata of the radiation field from a file
			* @param file The file to get the metadata from
			* @return The metadata of the radiation field
			*/
			static std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> load_metadata(const std::string& file);

			/** Fully retrieves the metadata of the radiation field from a buffer
			* @param buffer The buffer to get the metadata from
			* @return The metadata of the radiation field
			*/
			static std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> load_metadata(std::istream& buffer);

			/** Quickly peeks at the mandatory metadata header of the radiation field from a file
			* @param file The file to get the metadata from
			* @return The metadata header of the radiation field
			*/
			static std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> peek_metadata(const std::string& file);

			/** Quickly peeks at the mandatory metadata header of the radiation field from a buffer. Resets the file stream to the beginning
			* @param buffer The buffer to get the metadata from
			* @return The metadata header of the radiation field
			*/
			static FieldType peek_field_type(std::istream& file_stream);

			/** Loads a single layer from a buffer without loding the entire radiation field
			* @param buffer The buffer to load the radiation field from
			* @return The radiation field
			* @throw RadiationFieldStoreException If the buffer is corrupted
			*/
			static std::shared_ptr<VoxelLayer> load_single_layer(std::istream& buffer, const std::string& channel, const std::string& layer);

			/** Quickly peeks at the mandatory metadata header of the radiation field from a buffer
			* @param buffer The buffer to get the metadata from
			* @return The metadata header of the radiation field
			*/
			static std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> peek_metadata(std::istream& buffer);

			/** Merge the radiation field to the one of an existing file
			* Creates a new stored radiation field if no radiation field was present at the file path.
			* @param field The radiation field to join
			* @param metadata The metadata of the radiation field
			* @param file The file to join the radiation field to
			* @param join_mode The mode to join the fields
			* @param check_mode The mode to check the fields
			* @param fallback_version The version of the store to use if the file does not exist
			*/
			static void join(std::shared_ptr<IRadiationField> field, std::shared_ptr<RadiationFieldMetadata> metadata, const std::string& file, FieldJoinMode join_mode, FieldJoinCheckMode check_mode = FieldJoinCheckMode::MetadataSimulationSimilar, StoreVersion fallback_version = StoreVersion::V1);
		
			/** Construct a field accessor from a file, that can be used for all files that share the same structure (metadata-size and field structure)
			* This is useful when parsing large datasets.
			* @param file The file to construct the accessor from
			* @return The field accessor
			*/
			static std::shared_ptr<FieldAccessor> construct_accessor(const std::string& file);

			/** Construct a field accessor from a buffer, that can be used for all files that share the same structure (metadata-size and field structure)
			* This is useful when parsing large datasets.
			* @param buffer The buffer to construct the accessor from
			* @return The field accessor
			*/
			static std::shared_ptr<FieldAccessor> construct_accessor(std::istream& buffer);
		};
	};
}