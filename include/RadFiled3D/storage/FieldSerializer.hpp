#pragma once
#include <memory>
#include <sstream>
#include "RadFiled3D/storage/Types.hpp"

namespace RadFiled3D {
	class VoxelBuffer;
	class VoxelLayer;
	class IRadiationField;

	namespace Storage {
		class BinayFieldBlockHandler {
		public:
			/** Serializes a radiation field to a binary string
			* @param field The radiation field
			* @return The binary string
			*/
			virtual void serializeField(std::shared_ptr<IRadiationField> field, std::ostream& buffer) const = 0;

			/** Serializes a voxel buffer to a binary string
			* @param voxel_buffer The voxel buffer
			* @return The binary string
			*/
			virtual std::unique_ptr<std::ostringstream> serializeChannel(std::shared_ptr<VoxelBuffer> voxel_buffer) const = 0;

			/** Deserializes a binary buffer of a channel to a voxel buffer
			* @param destination The destination voxel buffer
			* @param data The binary buffer
			* @param size The size of the binary string
			* @return The destination voxel buffer
			*/
			virtual std::shared_ptr<VoxelBuffer> deserializeChannel(std::shared_ptr<VoxelBuffer> destination, char* data, size_t size) const = 0;

			/** Deserializes a channel directly from a stream, reading each layer's voxel data
			* straight into its final buffer (no intermediate channel/layer copies).
			* @param destination The destination voxel buffer
			* @param buffer The stream positioned at the start of the channel's layer data
			* @param channel_bytes The total size of the channel's layer data in bytes
			*/
			virtual void deserializeChannelFromStream(std::shared_ptr<VoxelBuffer> destination, std::istream& buffer, size_t channel_bytes) const = 0;

			/** Deserializes a binary buffer of a channel to a voxel buffer
			* @param data The binary buffer
			* @param size The size of the binary string
			* @return The destination voxel buffer
			*/
			virtual VoxelLayer* deserializeLayer(char* data, size_t size) const = 0;

			/** Deserializes a single layer directly from a stream, reading the voxel data
			* straight into the layer's final buffer (no intermediate copy).
			* @param buffer The stream positioned at the start of the layer block
			* @param size The total size of the layer block in bytes
			* @return The deserialized layer
			*/
			virtual VoxelLayer* deserializeLayerFromStream(std::istream& buffer, size_t size) const = 0;

			/** Deserializes a radiation field from a binary string
			* @param buffer The binary string
			* @return The radiation field
			*/
			virtual std::shared_ptr<IRadiationField> deserializeField(std::istream& buffer) const = 0;

			virtual FieldType getFieldType(std::istream& buffer) const = 0;
		};

		namespace V1 {
			class BinayFieldBlockHandler : public RadFiled3D::Storage::BinayFieldBlockHandler {
			protected:
				/** Adds a histogram layer to the voxel buffer
				* @param field The voxel buffer
				* @param layer The layer name
				* @param bytes_per_element The number of bytes per element
				* @param max_energy_eV The maximum energy in eV
				* @param unit The unit of the histogram
				*/
				static void add_hist_layer(std::shared_ptr<VoxelBuffer> field, const std::string& layer, size_t bytes_per_element, float max_energy_eV, const std::string& unit, void* header_data);
				static void add_spherical_layer(std::shared_ptr<VoxelBuffer> field, const std::string& layer, size_t bytes_per_element, const std::string& unit, void* header_data);

				/** Builds a layer that takes ownership of `owned_data` (already filled with the
				* voxel data) and constructs the per-voxel wrappers for it. No fill, no copy. */
				static VoxelLayer* constructOwnedLayer(const FiledTypes::V1::VoxelGridLayerHeader& layer_desc, size_t voxel_count, char* owned_data, const void* header_data);
			public:
				BinayFieldBlockHandler() = default;

				virtual void serializeField(std::shared_ptr<IRadiationField> field, std::ostream& buffer) const override;

				/** Serializes a voxel buffer to a binary string
				* @param voxel_buffer The voxel buffer
				* @return The binary string
				*/
				virtual std::unique_ptr<std::ostringstream> serializeChannel(std::shared_ptr<VoxelBuffer> voxel_buffer) const override;

				/** Deserializes a binary buffer of a channel to a voxel buffer
				* @param destination The destination voxel buffer
				* @param data The binary buffer
				* @param size The size of the binary string
				* @return The destination voxel buffer
				*/
				virtual std::shared_ptr<VoxelBuffer> deserializeChannel(std::shared_ptr<VoxelBuffer> destination, char* data, size_t size) const override;

				virtual void deserializeChannelFromStream(std::shared_ptr<VoxelBuffer> destination, std::istream& buffer, size_t channel_bytes) const override;

				/** Deserializes a binary buffer of a channel to a voxel buffer
				* @param data The binary buffer
				* @param size The size of the binary string
				* @return The destination voxel buffer
				*/
				virtual VoxelLayer* deserializeLayer(char* data, size_t size) const override;

				virtual VoxelLayer* deserializeLayerFromStream(std::istream& buffer, size_t size) const override;

				/** Deserializes a radiation field from a binary string
				* @param buffer The binary string
				* @return The radiation field
				*/
				virtual std::shared_ptr<IRadiationField> deserializeField(std::istream& buffer) const override;

				virtual FieldType getFieldType(std::istream& buffer) const override;
			};
		};
	};
}
