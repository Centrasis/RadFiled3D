#pragma once
#include <RadFiled3D/Voxel.hpp>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>

namespace RadFiled3D {
	namespace Storage {
		class FieldAccessor;
	}
}

using namespace RadFiled3D;

namespace RadFiled3D::Dataset {
	/** A request to load a collection of voxels from a file.
	* Use with VoxelCollectionAccessor to access a collection of voxels from a file.
	*/
	struct RADFILED_API VoxelCollectionRequest {
		const std::string filePath;
		const std::vector<size_t> voxelIndices;

		VoxelCollectionRequest(const std::string& filePath, const std::vector<size_t>& voxelIndices)
			: filePath(filePath), voxelIndices(voxelIndices) {
		}
	};

	/** A collection of voxels organized by channels and layers.
	* Returned by VoxelCollectionAccessor::access.
	* Provides a way to extract a dense data buffer from a specific channel and layer.
	*/
	struct RADFILED_API VoxelCollection {
		struct RADFILED_API Layer {
			std::string name;
			std::vector<std::shared_ptr<IVoxel>> voxels;
		};

		struct RADFILED_API Channel {
			std::map<std::string, Layer> layers;
		};

		std::map<std::string, Channel> channels;

		/** Constructs a VoxelCollection with the specified channels and layers.
		* @param channels The names of the channels to include in the collection
		* @param layers The names of the layers to include in each channel
		* @param voxelCount The number of voxels in each layer
		*/
		VoxelCollection(const std::vector<std::string>& channels, const std::vector<std::string>& layers, size_t voxelCount);

		/** Extracts a dense data buffer from the specified channel and layer.
		* @param channel The name of the channel to extract from
		* @param layer The name of the layer to extract from
		* @return A pointer to the extracted data buffer. The caller is responsible for deallocating the buffer.
		*/
		char* extract_data_buffer_from(const std::string& channel, const std::string& layer);
	};

	class RADFILED_API VoxelCollectionAccessor {
	protected:
		std::shared_ptr<Storage::FieldAccessor> accessor;
		std::vector<std::string> channels;
		std::vector<std::string> layers;

	public:
		VoxelCollectionAccessor(std::shared_ptr<Storage::FieldAccessor> accessor, const std::vector<std::string>& channels, const std::vector<std::string>& layers)
			: accessor(accessor), channels(channels), layers(layers) {
		}

		VoxelCollectionAccessor(const VoxelCollectionAccessor&) = default;
		VoxelCollectionAccessor& operator=(const VoxelCollectionAccessor&) = default;
		VoxelCollectionAccessor(VoxelCollectionAccessor&&) = default;
		VoxelCollectionAccessor& operator=(VoxelCollectionAccessor&&) = default;
		virtual ~VoxelCollectionAccessor() = default;

		std::shared_ptr<VoxelCollection> access(const std::vector<VoxelCollectionRequest>& requests);
	};
}