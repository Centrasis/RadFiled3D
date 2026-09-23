#include <RadFiled3D/dataset/helpers.hpp>
#include <RadFiled3D/storage/FieldAccessor.hpp>
#include <istream>
#include <fstream>
#include <RadFiled3D/storage/RadiationFieldStore.hpp>


using namespace RadFiled3D;
using namespace RadFiled3D::Dataset;


std::shared_ptr<VoxelCollection> RadFiled3D::Dataset::VoxelCollectionAccessor::access(const std::vector<VoxelCollectionRequest>& requests)
{
	size_t voxelsCount = 0;
	for (const auto& request : requests) {
		voxelsCount += request.voxelIndices.size();
	}
	std::shared_ptr<VoxelCollection> collection = std::make_shared<VoxelCollection>(this->channels, this->layers, voxelsCount);

	voxelsCount = 0;
	for (const auto& request : requests) {
		std::ifstream buffer = Storage::open_file_for_reading(request.filePath);
		for (const std::string& channel : this->channels) {
			for (const std::string& layer : this->layers) {
				auto voxels = this->accessor->accessVoxelsRawFlat(buffer, channel, layer, request.voxelIndices);
				auto& channel_data = collection->channels[channel];
				auto& layer_data = channel_data.layers[layer];
				size_t inner_voxels_count = 0;
				for (auto voxel : voxels) {
					if (voxel == nullptr) {
						throw std::runtime_error("Failed to access voxel data for channel: " + channel + ", layer: " + layer);
					}
					layer_data.voxels[voxelsCount + inner_voxels_count] = std::shared_ptr<IVoxel>(voxel, [](IVoxel* p) { p->selfDestruct(); });
					inner_voxels_count++;
				}
				buffer.seekg(0, std::ios::beg); // Reset the buffer position for the next channel/layer
			}
		}
		voxelsCount += request.voxelIndices.size();
	}

	return collection;
}


RadFiled3D::Dataset::VoxelCollection::VoxelCollection(const std::vector<std::string>& channels, const std::vector<std::string>& layers, size_t voxelCount)
{
	for (auto& channelName : channels) {
		this->channels[channelName] = Channel();
		Channel& channel = this->channels[channelName];
		for (auto& layerName : layers) {
			channel.layers[layerName] = Layer();
			Layer& layer = channel.layers[layerName];
			layer.voxels.resize(voxelCount);
		}
	}
}

char* RadFiled3D::Dataset::VoxelCollection::extract_data_buffer_from(const std::string& channel, const std::string& layer)
{
	Channel& targetChannel = this->channels[channel];
	Layer& targetLayer = targetChannel.layers[layer];
	if (targetLayer.voxels.empty()) {
		throw std::runtime_error("No voxels found in layer: " + layer);
	}
	auto vx_size = targetLayer.voxels[0]->get_bytes();
	char* buffer = new char[targetLayer.voxels.size() * vx_size];

	for (size_t i = 0; i < targetLayer.voxels.size(); i++) {
		std::memcpy(buffer + i * vx_size, targetLayer.voxels[i]->get_raw(), vx_size);
	}

	return buffer;
}
