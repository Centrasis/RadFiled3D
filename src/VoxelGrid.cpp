#include "RadFiled3D/VoxelGrid.hpp"
#include <limits>


using namespace RadFiled3D;

namespace {
	// Same rounding as CartesianRadiationField: without the epsilon, a size rebuilt as counts * voxel size (e.g. when
	// loading 60 voxels of 0.02 m) divides to 59.99999 and loses a voxel.
	glm::uvec3 voxel_counts_of(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions)
	{
		return glm::uvec3((field_dimensions + glm::vec3(std::numeric_limits<float>::epsilon())) / voxel_dimensions);
	}

	size_t voxel_count_of(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions)
	{
		const glm::uvec3 counts = voxel_counts_of(field_dimensions, voxel_dimensions);
		return static_cast<size_t>(counts.x) * counts.y * counts.z;
	}
}

VoxelGrid::VoxelGrid(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions, std::shared_ptr<VoxelLayer> layer)
	: voxel_dimensions(voxel_dimensions),
	  voxel_counts(voxel_counts_of(field_dimensions, voxel_dimensions)),
	  layer(layer)
{
}

VoxelGridBuffer::VoxelGridBuffer(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions)
	: voxel_grid(field_dimensions, voxel_dimensions),
	  VoxelBuffer(voxel_count_of(field_dimensions, voxel_dimensions))
{
}

VoxelBuffer* VoxelGridBuffer::copy() const
{
	glm::vec3 field_dimensions = glm::vec3(this->voxel_grid.get_voxel_counts()) * this->voxel_grid.get_voxel_dimensions();
	VoxelGridBuffer* copy = new VoxelGridBuffer(field_dimensions, this->get_voxel_dimensions());
	for (auto& layer : this->layers)
	{
		auto& layer_info = layer.second;
		IVoxel* initial_vx = (IVoxel*)layer_info.voxels;
		size_t bytes_per_voxel_databuffer = initial_vx->get_bytes();
		auto data = new char[this->voxel_count * bytes_per_voxel_databuffer];
		auto voxels = new char[this->voxel_count * layer_info.bytes_per_voxel];
		memcpy(data, layer_info.data, this->voxel_count * bytes_per_voxel_databuffer);
		memcpy(voxels, layer_info.voxels, this->voxel_count * layer_info.bytes_per_voxel);
		for (size_t i = 0; i < this->voxel_count; i++)
		{
			IVoxel* vx = (IVoxel*)(voxels + i * layer_info.bytes_per_voxel);
			vx->set_data((void*)(data + i * bytes_per_voxel_databuffer));
		}
		copy->layers[layer.first] = VoxelLayer(layer_info.bytes_per_voxel, layer_info.bytes_per_data_element, voxels, data, layer_info.unit, layer_info.statistical_error, this->voxel_count);
	}
	return copy;
}