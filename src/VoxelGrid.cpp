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
	this->copy_layers_to(*copy);
	return copy;
}