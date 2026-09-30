#include "radfiled3d/radiation_field.hpp"
#include <limits>
#include <type_traits>


using namespace radfiled3d;


CartesianRadiationField::CartesianRadiationField(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions)
	: voxel_dimensions(voxel_dimensions),
	  field_dimensions(field_dimensions),
	  voxel_counts(glm::uvec3((field_dimensions + glm::vec3(std::numeric_limits<float>::epsilon())) / voxel_dimensions))
{
}

std::shared_ptr<VoxelBuffer> CartesianRadiationField::add_channel(const std::string& channel_name)
{
	auto found = this->channels.find(channel_name);
	if (found != this->channels.end())
		return found->second;
	auto grid = std::make_shared<VoxelGridBuffer>(this->field_dimensions, this->voxel_dimensions);
	this->channels[channel_name] = grid;
	return grid;
}

PolarRadiationField::PolarRadiationField(const glm::uvec2& segments_count)
	: segments_count(segments_count)
{
}

std::shared_ptr<VoxelBuffer> PolarRadiationField::add_channel(const std::string& channel_name)
{
	auto found = this->channels.find(channel_name);
	if (found != this->channels.end())
		return found->second;
	auto grid = std::make_shared<PolarSegmentsBuffer>(this->segments_count);
	this->channels[channel_name] = grid;
	return grid;
}
