#include "radfiled3d/polar_segments.hpp"


using namespace radfiled3d;

PolarSegments::PolarSegments(const glm::uvec2& segments_count, std::shared_ptr<VoxelLayer> layer)
	: segments_count(segments_count),
		layer(layer)
{
}

PolarSegmentsBuffer::PolarSegmentsBuffer(const glm::uvec2& segments_count)
	: segments(segments_count),
		VoxelBuffer(segments_count.x * segments_count.y)
{
}

VoxelBuffer* PolarSegmentsBuffer::copy() const
{
	PolarSegmentsBuffer* copy = new PolarSegmentsBuffer(this->segments.get_segments_count());
	this->copy_layers_to(*copy);
	return copy;
}