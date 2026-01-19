#pragma once
#include <memory>
#include <vector>
#include "RadFiled3D/VoxelGrid.hpp"

namespace RadFiled3D {

	enum class RADFILED_API GridTracerAlgorithm {
		SAMPLING = 0,
		BRESENHAM = 1,
		LINETRACING = 2
	};

	class RADFILED_API GridTracer: public std::enable_shared_from_this<GridTracer> {
	protected:
		VoxelGridBuffer& buffer;
		const glm::vec3 field_dimensions;
	public:
		GridTracer(VoxelGridBuffer& buffer) 
			: buffer(buffer),
			  field_dimensions(glm::vec3(buffer.get_voxel_counts())* buffer.get_voxel_dimensions())
		{}

		virtual std::vector<size_t> trace(const glm::vec3& p1, const glm::vec3& p2) = 0;
	};

	/** Traces a line between two points in the grid using a sampling approach.
		In this approach the minimum sampling size is the length of the line segment.
		If the line segment is longer than the minimum sampling size, which is half the L2-Norm of the voxel size, the line is divided into segments of the minimum sampling size.
		This approach counts the hits if the line segment is incident to a voxel, only!
		*/
	class RADFILED_API SamplingGridTracer : public GridTracer, public std::enable_shared_from_this<SamplingGridTracer> {
	public:
		SamplingGridTracer(VoxelGridBuffer& buffer) : GridTracer(buffer) {}

		virtual std::vector<size_t> trace(const glm::vec3& p1, const glm::vec3& p2) override;
	};

	/** Traces a line between two points in the grid using the Bresenham algorithm.
		This algorithm is a line rasterization algorithm that is used to trace a line between two points in a grid. The starting point is excluded as this can only exit a voxel.
		*/
	class RADFILED_API BresenhamGridTracer : public GridTracer, public std::enable_shared_from_this<BresenhamGridTracer> {
	public:
		BresenhamGridTracer(VoxelGridBuffer& buffer) : GridTracer(buffer) {}

		virtual std::vector<size_t> trace(const glm::vec3& p1, const glm::vec3& p2) override;

		bool isInside(const glm::ivec3& point) const;
	};

	/** This class traces a line between two points in the grid using a combination of the sampling tracer and a line tracing algorithm.
		First the lossy sampling tracer is used to trace the line. Then all adjacent voxels to the voxels that were hit are tested using a line-segment intersection test algorithm.
		*/
	class RADFILED_API LinetracingGridTracer : public GridTracer, public std::enable_shared_from_this<LinetracingGridTracer> {
	protected:
		const glm::vec3 gridDimensions;
		SamplingGridTracer lossyTracer;
	public:
		LinetracingGridTracer(VoxelGridBuffer& buffer)
			: GridTracer(buffer),
			  gridDimensions(glm::vec3(this->buffer.get_voxel_counts()) * this->buffer.get_voxel_dimensions()),
			lossyTracer(buffer)
		{}

		virtual std::vector<size_t> trace(const glm::vec3& p1, const glm::vec3& p2) override;

		/** Clips a line to the grid dimensions by modifying the start and end points
		* @param start The start point of the line
		* @param end The end point of the line
		* @return True if the line is at least partially inside the grid, false otherwise
		*/
		bool clipLine(glm::vec3& start, glm::vec3& end) const;

		/** Checks if a line intersects an axis-aligned bounding box
		* @param line_start The start point of the line
		* @param line_end The end point of the line
		* @param vx_pos The start point of the bounding box
		* @param vx_pos_end The end point of the bounding box
		* @return True if the line intersects the bounding box, false otherwise
		*/
		bool intersectsAABB(const glm::vec3& line_start, const glm::vec3& line_end, const glm::vec3& vx_pos, const glm::vec3& vx_pos_end) const;

		/** Checks if a point is inside the grid
		* @param point The point to check
		* @return True if the point is inside the grid, false otherwise
		*/
		bool isInside(const glm::vec3& point) const;
	};
}