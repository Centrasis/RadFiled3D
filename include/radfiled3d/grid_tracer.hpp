#pragma once
#include <memory>
#include <vector>
#include "radfiled3d/voxel_grid.hpp"

namespace radfiled3d {

	enum class GridTracerAlgorithm {
		SAMPLING = 0,
		BRESENHAM = 1,
		LINETRACING = 2
	};

	/** A voxel touched by a traced line segment and the segment's length inside it, in units of the voxel's x edge. */
	struct TracedVoxel {
		size_t index;
		float path_fraction;
		/** Whether trace() returns this voxel too. trace() leaves out the voxel a segment starts in (so consecutive
		* segments of a track count each voxel once) and, depending on the algorithm, voxels a line only crosses near
		* an edge; their part of the path still belongs to the segment. */
		bool counted = true;
		/** The segment starts inside this voxel: its path here leaves the start point (e.g. an interaction) instead of
		* entering the voxel from outside. */
		bool starts_segment = false;
	};

	class GridTracer {
	protected:
		VoxelGridBuffer& buffer;
		const glm::vec3 field_dimensions;
	public:
		GridTracer(VoxelGridBuffer& buffer) 
			: buffer(buffer),
			  field_dimensions(glm::vec3(buffer.get_voxel_counts())* buffer.get_voxel_dimensions())
		{}

		virtual std::vector<size_t> trace(const glm::vec3& p1, const glm::vec3& p2) = 0;

		/** Every voxel the segment passes through (exact grid traversal, Amanatides & Woo 1987) with the length of the
		* segment inside it (path_fraction), plus any further voxel trace() returns (fraction 0, e.g. only touched).
		* The fractions add up to the segment length inside the grid. Weighting a score by them gives the track-length
		* estimate: a corner clip counts almost nothing, a crossing parallel to an axis 1, the space diagonal of a cube
		* sqrt(3), and no direction is favoured by grazing more voxels. `counted` marks the voxels trace() returns.
		*/
		std::vector<TracedVoxel> trace_with_path_fractions(const glm::vec3& p1, const glm::vec3& p2);

		/** Length of the segment p1 -> p2 inside the voxel, in units of the voxel's x edge (0 if it misses the voxel). */
		float path_fraction(const glm::vec3& p1, const glm::vec3& p2, size_t voxel_idx) const;
	};

	/** Traces a line between two points in the grid using a sampling approach.
		In this approach the minimum sampling size is the length of the line segment.
		If the line segment is longer than the minimum sampling size, which is half the L2-Norm of the voxel size, the line is divided into segments of the minimum sampling size.
		This approach counts the hits if the line segment is incident to a voxel, only!
		*/
	class SamplingGridTracer : public GridTracer {
	public:
		SamplingGridTracer(VoxelGridBuffer& buffer) : GridTracer(buffer) {}

		virtual std::vector<size_t> trace(const glm::vec3& p1, const glm::vec3& p2) override;
	};

	/** Traces a line between two points in the grid using the Bresenham algorithm.
		This algorithm is a line rasterization algorithm that is used to trace a line between two points in a grid. The starting point is excluded as this can only exit a voxel.
		*/
	class BresenhamGridTracer : public GridTracer {
	public:
		BresenhamGridTracer(VoxelGridBuffer& buffer) : GridTracer(buffer) {}

		virtual std::vector<size_t> trace(const glm::vec3& p1, const glm::vec3& p2) override;

		bool isInside(const glm::ivec3& point) const;
	};

	/** This class traces a line between two points in the grid using a combination of the sampling tracer and a line tracing algorithm.
		First the lossy sampling tracer is used to trace the line. Then all adjacent voxels to the voxels that were hit are tested using a line-segment intersection test algorithm.
		*/
	class LinetracingGridTracer : public GridTracer {
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