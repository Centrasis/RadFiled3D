#pragma once
#include "RadFiled3D/VoxelBuffer.hpp"
#include <glm/vec2.hpp>
#include <math.h>
#include <memory>


namespace RadFiled3D
{
	class RADFILED_API PolarSegments: public std::enable_shared_from_this<PolarSegments> {
	protected:
		const glm::uvec2 segments_count;
		std::shared_ptr<VoxelLayer> layer;
	public:
		PolarSegments(const glm::uvec2& segments_count, std::shared_ptr<VoxelLayer> layer = std::shared_ptr<VoxelLayer>(nullptr));

		/** Access a voxels flat index at the given quantized coordinates within the range (0, 0) to (segments_count.x - 1, segments_count.y - 1)
		* @param x The x coordinate of the voxel in the segment grid [0, segments_count.x - 1]
		* @param y The y coordinate of the voxel in the segment grid [0, segments_count.y - 1]
		* @return The flat voxel index
		*/
		inline size_t get_segment_idx(size_t x, size_t y) const {
			return y * segments_count.x + x;
		};

		/** Access a voxels flat index at the given continuous coordinates within the range [0, 4pi] in both dimensions
		* @param phi The phi coordinate of the voxel in radians [0, 4pi]
		* @param theta The theta coordinate of the voxel in radians [0, 4pi]
		* @return The flat voxel index
		*/
		inline size_t get_segment_idx_by_coord(float phi, float theta) const {
			// calc x index from phi
			size_t x = std::floor(((1.f + std::sin(phi / 2.f)) / 2.f) * this->segments_count.x - 1);
			// calc y index from theta
			size_t y = std::floor(((1.f + std::sin(theta / 2.f)) / 2.f) * this->segments_count.y - 1);

			return this->get_segment_idx(x, y);
		};

		/** get the number of segments */
		inline const glm::uvec2& get_segments_count() const {
			return this->segments_count;
		};

		/** access voxel at given spherical coordinates
		* @param phi phi-coordinate of the voxel in radians, per [0, 4pi]
		* @param theta theta-coordinate of the voxel in radians [0, 4pi]
		* @return reference to the voxel
		*/
		template<class VoxelT = IVoxel>
		inline VoxelT& get_segment_by_coord(float phi, float theta) const {
			if (layer == nullptr)
				throw std::runtime_error("Layer not set");
			return this->layer->get_voxel_flat<VoxelT>(this->get_segment_idx_by_coord(phi, theta));
		};

		/** access voxel at given quantized xy-coordinates within the range (0, 0) to (segments_count.x - 1, segments_count.y - 1)
		* @param x x-coordinate of the voxel
		* @param y y-coordinate of the voxel
		* @return reference to the voxel
		*/
		template<class VoxelT = IVoxel>
		inline VoxelT& get_segment(size_t x, size_t y) const {
			if (layer == nullptr)
				throw std::runtime_error("Layer not set");
			return this->layer->get_voxel_flat<VoxelT>(this->get_segment_idx(x, y));
		};

		inline std::shared_ptr<VoxelLayer> get_layer() const {
			return this->layer;
		};
	};

	class RADFILED_API PolarSegmentsBuffer : public VoxelBuffer, public std::enable_shared_from_this<PolarSegmentsBuffer> {
	protected:
		PolarSegments segments;
	public:
		/** Create a new PolarSegmentsBuffer
		* @param segments_count The number of segments in each dimension
		*/
		PolarSegmentsBuffer(const glm::uvec2& segments_count);

		/** Access a voxels flat index at the given quantized coordinates within the range (0, 0) to (segments_count.x - 1, segments_count.y - 1)
		* @param x The x coordinate of the voxel in the segment grid [0, segments_count.x - 1]
		* @param y The y coordinate of the voxel in the segment grid [0, segments_count.y - 1]
		* @return The flat voxel index
		*/
		inline size_t get_segment_idx(size_t x, size_t y) const {
			return this->segments.get_segment_idx(x, y);
		};

		/** Access a voxels flat index at the given continuous coordinates within the range [0, 4pi] in both dimensions
		* @param phi The phi coordinate of the voxel in radians [0, 4pi]
		* @param theta The theta coordinate of the voxel in radians [0, 4pi]
		* @return The flat voxel index
		*/
		inline size_t get_segment_idx_by_coord(float phi, float theta) const {
			return this->segments.get_segment_idx_by_coord(phi, theta);
		};

		/** access voxel at given spherical coordinates
		* @param layer_name name of the layer
		* @param phi phi-coordinate of the voxel in radians, per [0, 4pi]
		* @param theta theta-coordinate of the voxel in radians [0, 4pi]
		* @return reference to the voxel
		*/
		template<class VoxelT = IVoxel>
		inline VoxelT& get_segment_by_coord(const std::string& layer_name, float phi, float theta) const {		
			return this->get_segment_flat<VoxelT>(layer_name, this->get_segment_idx_by_coord(phi, theta));
		};

		/** access voxel at given quantized xy-coordinates within the range (0, 0) to (segments_count.x - 1, segments_count.y - 1)
		* @param layer_name name of the layer
		* @param x x-coordinate of the voxel
		* @param y y-coordinate of the voxel
		* @return reference to the voxel
		*/
		template<class VoxelT = IVoxel>
		inline VoxelT& get_segment(const std::string& layer_name, size_t x, size_t y) const {
			return this->get_segment_flat<VoxelT>(layer_name, this->get_segment_idx(x, y));
		};

		/** access voxel at given flat index
		* @param layer_name name of the layer
		* @param idx flat index of the voxel
		* @return reference to the voxel
		*/
		template<class VoxelT = IVoxel>
		inline VoxelT& get_segment_flat(const std::string& layer_name, size_t idx) const {
			auto found = this->layers.find(layer_name);
			if (found == this->layers.end())
				throw std::runtime_error("Layer: '" + layer_name + "' not found");

			return *(VoxelT*)(found->second.voxels + idx * found->second.bytes_per_voxel);
		};

		/** get the number of segments */
		inline const glm::uvec2& get_segments_count() const {
			return this->segments.get_segments_count();
		};

		/** Create a deep copy of the voxel buffer
		* @return The deep copy of the voxel buffer
		* @note The copy will have the same number of voxels and the same data buffer values
		*/
		virtual VoxelBuffer* copy() const override;
	};
}
