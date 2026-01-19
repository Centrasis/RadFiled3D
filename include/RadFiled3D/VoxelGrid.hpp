#pragma once
#include "RadFiled3D/VoxelBuffer.hpp"
#include <glm/vec3.hpp>
#include <memory>


namespace RadFiled3D
{
	class RADFILED_API VoxelGrid: public std::enable_shared_from_this<VoxelGrid> {
	protected:
		const glm::vec3 voxel_dimensions;
		const glm::uvec3 voxel_counts;
		std::shared_ptr<VoxelLayer> layer;
	public:
		VoxelGrid(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions, std::shared_ptr<VoxelLayer> layer = std::shared_ptr<VoxelLayer>(nullptr));

		/** Access a voxels flat index at the given quantized coordinates within the range (0, 0, 0) to (voxel_counts.x - 1, voxel_counts.y - 1, voxel_counts.z - 1)
		* @param x The x coordinate of the voxel
		* @param y The y coordinate of the voxel
		* @param z The z coordinate of the voxel
		* @return The flat voxel index
		*/
		inline size_t get_voxel_idx(size_t x, size_t y, size_t z) const {
			return z * this->voxel_counts.y * this->voxel_counts.x + (y * this->voxel_counts.x) + x;
		};

		inline std::shared_ptr<VoxelLayer> get_layer() const {
			return this->layer;
		};

		/** Access a voxels flat index at the given spatial coordinates within the range (0, 0, 0) to (field_dimensions.x, field_dimensions.y, field_dimensions.z)
		* @param x The x coordinate of the voxel
		* @param y The y coordinate of the voxel
		* @param z The z coordinate of the voxel
		* @return The flat voxel index
		*/
		inline size_t get_voxel_idx_by_coord(float x, float y, float z) const {
			size_t x_index = static_cast<size_t>(x / this->voxel_dimensions.x);
			size_t y_index = static_cast<size_t>(y / this->voxel_dimensions.y);
			size_t z_index = static_cast<size_t>(z / this->voxel_dimensions.z);
			return this->get_voxel_idx(x_index, y_index, z_index);
		};

		/** Access a voxels number in each dimension at the given flat index
		* @param idx The flat voxel index
		*  @return The number of voxels in each dimension
		*/
		inline glm::uvec3 get_voxel_indices(size_t idx) const {
			size_t z = idx / (this->voxel_counts.y * this->voxel_counts.x);
			size_t y = (idx - z * this->voxel_counts.y * this->voxel_counts.x) / this->voxel_counts.x;
			size_t x = idx - z * this->voxel_counts.y * this->voxel_counts.x - y * this->voxel_counts.x;
			return glm::uvec3(x, y, z);
		};

		/** Access a voxels spatial coordinates at the given flat index
		* @param idx The flat voxel index
		* @return The spatial coordinates of the voxel
		*/
		inline glm::vec3 get_voxel_coords(size_t idx) const {
			glm::uvec3 indices = this->get_voxel_indices(idx);
			return glm::vec3(indices) * this->voxel_dimensions;
		};

		/** Access a voxel at the given quantized coordinates within the range (0, 0, 0) to (voxel_counts.x - 1, voxel_counts.y - 1, voxel_counts.z - 1)
		* @param x The x coordinate of the voxel
		* @param y The y coordinate of the voxel
		* @param z The z coordinate of the voxel
		* @return A reference to the voxel at the given coordinates
		*/
		template<class VoxelT = IVoxel>
		inline VoxelT& get_voxel(size_t x, size_t y, size_t z) const {
			if (this->layer.get() == nullptr)
				throw std::runtime_error("Layer not set");
			return this->layer->get_voxel_flat<VoxelT>(this->get_voxel_idx(x, y, z));
		};

		/** Access a voxel at the given spatial coordinates within the range (0, 0, 0) to (field_dimensions.x, field_dimensions.y, field_dimensions.z)
		* @param x The x coordinate of the voxel
		* @param y The y coordinate of the voxel
		* @param z The z coordinate of the voxel
		* @return A reference to the voxel at the given coordinates
		*/
		template<class VoxelT = IVoxel>
		inline VoxelT& get_voxel_by_coord(float x, float y, float z) const {
			if (this->layer.get() == nullptr)
				throw std::runtime_error("Layer not set");
			return this->layer->get_voxel_flat<VoxelT>(this->get_voxel_idx_by_coord(x, y, z));
		};

		/** Returns the dimensions of the voxels in the grid
		* @return The dimensions of the voxels in the grid
		*/
		inline const glm::vec3& get_voxel_dimensions() const {
			return this->voxel_dimensions;
		};

		/** Returns the number of voxels in each dimension of the grid
		* @return The number of voxels in each dimension of the grid
		*/
		inline const glm::uvec3& get_voxel_counts() const {
			return this->voxel_counts;
		};
	};


	/** A 3D grid of equally sized voxels using cartesian coordinates starting at (0, 0, 0) and going to (field_dimensions.x, field_dimensions.y, field_dimensions.z)
	*/
	class RADFILED_API VoxelGridBuffer : public VoxelBuffer, public std::enable_shared_from_this<VoxelGridBuffer> {
	protected:
		VoxelGrid voxel_grid;
	public:
		/** Returns the contained voxel grid */
		const VoxelGrid& get_grid() const {
			return this->voxel_grid;
		}

		/** Create a new voxel grid with the given dimensions and voxel dimensions
		* @param field_dimensions The dimensions of the field in which the voxels are placed
		* @param voxel_dimensions The dimensions of the voxels
		*/
		VoxelGridBuffer(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions);

		/** Access a voxels flat index at the given quantized coordinates within the range (0, 0, 0) to (voxel_counts.x - 1, voxel_counts.y - 1, voxel_counts.z - 1)
		* @param x The x coordinate of the voxel
		* @param y The y coordinate of the voxel
		* @param z The z coordinate of the voxel
		* @return The flat voxel index
		*/
		inline size_t get_voxel_idx(size_t x, size_t y, size_t z) const {
			return this->voxel_grid.get_voxel_idx(x, y, z);
		};

		/** Access a voxels flat index at the given spatial coordinates within the range (0, 0, 0) to (field_dimensions.x, field_dimensions.y, field_dimensions.z)
		* @param x The x coordinate of the voxel
		* @param y The y coordinate of the voxel
		* @param z The z coordinate of the voxel
		* @return The flat voxel index
		*/
		inline size_t get_voxel_idx_by_coord(float x, float y, float z) const {
			return this->voxel_grid.get_voxel_idx_by_coord(x, y, z);
		};

		/** Access a voxel at the given quantized coordinates within the range (0, 0, 0) to (voxel_counts.x - 1, voxel_counts.y - 1, voxel_counts.z - 1)
		* @param layer_name The name of the layer to access
		* @param x The x coordinate of the voxel
		* @param y The y coordinate of the voxel
		* @param z The z coordinate of the voxel
		* @return A reference to the voxel at the given coordinates
		*/
		template<class VoxelT = IVoxel>
		inline VoxelT& get_voxel(const std::string& layer_name, size_t x, size_t y, size_t z) const {
			auto found = this->layers.find(layer_name);
			if (found == this->layers.end())
				throw std::runtime_error("Layer: '" + layer_name + "' not found");
			return *(VoxelT*)(found->second.voxels + this->get_voxel_idx(x, y, z) * found->second.bytes_per_voxel);
		};

		/** Access a voxel at the given spatial coordinates within the range (0, 0, 0) to (field_dimensions.x, field_dimensions.y, field_dimensions.z)
		* @param layer_name The name of the layer to access
		* @param x The x coordinate of the voxel
		* @param y The y coordinate of the voxel
		* @param z The z coordinate of the voxel
		* @return A reference to the voxel at the given coordinates
		*/
		template<class VoxelT = IVoxel>
		inline VoxelT& get_voxel_by_coord(const std::string& layer_name, float x, float y, float z) const {
			size_t x_index = (size_t)(x / this->voxel_grid.get_voxel_dimensions().x);
			size_t y_index = (size_t)(y / this->voxel_grid.get_voxel_dimensions().y);
			size_t z_index = (size_t)(z / this->voxel_grid.get_voxel_dimensions().z);
			return this->get_voxel<VoxelT>(layer_name, x_index, y_index, z_index);
		};

		/** Returns the dimensions of the voxels in the grid
		* @return The dimensions of the voxels in the grid
		*/
		inline const glm::vec3& get_voxel_dimensions() const {
			return this->voxel_grid.get_voxel_dimensions();
		};

		/** Returns the number of voxels in each dimension of the grid
		* @return The number of voxels in each dimension of the grid
		*/
		inline const glm::uvec3& get_voxel_counts() const {
			return this->voxel_grid.get_voxel_counts();
		};

		/** Create a deep copy of the voxel buffer
		* @return The deep copy of the voxel buffer
		* @note The copy will have the same number of voxels and the same data buffer values
		*/
		virtual VoxelBuffer* copy() const override;
	};
}
