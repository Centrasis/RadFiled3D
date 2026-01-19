#pragma once
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <map>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>
#include <RadFiled3D/VoxelBuffer.hpp>
#include <RadFiled3D/VoxelGrid.hpp>
#include <RadFiled3D/PolarSegments.hpp>


namespace RadFiled3D
{
	/** Interface for radiation fields.
	* A radiation field is a collection of channels which are a layered collection of voxels and therefore a collection of VoxelBuffers.
	* Each channel is a collection of layered voxels aka a VoxelBuffer.
	* The channels are identified by a name.
	* The radiation field is responsible for managing the channels.
	* The radiation field is responsible for creating the channels.
	* The radiation field is responsible for providing the channels.
	* The radiation field is responsible for providing the type name of the concrete class.
	*/
	class RADFILED_API IRadiationField : public std::enable_shared_from_this<IRadiationField> {
	public:
		/** Get a channel by name
		* @param channel_name The name of the channel to get
		* @return The channel
		*/
		virtual std::vector<std::pair<std::string, std::shared_ptr<VoxelBuffer>>> get_channels() const = 0;

		/** Get if a channel exists by name
		* @param channel_name The name of the channel to check
		* @return True if the channel exists, false otherwise
		*/
		virtual bool has_channel(const std::string& channel_name) const = 0;

		/** Statically returns the type name of the concrete class
		* @return The type name
		*/
		virtual const std::string& get_typename() const = 0;

		/** Add a channel to the field
		* @param channel_name The name of the channel
		* @return The channel of the type of the concrete radiation field
		*/
		virtual std::shared_ptr<VoxelBuffer> add_channel(const std::string& channel_name) = 0;

		/** Get a channel by name and as its base class
		* @param channel_name The name of the channel
		* @return The channel of the base type
		*/
		virtual std::shared_ptr<VoxelBuffer> get_generic_channel(const std::string& channel_name) = 0;

		/** Get the channels of the channels present
		* @return The names of the channels
		*/
		virtual std::vector<std::string> get_channel_names() const = 0;

		/** Create a deep copy of the radiation field
		* @return The copy of the radiation field
		*/
		virtual std::shared_ptr<IRadiationField> copy() const = 0;
	};

	template<class BufferT>
	class RADFILED_API RadiationField : public IRadiationField, public std::enable_shared_from_this<RadiationField<BufferT>> {
	protected:
		std::map<std::string, std::shared_ptr<BufferT>> channels;
	public:
		/** Get a channel by name
		* 	@param channel_name The name of the channel to get
		* 	@return The channel
		* 	@throws std::runtime_error if the channel is not found
		*/
		inline std::shared_ptr<BufferT> get_channel(const std::string& channel_name) const {
			auto found = this->channels.find(channel_name);
			if (found == this->channels.end())
				throw std::runtime_error("Channel: '" + channel_name + "' not found");
			return found->second;
		};

		virtual std::shared_ptr<VoxelBuffer> get_generic_channel(const std::string& channel_name) override {
			return this->get_channel(channel_name);
		}

		/** Get if a channel exists by name
		* @param channel_name The name of the channel to check
		* @return True if the channel exists, false otherwise
		*/
		virtual bool has_channel(const std::string& channel_name) const override {
			return this->channels.find(channel_name) != this->channels.end();
		};

		/** Get the names of the channels present
		* @return The names of the channels
		*/
		virtual std::vector<std::string> get_channel_names() const override {
			std::vector<std::string> names;
			for (auto& channel : this->channels) {
				names.push_back(channel.first);
			}
			return names;
		}

		/** Get a channel by name
		* @return vector of 2-Tuple of name and channel
		*/
		virtual std::vector<std::pair<std::string, std::shared_ptr<VoxelBuffer>>> get_channels() const override {
			std::vector<std::pair<std::string, std::shared_ptr<VoxelBuffer>>> channels;
			for (auto& channel : this->channels) {
				channels.push_back(std::make_pair(channel.first, channel.second));
			}
			return channels;
		}
	};

	/** A Cartesian radiation field.
	* A Cartesian radiation field is a radiation field with a Cartesian grid of voxels (VoxelGridBuffer).
	* The field is defined by the dimensions of the field and the dimensions of a voxel.
	* The field is responsible for creating channels of voxels.
	*/
	class RADFILED_API CartesianRadiationField : public RadiationField<VoxelGridBuffer>, public std::enable_shared_from_this<CartesianRadiationField> {
	protected:
		const glm::vec3 voxel_dimensions;
		const glm::uvec3 voxel_counts;
		const glm::vec3 field_dimensions;
	public:
		/** Create a new CartesianRadiationField
		* @param field_dimensions The dimensions of the field
		* @param voxel_dimensions The dimensions of a voxel
		*/
		CartesianRadiationField(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions);

		/** Get the type name of the concrete class
		* @return The type name
		*/
		virtual const std::string& get_typename() const override {
			static const std::string name = "CartesianRadiationField";
			return name;
		}

		/** Get the dimensions of a voxel
		* @return The dimensions of a voxel
		*/
		inline const glm::vec3& get_voxel_dimensions() const {
			return this->voxel_dimensions;
		};

		/** Get the number of voxels in each dimension
		* @return The number of voxels in each dimension
		*/
		inline const glm::uvec3& get_voxel_counts() const {
			return this->voxel_counts;
		};

		/** Get the dimensions of the field
		* @return The dimensions of the field
		*/
		inline const glm::vec3& get_field_dimensions() const {
			return this->field_dimensions;
		};

		/** Add a channel to the field
		* @param channel_name The name of the channel
		* @return The channel of the type of the concrete radiation field (VoxelGridBuffer)
		*/
		virtual std::shared_ptr<VoxelBuffer> add_channel(const std::string& channel_name) override;

		/** Create a deep copy of the radiation field
		* @return The copy of the radiation field
		*/
		virtual std::shared_ptr<IRadiationField> copy() const override {
			auto field = std::make_shared<CartesianRadiationField>(this->field_dimensions, this->voxel_dimensions);
			for (auto& channel : this->channels) {
				VoxelGridBuffer* buffer_cpy = (VoxelGridBuffer*)channel.second->copy();
				field->channels[channel.first] = std::shared_ptr<VoxelGridBuffer>(buffer_cpy);
			}
			return field;
		}
	};

	/** A Polar radiation field.
	* A Polar radiation field is a radiation field with a spherical grid of voxels (PolarSegmentsBuffer).
	* The field is defined by the number of segments in each dimension.
	* The field is responsible for creating channels of voxels.
	*/
	class RADFILED_API PolarRadiationField : public RadiationField<PolarSegmentsBuffer>, public std::enable_shared_from_this<PolarRadiationField> {
	protected:
		const glm::uvec2 segments_count;
	public:
		/** Create a new PolarRadiationField
		* @param segments_count The number of segments in each dimension
		* @param segment_dimensions The dimensions of a segment
		*/
		PolarRadiationField(const glm::uvec2& segments_count);

		/** Add a channel to the field
		* @param channel_name The name of the channel
		* @return The channel of the type of the concrete radiation field (PolarSegmentsBuffer)
		*/
		virtual std::shared_ptr<VoxelBuffer> add_channel(const std::string& channel_name) override;

		/** Get the type name of the concrete class
		* @return The type name
		*/
		virtual const std::string& get_typename() const override {
			static const std::string name = "PolarRadiationField";
			return name;
		}

		/** Get the number of segments in each dimension
		* @return The number of segments in each dimension
		*/
		inline const glm::uvec2& get_segments_count() const {
			return this->segments_count;
		};

		/** Create a deep copy of the radiation field
		* @return The copy of the radiation field
		*/
		virtual std::shared_ptr<IRadiationField> copy() const override {
			auto field = std::make_shared<PolarRadiationField>(this->segments_count);
			for (auto& channel : this->channels) {
				PolarSegmentsBuffer* buffer_cpy = (PolarSegmentsBuffer*)channel.second->copy();
				field->channels[channel.first] = std::shared_ptr<PolarSegmentsBuffer>(buffer_cpy);
			}
			return field;
		}
	};
}
