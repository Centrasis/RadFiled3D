#pragma once
#include <vector>
#include <memory>
#include <map>
#include <cstring>
#include "RadFiled3D/Voxel.hpp"
#include <stdexcept>
#include <functional>
#include <algorithm>


namespace RadFiled3D {
	class RADFILED_API VoxelBufferException : public std::runtime_error {
	public:
		VoxelBufferException(const std::string& message) : std::runtime_error("VoxelBufferException: " + message) {}
	};

	/** A VoxelLayer is a layer of voxels in a VoxelBuffer
	* A VoxelLayer contains a buffer of voxels and a buffer of data values.
	* It is NOT intendet to be used directly, but rather through the VoxelBuffer class.
	* @see VoxelBuffer
	*/
	class RADFILED_API VoxelLayer: public std::enable_shared_from_this<VoxelLayer> {
		friend class VoxelBuffer;
		friend class PolarSegmentsBuffer;
		friend class VoxelGridBuffer;

	protected:
		size_t voxel_count;
		char* voxels;
		char* data;
		std::string unit;
		size_t bytes_per_voxel;
		size_t bytes_per_data_element;
		float statistical_error = 0.f;
		bool shall_free_buffers;

		/** Destructor of the layer data buffers.
		* Will NOT be called by the VoxelBuffer destructor by default. Set shall_free_buffers to true to enable this.
		*/
		void free_buffers() noexcept;

	public:
		VoxelLayer();
		VoxelLayer(size_t bytes_per_voxel, size_t bytes_per_data_element, char* voxels, char* data, const std::string& unit, float statistical_error, size_t voxel_count, bool shall_free_buffers = false);
		~VoxelLayer();

		inline IVoxel* get_voxel_flat_raw(size_t idx) const {
			return (IVoxel*)(this->voxels + idx * this->bytes_per_voxel);
		}

		/** Create a new VoxelLayer with a given number of voxels and a statistical error
		* @param unit The unit of the layer
		* @param voxel_count The number of voxels in the layer
		* @param statistical_error The statistical error of the layer
		* @param initial_voxel_data The initial value to assign to each voxel of the layer
		* @param shall_free_buffers If true, the data buffers will be freed by the destructor
		* @return A pointer to the new VoxelLayer
		*/
		template<typename dtype = float, class VoxelT = ScalarVoxel<dtype>>
		static VoxelLayer* Construct(const std::string& unit, size_t voxel_count, float statistical_error, const dtype& initial_voxel_data, bool shall_free_buffers = false) {
			VoxelT* voxels = new VoxelT[voxel_count];
			dtype* data_buffer = new dtype[voxel_count];

			std::fill(data_buffer, data_buffer + voxel_count, initial_voxel_data);
			std::fill(voxels, voxels + voxel_count, VoxelT(nullptr));
			for (size_t i = 0; i < voxel_count; i++) {
				voxels[i].set_data(data_buffer + i);
			}

			return new VoxelLayer(sizeof(VoxelT), sizeof(dtype), (char*)voxels, (char*)data_buffer, unit, statistical_error, voxel_count, shall_free_buffers);
		}

		/** Create a new VoxelLayer with a given number of voxels and a statistical error from a template voxel instance
		* @param unit The unit of the layer
		* @param voxel_count The number of voxels in the layer
		* @param statistical_error The statistical error of the layer
		* @param voxel_template The template voxel to use for each voxel in the layer
		* @param shall_free_buffers If true, the data buffers will be freed by the destructor
		* @return A pointer to the new VoxelLayer
		*/
		template<typename dtype = float, class VoxelT = ScalarVoxel<dtype>>
		static VoxelLayer* ConstructRaw(const std::string& unit, size_t voxel_count, float statistical_error, const VoxelT& voxel_template, bool shall_free_buffers = false) {
			VoxelT* voxels = new VoxelT[voxel_count];
			std::fill(voxels, voxels + voxel_count, voxel_template);
			size_t voxel_size = voxels[0].get_bytes();
			size_t elements_per_voxel = voxel_size / sizeof(dtype);
			dtype* data_buffer = new dtype[elements_per_voxel * voxel_count];
			for (size_t i = 0; i < voxel_count; i++) {
				voxels[i].set_data(((char*)data_buffer) + i * voxel_size);
			}

			return new VoxelLayer(sizeof(VoxelT), sizeof(dtype), (char*)voxels, (char*)data_buffer, unit, statistical_error, voxel_count, shall_free_buffers);
		}

		/** Create a new VoxelLayer with a given number of voxels and a statistical error from a template voxel instance and a data buffer
		* @param unit The unit of the layer
		* @param voxel_count The number of voxels in the layer
		* @param statistical_error The statistical error of the layer
		* @param src_data_buffer The data buffer to use for the layer. Will be copied.
		* @param voxel_template The template voxel to use for each voxel in the layer
		* @param shall_free_buffers If true, the data buffers will be freed by the destructor
		* @return A pointer to the new VoxelLayer
		*/
		template<typename dtype = float, class VoxelT = ScalarVoxel<dtype>>
		static VoxelLayer* ConstructFromBufferRaw(const std::string& unit, size_t voxel_count, float statistical_error, const char* src_data_buffer, bool shall_free_buffers = false, const VoxelT& voxel_template = VoxelT()) {
			VoxelT* voxels = new VoxelT[voxel_count];
			std::fill(voxels, voxels + voxel_count, voxel_template);
			size_t voxel_size = voxels[0].get_bytes();
			size_t elements_per_voxel = voxel_size / sizeof(dtype);
			dtype* data_buffer = new dtype[elements_per_voxel * voxel_count];
			std::memcpy(data_buffer, src_data_buffer, elements_per_voxel * voxel_count * sizeof(dtype));
			for (size_t i = 0; i < voxel_count; i++) {
				voxels[i].set_data(((char*)data_buffer) + i * voxel_size);
			}

			return new VoxelLayer(sizeof(VoxelT), sizeof(dtype), (char*)voxels, (char*)data_buffer, unit, statistical_error, voxel_count, shall_free_buffers);
		}


		/** Accesses a voxel in a layer by its flat index
		* @param idx The flat index of the voxel
		* @return A reference to the voxel
		*/
		template<class VoxelT = IVoxel>
		VoxelT& get_voxel_flat(size_t idx) const {
			return *(VoxelT*)this->get_voxel_flat_raw(idx);
		};

		const std::string get_unit() const {
			return this->unit;
		}

		size_t get_voxel_count() const {
			return this->voxel_count;
		}

		float get_statistical_error() const {
			return this->statistical_error;
		}

		const char* get_raw_data() const {
			return this->data;
		}

		const size_t get_bytes_per_data_element() const {
			return this->bytes_per_data_element;
		}

		const size_t get_bytes_per_voxel() const {
			return this->bytes_per_voxel;
		}
	};

	class RADFILED_API VoxelBuffer: public std::enable_shared_from_this<VoxelBuffer> {
	protected:
		std::map<std::string, VoxelLayer> layers;
		const size_t voxel_count;
	public:
		/** Construct a voxel buffer with a given number of voxels
		* @param voxel_count The number of voxels in the buffer
		*/
		VoxelBuffer(size_t voxel_count);

		/** Destructor */
		~VoxelBuffer();

		/** Adds a layer to the voxel buffer.
		* @param name The name of the layer
		* @param initial_voxel_data The initial value to assign to each voxel of the new layer
		* @param unit The unit of the layer
		*/
		template<typename dtype = float, class VoxelT = ScalarVoxel<dtype>>
		void add_layer(const std::string& name, const dtype& initial_voxel_data, const std::string& unit = "") {
			VoxelLayer* layer = VoxelLayer::Construct<dtype, VoxelT>(unit, this->voxel_count, -1.f, initial_voxel_data);

			this->layers.insert({ 
				name,
				*layer
			});
			delete layer;
		}

		/** Tests if a layer exists in the buffer
		* @param layer_name The name of the layer
		* @return True if the layer exists, false otherwise
		*/
		inline bool has_layer(const std::string& layer_name) const {
			return this->layers.find(layer_name) != this->layers.end();
		}

		/** Accesses a voxel in a layer by its flat index
		* @param layer_name The name of the layer
		* @param idx The flat index of the voxel
		* @return A reference to the voxel
		*/
		template<class VoxelT = IVoxel>
		inline VoxelT& get_voxel_flat(const std::string& layer_name, size_t idx) const {
			auto found = this->layers.find(layer_name);
			if (found == this->layers.end())
				throw VoxelBufferException("Layer: '" + layer_name + "' not found");
			return found->second.get_voxel_flat<VoxelT>(idx);
		};

		/** Accesses a layer in the buffer by its name
		* @param layer_name The name of the layer
		* @return A reference to the layer
		* @note This function is unsafe and should only be used when the Voxel type is unknown
		*/
		inline const VoxelLayer& get_layer(const std::string& layer_name) const {
			auto found = this->layers.find(layer_name);
			if (found == this->layers.end())
				throw VoxelBufferException("Layer: '" + layer_name + "' not found");
			return found->second;
		}

		/** Returns the number of voxels in the buffer
		* @return The number of voxels in the buffer
		*/
		inline size_t get_voxel_count() const {
			return this->voxel_count;
		}

		/** Set the voxel databuffer of a layer to a new value
		* @param layer_name The name of the layer
		* @param clear_value The new data buffer to assign to the layer
		* @param elements_per_voxel The number of elements per voxel in the data buffer (default 1; Only change for compisite Voxels like HistogramVoxel)
		*/
		template<typename T>
		void clear_layer(const std::string& layer_name, const T& clear_value, size_t elements_per_voxel = 1) {
			auto found = this->layers.find(layer_name);
			if (found == this->layers.end())
				throw VoxelBufferException("Layer: '" + layer_name + "' not found");
			std::fill((T*)found->second.data, ((T*)found->second.data) + this->voxel_count * elements_per_voxel, clear_value);
		}

		/** Adds a custom layer to the voxel buffer using a preconstructed voxel as a template for each Voxel.
		* This is useful when the Voxel type is not a simple scalar type and owns a complex constructor.
		* @param name The name of the layer
		* @param voxel_template The template voxel to use for each voxel in the layer
		* @param initial_voxel_scalar_value The initial value to assign to each voxel of the new layer
		* @param unit The unit of the layer
		*/
		template<class VoxelT, typename dtype>
		void add_custom_layer(const std::string& name, const VoxelT& voxel_template, dtype initial_voxel_scalar_value, const std::string& unit = "") {
			const size_t elements_per_voxel = voxel_template.get_bytes() / sizeof(dtype);
			VoxelT* voxels = new VoxelT[this->voxel_count];
			dtype* data_buffer = new dtype[this->voxel_count * elements_per_voxel];

			std::fill(data_buffer, data_buffer + this->voxel_count * elements_per_voxel, initial_voxel_scalar_value);
			std::fill(voxels, voxels + this->voxel_count, voxel_template);

			for (size_t i = 0; i < this->voxel_count; i++) {
				voxels[i].set_data(data_buffer + i * elements_per_voxel);
			}

			this->layers.insert({
				name,
				VoxelLayer(
					sizeof(VoxelT),
					sizeof(dtype),
					(char*)voxels,
					(char*)data_buffer,
					unit,
					-1.f,
					this->voxel_count
				)
			});
		}

		/** Adds a custom layer to the voxel buffer using a preconstructed voxel as a template for each Voxel.
		* This is useful when the Voxel type is unknown such as for constructing a layer from another buffer.
		* @param name The name of the layer
		* @param voxel_template The template voxel to use for each voxel in the layer
		* @param unit The unit of the layer
		*/
		void add_custom_layer_unsafe(const std::string& name, const IVoxel* voxel_template, const std::string& unit = "") {
			const size_t data_bytes = voxel_template->get_bytes();
			const size_t data_buffer_size = this->voxel_count * data_bytes;
			const size_t vx_bytes = voxel_template->get_voxel_bytes();
			const size_t voxel_buffer_size = this->voxel_count * vx_bytes;
			const char* initial_data = (char*)voxel_template->get_raw();

			char* voxels = new char[voxel_buffer_size];
			char* data_buffer = new char[data_buffer_size];

			for (size_t i = 0; i < this->voxel_count; i++) {
				std::memcpy(voxels + i * vx_bytes, (void*)voxel_template, vx_bytes);
				std::memcpy(data_buffer + i * data_bytes, initial_data, data_bytes);
				((IVoxel*)(voxels + i * vx_bytes))->set_data((void*)(data_buffer + i * data_bytes));
			}

			this->layers.insert({
				name,
				VoxelLayer(
					vx_bytes,
					data_bytes,
					voxels,
					data_buffer,
					unit,
					-1.f,
					this->voxel_count
				)
			});
		}

		/** Returns the pointer to the value data buffer of a layer
		* @param layer_name The name of the layer
		* @return The pointer to the data buffer of the layer
		*/
		template<typename dtype = float>
		inline dtype* get_layer(const std::string& layer_name) const {
			auto found = this->layers.find(layer_name);
			if (found == this->layers.end())
				throw VoxelBufferException("Layer: '" + layer_name + "' not found");
			return (dtype*)found->second.data;
		};

		/** Returns the unit of a layer
		* @param layer_name The name of the layer
		* @return The unit of the layer
		*/
		inline const std::string& get_layer_unit(const std::string& layer_name) const {
			auto found = this->layers.find(layer_name);
			if (found == this->layers.end())
				throw VoxelBufferException("Layer: '" + layer_name + "' not found");
			return found->second.unit;
		}

		/** Returns the statistical error of a layer
		* @param layer_name The name of the layer
		* @return The statistical error of the layer
		*/
		inline float get_statistical_error(const std::string& layer_name) const {
			auto found = this->layers.find(layer_name);
			if (found == this->layers.end())
				throw VoxelBufferException("Layer: '" + layer_name + "' not found");
			return found->second.statistical_error;
		}

		/** Sets the statistical error of a layer
		* @param layer_name The name of the layer
		* @param statistical_error The statistical error to set
		*/
		inline void set_statistical_error(const std::string& layer_name, float statistical_error) {
			auto found = this->layers.find(layer_name);
			if (found == this->layers.end())
				throw VoxelBufferException("Layer: '" + layer_name + "' not found");
			found->second.statistical_error = statistical_error;
		}

		/** Returns a list of the names of the layers in the buffer
		* @return A list of the names of the layers in the buffer
		*/
		inline std::vector<std::string> get_layers() const {
			std::vector<std::string> layers;
			for (auto& [name, layer] : this->layers) {
				layers.push_back(name);
			}
			return layers;
		}

		/** Compares the values data buffers of two voxel buffers
		* @param other The other voxel buffer to compare with
		* @return True if the values data buffers of the two voxel buffers are equal, false otherwise
		*/
		bool operator ==(VoxelBuffer const& other) const;

		/** Reinitialize the data buffer of a layer with a new value
		* @param layer_name The name of the layer
		* @param new_value The new value to assign to each voxel of the layer
		*/
		template<typename dtype = float>
		void reinitialize_layer(const std::string& layer_name, const dtype& new_value) {
			auto found = this->layers.find(layer_name);
			if (found == this->layers.end())
				throw VoxelBufferException("Layer: '" + layer_name + "' not found");
			std::fill((dtype*)found->second.data, ((dtype*)found->second.data) + this->voxel_count, new_value);
		}

		/** Merge two layers data buffers directly together using a custom merge function
		* @param layer_name The name of the layer to merge into
		* @param other The other voxel buffer to merge with
		* @param merge_function The function to use to merge the two layers
		*/
		template<typename dtype = float>
		void merge_data_buffer(const std::string& layer_name, VoxelBuffer& other, const std::function<dtype(const dtype&, const dtype&)>& merge_function) {
			auto found = this->layers.find(layer_name);
			if (found == this->layers.end())
				throw VoxelBufferException("Layer: '" + layer_name + "' not found");

			auto other_layer = other.layers.find(layer_name);
			if (other_layer == other.layers.end())
				throw VoxelBufferException("Layer: '" + layer_name + "' not found in the other buffer");

			if (found->second.bytes_per_data_element != other_layer->second.bytes_per_data_element)
				throw VoxelBufferException("Layer: '" + layer_name + "' has different data element sizes");

			dtype* this_data = (dtype*)found->second.data;
			dtype* other_data = (dtype*)other_layer->second.data;

			for (size_t i = 0; i < this->voxel_count; i++) {
				this_data[i] = merge_function(this_data[i], other_data[i]);
			}
		}

		/** Merge layer voxels with a custom voxel type.
		* Will implicitly modify the data buffer, but through calling the voxel objects.
		* @param layer_name The name of the layer to merge into
		* @param other The other voxel buffer to merge with
		* @param merge_function The function to use to merge the two layers
		*/
		template<class VoxelT>
		void merge_voxel_buffer(const std::string& layer_name, VoxelBuffer& other, const std::function<VoxelT(const VoxelT&, const VoxelT&)>& merge_function) {
			auto found = this->layers.find(layer_name);
			if (found == this->layers.end())
				throw VoxelBufferException("Layer: '" + layer_name + "' not found");

			auto other_layer = other.layers.find(layer_name);
			if (other_layer == other.layers.end())
				throw VoxelBufferException("Layer: '" + layer_name + "' not found in the other buffer");

			if (found->second.bytes_per_data_element != other_layer->second.bytes_per_data_element)
				throw VoxelBufferException("Layer: '" + layer_name + "' has different data element sizes");

			VoxelT* this_data = (VoxelT*)found->second.voxels;
			VoxelT* other_data = (VoxelT*)other_layer->second.voxels;

			for (size_t i = 0; i < this->voxel_count; i++) {
				this_data[i] = merge_function(this_data[i], other_data[i]);
			}
		}

		/** Returns the type string of a layer
		* @param layer_name The name of the layer
		* @return The type string of the layer
		* @see Typing::Helper::get_dtype to safely convert the type string to a DType enumeration
		*/
		inline std::string get_type(const std::string& layer_name) const {
			auto found = this->layers.find(layer_name);
			if (found == this->layers.end())
				throw VoxelBufferException("Layer: '" + layer_name + "' not found");
			return ((IVoxel*)found->second.voxels)->get_type();
		}

		/** Create a deep copy of the voxel buffer
		* @return The deep copy of the voxel buffer
		* @note The copy will have the same number of voxels and the same data buffer values
		*/
		virtual VoxelBuffer* copy() const;

		VoxelBuffer& operator +=(const VoxelBuffer& other);
		VoxelBuffer& operator -=(const VoxelBuffer& other);
		VoxelBuffer& operator *=(const VoxelBuffer& other);
		VoxelBuffer& operator /=(const VoxelBuffer& other);
		VoxelBuffer& operator +=(const float& scalar);
		VoxelBuffer& operator -=(const float& scalar);
		VoxelBuffer& operator *=(const float& scalar);
		VoxelBuffer& operator /=(const float& scalar);
	};
};