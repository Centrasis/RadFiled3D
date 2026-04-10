#pragma once
#include <string>
#include <vector>
#include <glm/vec3.hpp>
#include "RadFiled3D/helpers/Typing.hpp"
#include <cstring>
#include <cmath>
#include <span>


namespace RadFiled3D {
	/** A Voxel is a single element in a VoxelBuffer. It can be of any type, but must be able to be converted to a raw byte array.
	* Voxels do not store their own data, but rather point to a buffer that contains the data. This is to allow for the data to be
	* stored in a single contiguous block of memory, which is more efficient for the GPU to read. The buffer is managed by the VoxelBuffer
	* class, and the Voxel class is only a view into the buffer. This means that the Voxel class does not own the data, and should not
	* be used to manage the data. The VoxelBuffer class is responsible for managing the data, and the Voxel class is only a view into
	* that data. The VoxelBuffer class is responsible for allocating and deallocating the data, and the Voxel class is only responsible
	* for reading and writing the data.
	*/
	class IVoxel {
	public:
#pragma pack(push, 4)
		struct VoxelBaseHeader {
			size_t header_bytes;
			void* header;

			VoxelBaseHeader(size_t header_bytes = 0, void* header = nullptr) : header_bytes(header_bytes), header(header) { }
		};
#pragma pack(pop)

		friend class VoxelBuffer;
	public:
		/** Sets the data pointer of the voxel to the given pointer
		* WARNING: This function should only be called by the VoxelBuffer class and it's subclasses. It is not safe to call this function directly.
		* @param data The pointer to the data
		* @return The voxel with the new data pointer
		*/
		virtual void set_data(void* data) = 0;
		/** Returns the size of the voxel data in bytes
		* @return The size of the voxel data in bytes
		*/
		virtual size_t get_bytes() const = 0;

		/** Returns the size of the voxel in bytes, excluding the data-size.
		* @return The size of the voxel in bytes
		*/
		virtual size_t get_voxel_bytes() const = 0;

		/** Returns a raw pointer to the data
		* @return A raw pointer to the data
		*/
		virtual void* get_raw() const = 0;

		/** Returns the type of the voxel as a string
		* @return The type of the voxel as a string
		* @return The type of the voxel as a string
		*/
		virtual std::string get_type() const = 0;

		/** Returns the header of a Voxel to reconstruct it after serialization.
		* Header struct should be extended by the Voxel implementation.
		* @see HistogramVoxel::HistogramDefinition
		* @see HistogramVoxel::get_header
		* @return The header of the Voxel
		*/
		virtual VoxelBaseHeader get_header() const = 0;

		/** Initializes the Voxel from a header block. For deserialization only!
		* @see HistogramVoxel::init_from_header
		* @param header The header to initialize the Voxel from
		*/
		virtual void init_from_header(const void* header) = 0;

		//virtual ~IVoxel() {}
	};

	/** A ScalarVoxel is a Voxel that contains a single scalar value. It is a simple wrapper around a single value, and is used to
	* represent a single value in a VoxelBuffer. It provides a simple interface for reading and writing the value, and can be used
	* in conjunction with the VoxelBuffer class to store and manipulate voxel data.
	* @tparam T The type of the scalar value to store in the voxel
	*/
	template<typename T = float>
	class ScalarVoxel : public IVoxel {
		friend class VoxelBuffer;
		friend class VoxelLayer;
	protected:
		static VoxelBaseHeader voxel_header;
		T* data;

		ScalarVoxel() : data(nullptr) { }
	public:
		/** Sets the data pointer of the voxel to the given pointer
		* WARNING: This function should only be called by the VoxelBuffer class and it's subclasses. It is not safe to call this function directly.
		* @param data The pointer to the data
		* @return The voxel with the new data pointer
		*/
		virtual void set_data(void* data) override {
			this->data = (T*)data;
		}

		//virtual ~ScalarVoxel() override {}

		/** Returns the reference to the value of the voxel
		* @return The reference to the value of the voxel
		*/
		inline T& get_data() const { return *this->data; }

		/** Returns the size of the voxel value in bytes
		* @return The size of the voxel value in bytes
		* @return The size of the voxel value in bytes
		*/
		virtual size_t get_bytes() const override { return sizeof(T); }

		/** Returns the size of the voxel in bytes, excluding the data-size.
		* @return The size of the voxel in bytes
		*/
		virtual size_t get_voxel_bytes() const override { return sizeof(ScalarVoxel<T>); }

		/** Returns a raw pointer to the value data
		* @return A raw pointer to the value data
		*/
		virtual void* get_raw() const override { return this->data; }

		/** Returns the type of the voxel as a string
		* @return The type of the voxel as a string
		* @return The type of the voxel as a string
		*/
		virtual std::string get_type() const override {
			return Typing::Helper::get_plain_type_name<T>();
		}

		/** Returns the header of a Voxel to reconstruct it after serialization.
		* Header struct should be extended by the Voxel implementation.
		* @return The header of the Voxel
		*/
		virtual VoxelBaseHeader get_header() const override {
			return VoxelBaseHeader();
		}

		/** Initializes the Voxel from a header block. For deserialization only!
		* @param header The header to initialize the Voxel from
		*/
		virtual void init_from_header(const void* header) override {};

		/** Create a new ScalarVoxel with the given data buffer
		* @param data_buffer The data buffer to use
		* @return A new ScalarVoxel with the given data buffer
		*/
		ScalarVoxel(T* data_buffer) : data(data_buffer) {}


		/** Create a new ScalarVoxel from an existing ScalarVoxel
		* @param buffer The ScalarVoxel to copy
		* @return A new ScalarVoxel with the same data as the given ScalarVoxel
		*/
		ScalarVoxel(ScalarVoxel<T>&& buffer) noexcept : data(buffer.data) {}

		/** Create a new ScalarVoxel from an existing ScalarVoxel
		* @param buffer The ScalarVoxel to copy
		* @return A new ScalarVoxel with the same data as the given ScalarVoxel
		*/
		ScalarVoxel(const ScalarVoxel<T>& buffer) : data(buffer.data) {}

		/** Allows to assign a value to the voxel
		* @param o The value to assign to the voxel
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator=(const T& o) {
			*this->data = o;
			return *this;
		}

		/** Allows to assign a value to the voxel
		* @param o The value to assign to the voxel
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator=(T&& o) {
			*this->data = o;
			return *this;
		}

		/** Allows to add a value to the voxel
		* @param o The value to add to the voxel
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator+=(const T& o) {
			*this->data += o;
			return *this;
		}

		/** Allows to subtract a value from the voxel
		* @param o The value to add to the voxel
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator-=(const T& o) {
			*this->data -= o;
			return *this;
		}

		ScalarVoxel<T> operator+(const T& o) const {
			ScalarVoxel<T> result = *this;
			result += o;
			return result;
		}

		ScalarVoxel<T> operator-(const T& o) const {
			ScalarVoxel<T> result = *this;
			result -= o;
			return result;
		}

		ScalarVoxel<T> operator*(const T& o) const {
			ScalarVoxel<T> result = *this;
			result *= o;
			return result;
		}

		ScalarVoxel<T> operator/(const T& o) const {
			ScalarVoxel<T> result = *this;
			result /= o;
			return result;
		}

		ScalarVoxel<T> operator*(float o) const {
			ScalarVoxel<T> result = *this;
			result *= o;
			return result;
		}

		ScalarVoxel<T> operator/(float o) const {
			ScalarVoxel<T> result = *this;
			result /= o;
			return result;
		}

		ScalarVoxel<T>& operator*=(float o) {
			*this->data *= o;
			return *this;
		}

		/** Allows to add a value to the voxel
		* @param o The value to add to the voxel
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator+=(T&& o) {
			*this->data += o;
			return *this;
		}

		/** Allows to subtract a value from the voxel
		* @param o The value to add to the voxel
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator-=(T&& o) {
			*this->data -= o;
			return *this;
		}

		/** Allows to multiply the voxel by a value
		* @param o The value to multiply the voxel by
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator*=(T&& o) {
			*this->data *= o;
			return *this;
		}

		/** Allows to multiply the voxel by a value
		* @param o The value to multiply the voxel by
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator*=(const T& o) {
			*this->data *= o;
			return *this;
		}

		/** Allows to divide the voxel by a value
		* @param o The value to divide the voxel by
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator/=(T&& o) {
			*this->data /= o;
			return *this;
		}

		/** Allows to divide the voxel by a value
		* @param o The value to divide the voxel by
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator/=(const T& o) {
			*this->data /= o;
			return *this;
		}

		/** Allows to add a value to the voxel
		* @param o The value to add to the voxel
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator+=(const ScalarVoxel<T>& o) {
			*this->data += *o.data;
			return *this;
		}

		/** Allows to subtract a value from the voxel
		* @param o The value to add to the voxel
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator-=(const ScalarVoxel<T>& o) {
			*this->data -= *o.data;
			return *this;
		}

		/** Allows to add a value to the voxel
		* @param o The value to add to the voxel
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator+=(ScalarVoxel<T>&& o) {
			*this->data += *o.data;
			return *this;
		}

		/** Allows to subtract a value from the voxel
		* @param o The value to add to the voxel
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator-=(ScalarVoxel<T>&& o) {
			*this->data -= *o.data;
			return *this;
		}

		/** Allows to multiply the voxel by a value
		* @param o The value to multiply the voxel by
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator*=(ScalarVoxel<T>&& o) {
			*this->data *= *o.data;
			return *this;
		}

		/** Allows to multiply the voxel by a value
		* @param o The value to multiply the voxel by
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator*=(const ScalarVoxel<T>& o) {
			*this->data *= *o.data;
			return *this;
		}

		/** Allows to divide the voxel by a value
		* @param o The value to divide the voxel by
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator/=(ScalarVoxel<T>&& o) {
			*this->data /= *o.data;
			return *this;
		}

		/** Allows to divide the voxel by a value
		* @param o The value to divide the voxel by
		* @return The voxel with the new value
		*/
		ScalarVoxel<T>& operator/=(const ScalarVoxel<T>& o) {
			*this->data /= *o.data;
			return *this;
		}

		/** Default move assignment operator
		* @param other The ScalarVoxel to move
		* @return The moved ScalarVoxel
		*/
		ScalarVoxel<T>& operator=(const ScalarVoxel<T>& other) = default;

		/** Compares two ScalarVoxels values for equality
		* @param other The ScalarVoxel to compare to
		* @return True if the values are equal, false otherwise
		*/
		bool operator ==(ScalarVoxel<T> const& other) const {
			return *this->data == *other.data;
		}
	};

	template<typename T>
	IVoxel::VoxelBaseHeader ScalarVoxel<T>::voxel_header = IVoxel::VoxelBaseHeader(Typing::Helper::get_plain_type_name<T>());

	/** An OwningScalarVoxel is a ScalarVoxel that owns the data it points to. It is a simple wrapper around a single value, and is used to
	* represent a single value in a VoxelBuffer. It provides a simple interface for reading and writing the value, and can be used
	* in conjunction with the VoxelBuffer class to store and manipulate voxel data. The OwningScalarVoxel class is responsible for
	* managing the data, and will automatically deallocate the data when the OwningScalarVoxel is destroyed.
	*/
	template<typename T = float>
	class OwningScalarVoxel : public ScalarVoxel<T> {
	protected:
		T physical_data;
	public:
		//virtual ~OwningScalarVoxel() override {}

		OwningScalarVoxel() : ScalarVoxel<T>(&this->physical_data) {}

		OwningScalarVoxel(T* data) : ScalarVoxel<T>(&this->physical_data) {
			this->physical_data = *data;
		}

		OwningScalarVoxel(const OwningScalarVoxel<T>& buffer) : ScalarVoxel<T>(&this->physical_data) {
			this->physical_data = buffer.physical_data;
		}

		OwningScalarVoxel(OwningScalarVoxel<T>&& buffer) : ScalarVoxel<T>(&this->physical_data) {
			this->physical_data = buffer.physical_data;
		}

		virtual void set_data(void* data) override {
			this->physical_data = *(T*)data;
		}
	};

	/** A HistogramVoxel is a Voxel that contains a histogram of scalar values. It is a simple wrapper around a buffer of values, and is used to
	* represent a histogram in a VoxelBuffer. It provides a simple interface for reading and writing the histogram, and can be used
	* in conjunction with the VoxelBuffer class to store and manipulate voxel data.
	*/
	class HistogramVoxel : public ScalarVoxel<float> {
	public:
#pragma pack(push, 4)
		struct HistogramDefinition : VoxelBaseHeader {
			float histogram_bin_width;
			size_t bins;

			HistogramDefinition(size_t bins = 0, float histogram_bin_width = 0.f) : histogram_bin_width(histogram_bin_width), bins(bins) { }
		};
#pragma pack(pop)

	protected:
		HistogramDefinition histogram_definition;

	public:
		/** Create a new HistogramVoxel with an empty data buffer
		* @return A new HistogramVoxel with an empty data buffer
		*/
		HistogramVoxel() noexcept : ScalarVoxel<float>(nullptr), histogram_definition(HistogramDefinition()) {}
		/** Create a new HistogramVoxel with the given data buffer
		* @param bins The number of bins in the histogram
		* @param histogram_bin_width The width of each bin in the histogram
		* @param buffer The data buffer to use
		* @return A new HistogramVoxel with the given data buffer
		*/
		HistogramVoxel(size_t bins, float histogram_bin_width, float* buffer) : ScalarVoxel<float>(buffer), histogram_definition(bins, histogram_bin_width) {}

		/** Create a new HistogramVoxel from an existing HistogramVoxel
		* @param buffer The HistogramVoxel to copy
		* @return A new HistogramVoxel with the same data as the given HistogramVoxel
		*/
		HistogramVoxel(HistogramVoxel&& buffer) noexcept : ScalarVoxel<float>(buffer.data), histogram_definition(HistogramDefinition()) {}

		/** Create a new HistogramVoxel from an existing HistogramVoxel
		* @param buffer The HistogramVoxel to copy
		* @return A new HistogramVoxel with the same data as the given HistogramVoxel
		*/
		HistogramVoxel(const HistogramVoxel& buffer) noexcept : ScalarVoxel<float>(buffer.data), histogram_definition(buffer.histogram_definition) {}

		/** Returns a vector containing the histogram data
		* @return A vector containing the histogram data
		*/
		inline std::span<float> get_histogram() const {
			return std::span<float>(this->data, this->histogram_definition.bins);
		}

		/** Returns the width of each bin in the histogram
		* @return The width of each bin in the histogram
		*/
		inline float get_histogram_bin_width() const { return this->histogram_definition.histogram_bin_width; }

		/** Returns the number of bins in the histogram
		* @return The number of bins in the histogram
		*/
		inline size_t get_bins() const { return this->histogram_definition.bins; }

		/** Returns the type of the voxel as a string
		* @return The type of the voxel as a string
		*/
		virtual std::string get_type() const override {
			return "histogram";
		}

		/** Returns the size of the voxel in bytes, excluding the data-size.
		* @return The size of the voxel in bytes
		*/
		virtual size_t get_voxel_bytes() const override { return sizeof(HistogramVoxel); }

		/** Returns the size of the voxel in bytes
		* @return The size of the voxel in bytes
		*/
		virtual size_t get_bytes() const override { return sizeof(float) * this->histogram_definition.bins; }

		/** Returns the reference to the first element in the data buffer
		* @return The reference to the first element in the data buffer
		*/
		inline float& get_data() const { return *this->data; }

		/** Returns a raw pointer to the first element in the data buffer
		* @return A raw pointer to the first element in the data buffer
		*/
		virtual void* get_raw() const override { return this->data; }

		/** Default move assignment operator
		* @param other The HistogramVoxel to move
		* @return The moved HistogramVoxel
		*/
		HistogramVoxel& operator=(const HistogramVoxel& other) = default;

		/** Inplace Adds two HistogramVoxels together
		* @param other The other HistogramVoxel to add
		* @return The sum of the two HistogramVoxels
		*/
		HistogramVoxel& operator+=(const HistogramVoxel& other) {
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				this->data[i] += other.data[i];
			}
			return *this;
		}

		/** Adds two HistogramVoxels together
		* @param other The other HistogramVoxel to add
		* @return The sum of the two HistogramVoxels
		*/
		HistogramVoxel operator+(const HistogramVoxel& other) const {
			HistogramVoxel result = *this;
			result += other;
			return result;
		}

		/** Inplace Subtracts a HistogramVoxel from another
		* @param other The other HistogramVoxel to subtract
		* @return The difference of the two HistogramVoxels
		*/
		HistogramVoxel& operator-=(const HistogramVoxel& other) {
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				this->data[i] -= other.data[i];
			}
			return *this;
		}

		/** Subtracts a HistogramVoxel from another
		* @param other The other HistogramVoxel to subtract
		* @return The difference of the two HistogramVoxels
		*/
		HistogramVoxel operator-(const HistogramVoxel& other) const {
			HistogramVoxel result = *this;
			result -= other;
			return result;
		}

		/** Inplace Multiplies two HistogramVoxels together
		* @param other The other HistogramVoxel to multiply
		* @return The product of the two HistogramVoxels
		*/
		HistogramVoxel& operator*=(const HistogramVoxel& other) {
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				this->data[i] *= other.data[i];
			}
			return *this;
		}

		/** Multiplies two HistogramVoxels together
		* @param other The other HistogramVoxel to multiply
		* @return The product of the two HistogramVoxels
		*/
		HistogramVoxel operator*(const HistogramVoxel& other) const {
			HistogramVoxel result = *this;
			result *= other;
			return result;
		}

		/** Inplace Divides a HistogramVoxel by another
		* @param other The other HistogramVoxel to divide by
		* @return The quotient of the two HistogramVoxels
		*/
		HistogramVoxel& operator/=(const HistogramVoxel& other) {
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				if (other.data[i] == 0.f) {
					this->data[i] = 0.f;
					continue;
				}
				this->data[i] /= other.data[i];
			}
			return *this;
		}

		/** Divides a HistogramVoxel by another
		* @param other The other HistogramVoxel to divide by
		* @return The quotient of the two HistogramVoxels
		*/
		HistogramVoxel operator/(const HistogramVoxel& other) const {
			HistogramVoxel result = *this;
			result /= other;
			return result;
		}

		/** Inplace Adds a hiostogram and a scalar value together
		* @param other The scalar value to add
		* @return The sum of the histogram and the scalar value
		*/
		HistogramVoxel& operator+=(float other) {
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				this->data[i] += other;
			}
			return *this;
		}

		/** Adds a hiostogram and a scalar value together
		* @param other The scalar value to add
		* @return The sum of the histogram and the scalar value
		*/
		HistogramVoxel operator+(float other) const {
			HistogramVoxel result = *this;
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				result.data[i] += other;
			}
			return result;
		}

		/** Inplace Subtracts a scalar value from a histogram
		* @param other The scalar value to subtract
		* @return The difference of the histogram and the scalar value
		*/
		HistogramVoxel& operator-=(float other) {
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				this->data[i] -= other;
			}
			return *this;
		}

		/** Subtracts a scalar value from a histogram
		* @param other The scalar value to subtract
		* @return The difference of the histogram and the scalar value
		*/
		HistogramVoxel operator-(float other) const {
			HistogramVoxel result = *this;
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				result.data[i] -= other;
			}
			return result;
		}

		/** Inplace Multiplies a histogram by a scalar value
		* @param other The scalar value to multiply
		* @return The product of the histogram and the scalar value
		*/
		HistogramVoxel& operator*=(float other) {
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				this->data[i] *= other;
			}
			return *this;
		}

		/** Multiplies a histogram by a scalar value
		* @param other The scalar value to multiply
		* @return The product of the histogram and the scalar value
		*/
		HistogramVoxel operator*(float other) const {
			HistogramVoxel result = *this;
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				result.data[i] *= other;
			}
			return result;
		}

		/** Inplace Divides a histogram by a scalar value
		* @param other The scalar value to divide by
		* @return The quotient of the histogram and the scalar value
		*/
		HistogramVoxel& operator/=(float other) {
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				this->data[i] /= other;
			}
			return *this;
		}

		/** Divides a histogram by a scalar value
		* @param other The scalar value to divide by
		* @return The quotient of the histogram and the scalar value
		*/
		HistogramVoxel operator/(float other) const {
			HistogramVoxel result = *this;
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				result.data[i] /= other;
			}
			return result;
		}

		/** Returns the header of a Voxel to reconstruct it after serialization.
		* Header struct should be extended by the Voxel implementation.
		* @return The header of the Voxel
		*/
		virtual VoxelBaseHeader get_header() const override {
			return VoxelBaseHeader(sizeof(HistogramDefinition), (void*)&this->histogram_definition);
		}

		/** Initializes the Voxel from a header block. For deserialization only!
		* @param header The header to initialize the Voxel from
		*/
		virtual void init_from_header(const void* header) override {
			HistogramDefinition* hist_def = (HistogramDefinition*)header;
			this->histogram_definition = *hist_def;
		}

		/** Adds a positive value to the histogram and scores it into the correct bin.
		* The correct bin is determined by dividing the value by the bin width and rounding to the nearest integer.
		* If the value is greater than the maximum value, it is scored in the last bin.
		* If the value is less than 0, it is scored in the first bin.
		* @param value The value to add to the histogram
		*/
		void add_value(float value) {
			size_t bin = (size_t)(value >= 0.f) ? ((value + this->histogram_definition.histogram_bin_width / 2.f) / this->histogram_definition.histogram_bin_width) : 0;
			if (bin >= this->histogram_definition.bins) {
				bin = this->histogram_definition.bins - 1;
			}
			this->data[bin]++;
		}

		/** Normalizes the histogram so that the sum of all bins is 1, if possible.
		*/
		void normalize() {
			float sum = 0.f;
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				sum += this->data[i];
			}
			if (sum == 0.f) {
				return;
			}
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				this->data[i] /= sum;
			}
		}

		/** Clears the histogram by setting all bins to 0 */
		void clear() {
			std::fill(this->data, this->data + this->histogram_definition.bins, 0.0f);
		}
	};

	/** Owning version of the HistogramVoxel class. This class owns the data it points to, and will automatically deallocate the data when the OwningHistogramVoxel is destroyed.
	*/
	class OwningHistogramVoxel : public HistogramVoxel {
	public:
		OwningHistogramVoxel(size_t bins = 0, float histogram_bin_width = 0.f) : HistogramVoxel(bins, histogram_bin_width, (bins > 0) ? new float[bins] : nullptr) {}

		OwningHistogramVoxel(size_t bins, float histogram_bin_width, float* buffer) : HistogramVoxel(bins, histogram_bin_width, (bins > 0) ? new float[bins] : nullptr) {
			if (bins > 0)
				memcpy(this->data, buffer, bins * sizeof(float));
		}

		OwningHistogramVoxel(const OwningHistogramVoxel& buffer) : HistogramVoxel(buffer) {
			if (this->histogram_definition.bins > 0) {
				this->data = new float[this->histogram_definition.bins];
				memcpy(this->data, buffer.data, this->histogram_definition.bins * sizeof(float));
			}
		}

		OwningHistogramVoxel(OwningHistogramVoxel&& buffer) noexcept : HistogramVoxel(buffer) {
			if (this->histogram_definition.bins > 0) {
				this->data = new float[this->histogram_definition.bins];
				memcpy(this->data, buffer.data, this->histogram_definition.bins * sizeof(float));
			}
		}

		~OwningHistogramVoxel() {
			if (this->data != nullptr)
				delete[] this->data;
		}

		virtual void init_from_header(const void* header) override {
			HistogramDefinition* hist_def = (HistogramDefinition*)header;
			this->histogram_definition = *hist_def;
			if (this->data != nullptr) {
				delete[] this->data;
			}
			this->data = new float[this->histogram_definition.bins];
		}

		virtual void set_data(void* data) override {
			memcpy(this->data, data, this->histogram_definition.bins * sizeof(float));
		}
	};

	class SphericalVoxel : public ScalarVoxel<float> {
	public:
#pragma pack(push, 4)
		struct SphericalDefinition : VoxelBaseHeader {
			size_t phi_segments;
			size_t theta_segments;

			SphericalDefinition(size_t phi_segments = 0, size_t theta_segments = 0) : phi_segments(phi_segments), theta_segments(theta_segments) {}
		};
#pragma pack(pop)

	protected:
		SphericalDefinition spherical_definition;

		/** Compute flat index from phi/theta grid indices */
		inline size_t calc_segment_idx(size_t phi_idx, size_t theta_idx) const {
			return theta_idx * this->spherical_definition.phi_segments + phi_idx;
		}

		/** Compute flat index from phi/theta in standard spherical coordinates (linear mapping)
		* @param phi Azimuthal angle in radians [0, 2*pi]
		* @param theta Polar angle in radians [0, pi]
		*/
		inline size_t calc_segment_idx_by_coord(float phi, float theta) const {
			const float pi = 3.14159265358979323846f;
			// Wrap phi to [0, 2*pi]
			if (phi < 0.f) phi += 2.f * pi;
			if (phi >= 2.f * pi) phi -= 2.f * pi;
			// Clamp theta to [0, pi]
			if (theta < 0.f) theta = 0.f;
			if (theta > pi) theta = pi;

			size_t phi_idx = static_cast<size_t>(phi / (2.f * pi) * this->spherical_definition.phi_segments);
			size_t theta_idx = static_cast<size_t>(theta / pi * this->spherical_definition.theta_segments);

			if (phi_idx >= this->spherical_definition.phi_segments) phi_idx = this->spherical_definition.phi_segments - 1;
			if (theta_idx >= this->spherical_definition.theta_segments) theta_idx = this->spherical_definition.theta_segments - 1;

			return calc_segment_idx(phi_idx, theta_idx);
		}

	public:
		/** Create a new SphericalVoxel with an empty data buffer */
		SphericalVoxel() noexcept : ScalarVoxel<float>(nullptr), spherical_definition(SphericalDefinition()) {}

		/** Create a new SphericalVoxel with the given data buffer */
		SphericalVoxel(size_t phi_segments, size_t theta_segments, float* buffer) : ScalarVoxel<float>(buffer), spherical_definition(phi_segments, theta_segments) {}

		/** Move constructor */
		SphericalVoxel(SphericalVoxel&& buffer) noexcept : ScalarVoxel<float>(buffer.data), spherical_definition(buffer.spherical_definition) {
			buffer.data = nullptr;
		}

		/** Copy constructor */
		SphericalVoxel(const SphericalVoxel& buffer) noexcept : ScalarVoxel<float>(buffer.data), spherical_definition(buffer.spherical_definition) {}

		/** Returns the total number of segments (phi * theta) */
		inline size_t get_total_segments() const { return this->spherical_definition.phi_segments * this->spherical_definition.theta_segments; }

		/** Returns the number of phi segments */
		inline size_t get_phi_segments() const { return this->spherical_definition.phi_segments; }

		/** Returns the number of theta segments */
		inline size_t get_theta_segments() const { return this->spherical_definition.theta_segments; }

		/** Returns a span containing all segment data */
		inline std::span<float> get_segments_data() const {
			return std::span<float>(this->data, this->get_total_segments());
		}

		/** Access a segment value by spherical coordinates
		* @param phi The phi coordinate in radians
		* @param theta The theta coordinate in radians
		* @return Reference to the segment value
		*/
		inline float& get_value_by_coord(float phi, float theta) const {
			return this->data[this->calc_segment_idx_by_coord(phi, theta)];
		}

		/** Access a segment value by grid indices
		* @param phi_idx The phi index [0, phi_segments - 1]
		* @param theta_idx The theta index [0, theta_segments - 1]
		* @return Reference to the segment value
		*/
		inline float& get_value(size_t phi_idx, size_t theta_idx) const {
			return this->data[this->calc_segment_idx(phi_idx, theta_idx)];
		}

		/** Returns the type of the voxel as a string */
		virtual std::string get_type() const override {
			return "spherical";
		}

		/** Returns the size of the voxel object in bytes, excluding the data */
		virtual size_t get_voxel_bytes() const override { return sizeof(SphericalVoxel); }

		/** Returns the size of the data in bytes */
		virtual size_t get_bytes() const override { return sizeof(float) * this->get_total_segments(); }

		/** Returns the reference to the first element in the data buffer */
		inline float& get_data() const { return *this->data; }

		/** Returns a raw pointer to the data buffer */
		virtual void* get_raw() const override { return this->data; }

		/** Default assignment operator */
		SphericalVoxel& operator=(const SphericalVoxel& other) = default;

		/** Inplace add two SphericalVoxels */
		SphericalVoxel& operator+=(const SphericalVoxel& other) {
			for (size_t i = 0; i < this->get_total_segments(); i++)
				this->data[i] += other.data[i];
			return *this;
		}

		/** Adds two SphericalVoxels together
		* @param other The other SphericalVoxel to add
		* @return The sum of the two SphericalVoxels
		*/
		SphericalVoxel operator+(const SphericalVoxel& other) const {
			SphericalVoxel result = *this;
			result += other;
			return result;
		}

		/** Inplace subtract two SphericalVoxels */
		SphericalVoxel& operator-=(const SphericalVoxel& other) {
			for (size_t i = 0; i < this->get_total_segments(); i++)
				this->data[i] -= other.data[i];
			return *this;
		}

		/** Subtracts a SphericalVoxel from another
		* @param other The other SphericalVoxel to subtract
		* @return The difference of the two SphericalVoxels
		*/
		SphericalVoxel operator-(const SphericalVoxel& other) const {
			SphericalVoxel result = *this;
			result -= other;
			return result;
		}

		/** Inplace multiply two SphericalVoxels */
		SphericalVoxel& operator*=(const SphericalVoxel& other) {
			for (size_t i = 0; i < this->get_total_segments(); i++)
				this->data[i] *= other.data[i];
			return *this;
		}

		/** Multiplies two SphericalVoxels together
		* @param other The other SphericalVoxel to multiply
		* @return The product of the two SphericalVoxels
		*/
		SphericalVoxel operator*(const SphericalVoxel& other) const {
			SphericalVoxel result = *this;
			result *= other;
			return result;
		}

		/** Inplace divide two SphericalVoxels */
		SphericalVoxel& operator/=(const SphericalVoxel& other) {
			for (size_t i = 0; i < this->get_total_segments(); i++) {
				if (other.data[i] == 0.f) {
					this->data[i] = 0.f;
					continue;
				}
				this->data[i] /= other.data[i];
			}
			return *this;
		}

		/** Divides a SphericalVoxel by another
		* @param other The other SphericalVoxel to divide by
		* @return The quotient of the two SphericalVoxels
		*/
		SphericalVoxel operator/(const SphericalVoxel& other) const {
			SphericalVoxel result = *this;
			result /= other;
			return result;
		}

		/** Inplace adds a scalar value to all segments
		* @param other The scalar value to add
		* @return The modified SphericalVoxel
		*/
		SphericalVoxel& operator+=(float other) {
			for (size_t i = 0; i < this->get_total_segments(); i++)
				this->data[i] += other;
			return *this;
		}

		/** Inplace subtracts a scalar value from all segments */
		SphericalVoxel& operator-=(float other) {
			for (size_t i = 0; i < this->get_total_segments(); i++)
				this->data[i] -= other;
			return *this;
		}

		/** Inplace multiplies all segments by a scalar value */
		SphericalVoxel& operator*=(float other) {
			for (size_t i = 0; i < this->get_total_segments(); i++)
				this->data[i] *= other;
			return *this;
		}

		/** Inplace divides all segments by a scalar value */
		SphericalVoxel& operator/=(float other) {
			for (size_t i = 0; i < this->get_total_segments(); i++)
				this->data[i] /= other;
			return *this;
		}

		/** Non-inplace scalar arithmetic operators */
		SphericalVoxel operator+(float other) const { SphericalVoxel r = *this; r += other; return r; }
		SphericalVoxel operator-(float other) const { SphericalVoxel r = *this; r -= other; return r; }
		SphericalVoxel operator*(float other) const { SphericalVoxel r = *this; r *= other; return r; }
		SphericalVoxel operator/(float other) const { SphericalVoxel r = *this; r /= other; return r; }

		/** Returns the header for serialization */
		virtual VoxelBaseHeader get_header() const override {
			return VoxelBaseHeader(sizeof(SphericalDefinition), (void*)&this->spherical_definition);
		}

		/** Initializes the Voxel from a header block (deserialization) */
		virtual void init_from_header(const void* header) override {
			SphericalDefinition* sph_def = (SphericalDefinition*)header;
			this->spherical_definition = *sph_def;
		}

		/** Adds a value at the given spherical direction
		* @param phi The phi coordinate in radians
		* @param theta The theta coordinate in radians
		* @param value The value to add
		*/
		void add_value(float phi, float theta, float value = 1.f) {
			this->data[this->calc_segment_idx_by_coord(phi, theta)] += value;
		}

		/** Clears all segments to 0 */
		void clear() {
			std::fill(this->data, this->data + this->get_total_segments(), 0.0f);
		}
	};

	/** Owning version of the SphericalVoxel class. Owns the data buffer and frees it on destruction. */
	class OwningSphericalVoxel : public SphericalVoxel {
	public:
		OwningSphericalVoxel(size_t phi_segments = 0, size_t theta_segments = 0) : SphericalVoxel(phi_segments, theta_segments, (phi_segments * theta_segments > 0) ? new float[phi_segments * theta_segments]() : nullptr) {}

		OwningSphericalVoxel(size_t phi_segments, size_t theta_segments, float* buffer) : SphericalVoxel(phi_segments, theta_segments, (phi_segments * theta_segments > 0) ? new float[phi_segments * theta_segments] : nullptr) {
			if (phi_segments * theta_segments > 0)
				memcpy(this->data, buffer, phi_segments * theta_segments * sizeof(float));
		}

		OwningSphericalVoxel(const OwningSphericalVoxel& buffer) : SphericalVoxel(buffer) {
			size_t total = this->get_total_segments();
			if (total > 0) {
				this->data = new float[total];
				memcpy(this->data, buffer.data, total * sizeof(float));
			}
		}

		OwningSphericalVoxel(OwningSphericalVoxel&& buffer) noexcept : SphericalVoxel(buffer) {
			size_t total = this->get_total_segments();
			if (total > 0) {
				this->data = new float[total];
				memcpy(this->data, buffer.data, total * sizeof(float));
			}
		}

		~OwningSphericalVoxel() {
			if (this->data != nullptr)
				delete[] this->data;
		}

		virtual void init_from_header(const void* header) override {
			SphericalDefinition* sph_def = (SphericalDefinition*)header;
			this->spherical_definition = *sph_def;
			if (this->data != nullptr)
				delete[] this->data;
			this->data = new float[sph_def->phi_segments * sph_def->theta_segments];
		}

		virtual void set_data(void* data) override {
			memcpy(this->data, data, this->get_total_segments() * sizeof(float));
		}
	};
};