#pragma once
#include <string>
#include <vector>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include "RadFiled3D/helpers/Typing.hpp"
#include <cstring>
#include <cmath>
#include <span>
#include <numbers>


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
		ScalarVoxel<T>& operator=(const ScalarVoxel<T>& other) = default;

		operator T& () { return *this->data; }
		operator const T& () const { return *this->data; }


		ScalarVoxel<T>& operator+=(const T& rhs) {
			(*this->data) += rhs;
			return *this;
		}

		ScalarVoxel<T>& operator-=(const T& rhs) {
			(*this->data) -= rhs;
			return *this;
		}

		ScalarVoxel<T>& operator*=(const T& rhs) {
			(*this->data) *= rhs;
			return *this;
		}

		ScalarVoxel<T>& operator/=(const T& rhs) {
			(*this->data) /= rhs;
			return *this;
		}

		ScalarVoxel<T> operator/=(const ScalarVoxel<T>& rhs) {
			*this->data /= *rhs.data;
			return *this;
		}

		ScalarVoxel<T> operator-=(const ScalarVoxel<T>& rhs) {
			*this->data -= *rhs.data;
			return *this;
		}

		ScalarVoxel<T> operator+=(const ScalarVoxel<T>& rhs) {
			*this->data += *rhs.data;
			return *this;
		}

		ScalarVoxel<T> operator*=(const ScalarVoxel<T>& rhs) {
			*this->data *= *rhs.data;
			return *this;
		}
		

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
	template<typename T = float>
	class HistogramVoxel : public ScalarVoxel<T> {
		using typename IVoxel::VoxelBaseHeader;
	public:
#pragma pack(push, 4)
		struct HistogramDefinition : public IVoxel::VoxelBaseHeader {
			T histogram_bin_width;
			size_t bins;

			HistogramDefinition(size_t bins = 0, T histogram_bin_width = T(0)) : histogram_bin_width(histogram_bin_width), bins(bins) { }
		};
#pragma pack(pop)

	protected:
		HistogramDefinition histogram_definition;

	public:
		/** Create a new HistogramVoxel with an empty data buffer
		* @return A new HistogramVoxel with an empty data buffer
		*/
		HistogramVoxel() noexcept : ScalarVoxel<T>(nullptr), histogram_definition(HistogramDefinition()) {}
		/** Create a new HistogramVoxel with the given data buffer
		* @param bins The number of bins in the histogram
		* @param histogram_bin_width The width of each bin in the histogram
		* @param buffer The data buffer to use
		* @return A new HistogramVoxel with the given data buffer
		*/
		HistogramVoxel(size_t bins, const T& histogram_bin_width, T* buffer) : ScalarVoxel<T>(buffer), histogram_definition(bins, histogram_bin_width) {}

		/** Create a new HistogramVoxel from an existing HistogramVoxel
		* @param buffer The HistogramVoxel to copy
		* @return A new HistogramVoxel with the same data as the given HistogramVoxel
		*/
		HistogramVoxel(HistogramVoxel&& buffer) noexcept : ScalarVoxel<T>(buffer.data), histogram_definition(HistogramDefinition()) {}

		/** Create a new HistogramVoxel from an existing HistogramVoxel
		* @param buffer The HistogramVoxel to copy
		* @return A new HistogramVoxel with the same data as the given HistogramVoxel
		*/
		HistogramVoxel(const HistogramVoxel& buffer) noexcept : ScalarVoxel<T>(buffer.data), histogram_definition(buffer.histogram_definition) {}

		/** Returns a vector containing the histogram data
		* @return A vector containing the histogram data
		*/
		inline std::span<T> get_histogram() const {
			return std::span<T>(this->data, this->histogram_definition.bins);
		}

		/** Returns the width of each bin in the histogram
		* @return The width of each bin in the histogram
		*/
		inline T get_histogram_bin_width() const { return this->histogram_definition.histogram_bin_width; }

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
		virtual size_t get_bytes() const override { return sizeof(T) * this->histogram_definition.bins; }

		/** Returns the reference to the first element in the data buffer
		* @return The reference to the first element in the data buffer
		*/
		inline T& get_data() const { return *this->data; }

		/** Returns a raw pointer to the first element in the data buffer
		* @return A raw pointer to the first element in the data buffer
		*/
		virtual void* get_raw() const override { return this->data; }

		/** Default move assignment operator
		* @param other The HistogramVoxel to move
		* @return The moved HistogramVoxel
		*/
		HistogramVoxel<T>& operator=(const HistogramVoxel<T>& other) = default;

		HistogramVoxel<T>& operator=(const T& val) {
			for (size_t i = 0; i < this->histogram_definition.bins; ++i) {
				this->data[i] = val;
			}
			return *this;
		}

		HistogramVoxel<T>& operator+=(const T& rhs) {
			for (size_t i = 0; i < this->histogram_definition.bins; ++i) {
				this->data[i] += rhs;
			}
			return *this;
		}

		HistogramVoxel<T>& operator-=(const T& rhs) {
			for (size_t i = 0; i < this->histogram_definition.bins; ++i) {
				this->data[i] -= rhs;
			}
			return *this;
		}

		HistogramVoxel<T>& operator*=(const T& rhs) {
			for (size_t i = 0; i < this->histogram_definition.bins; ++i) {
				this->data[i] *= rhs;
			}
			return *this;
		}

		HistogramVoxel<T>& operator/=(const T& rhs) {
			for (size_t i = 0; i < this->histogram_definition.bins; ++i) {
				this->data[i] /= rhs;
			}
			return *this;
		}

		HistogramVoxel<T>& operator/=(const HistogramVoxel<T>& rhs) {
			for (size_t i = 0; i < this->histogram_definition.bins; ++i) {
				this->data[i] /= rhs.data[i];
			}
			return *this;
		}

		HistogramVoxel<T>& operator-=(const HistogramVoxel<T>& rhs) {
			for (size_t i = 0; i < this->histogram_definition.bins; ++i) {
				this->data[i] -= rhs.data[i];
			}
			return *this;
		}

		HistogramVoxel<T>& operator+=(const HistogramVoxel<T>& rhs) {
			for (size_t i = 0; i < this->histogram_definition.bins; ++i) {
				this->data[i] += rhs.data[i];
			}
			return *this;
		}

		HistogramVoxel<T>& operator*=(const HistogramVoxel<T>& rhs) {
			for (size_t i = 0; i < this->histogram_definition.bins; ++i) {
				this->data[i] *= rhs.data[i];
			}
			return *this;
		}

		friend HistogramVoxel<T> operator*(const HistogramVoxel<T>& lhs, T scalar) {
			HistogramVoxel<T> result(lhs);
			for (size_t i = 0; i < result.histogram_definition.bins; ++i) {
				result.data[i] *= scalar;
			}
			return result;
		}

		friend HistogramVoxel<T> operator+(const HistogramVoxel<T>& lhs, T scalar) {
			HistogramVoxel<T> result(lhs);
			for (size_t i = 0; i < result.histogram_definition.bins; ++i) {
				result.data[i] += scalar;
			}
			return result;
		}

		friend HistogramVoxel<T> operator-(const HistogramVoxel<T>& lhs, T scalar) {
			HistogramVoxel<T> result(lhs);
			for (size_t i = 0; i < result.histogram_definition.bins; ++i) {
				result.data[i] -= scalar;
			}
			return result;
		}

		friend HistogramVoxel<T> operator/(const HistogramVoxel<T>& lhs, T scalar) {
			HistogramVoxel<T> result(lhs);
			for (size_t i = 0; i < result.histogram_definition.bins; ++i) {
				result.data[i] /= scalar;
			}
			return result;
		}

		friend HistogramVoxel<T> operator*(const HistogramVoxel<T>& lhs, const HistogramVoxel<T>& rhs) {
			HistogramVoxel<T> result(lhs);
			for (size_t i = 0; i < lhs.histogram_definition.bins; ++i) {
				result.data[i] *= rhs.data[i];
			}
			return result;
		}

		friend HistogramVoxel<T> operator+(const HistogramVoxel<T>& lhs, const HistogramVoxel<T>& rhs) {
			HistogramVoxel<T> result(lhs);
			for (size_t i = 0; i < lhs.histogram_definition.bins; ++i) {
				result.data[i] += rhs.data[i];
			}
			return result;
		}

		friend HistogramVoxel<T> operator-(const HistogramVoxel<T>& lhs, const HistogramVoxel<T>& rhs) {
			HistogramVoxel<T> result(lhs);
			for (size_t i = 0; i < lhs.histogram_definition.bins; ++i) {
				result.data[i] -= rhs.data[i];
			}
			return result;
		}

		friend HistogramVoxel<T> operator/(const HistogramVoxel<T>& lhs, const HistogramVoxel<T>& rhs) {
			HistogramVoxel<T> result(lhs);
			for (size_t i = 0; i < lhs.histogram_definition.bins; ++i) {
				result.data[i] /= rhs.data[i];
			}
			return result;
		}

		/** Returns the header of a Voxel to reconstruct it after serialization.
		* Header struct should be extended by the Voxel implementation.
		* @return The header of the Voxel
		*/
		virtual IVoxel::VoxelBaseHeader get_header() const override {
			return IVoxel::VoxelBaseHeader(sizeof(HistogramDefinition), (void*)&this->histogram_definition);
		}

		/** Initializes the Voxel from a header block. For deserialization only!
		* @param header The header to initialize the Voxel from
		*/
		virtual void init_from_header(const void* header) override {
			this->histogram_definition = *(HistogramVoxel<T>::HistogramDefinition*)header;

		}

		/** Adds a positive value to the histogram and scores it into the correct bin.
		* The correct bin is determined by dividing the value by the bin width and rounding to the nearest integer.
		* If the value is greater than the maximum value, it is scored in the last bin.
		* If the value is less than 0, it is scored in the first bin.
		* @param value The value to add to the histogram
		*/
		void add_value(T& value) {
			size_t bin = static_cast<size_t>((value >= 0.f) ? ((value + this->histogram_definition.histogram_bin_width / T(2)) / this->histogram_definition.histogram_bin_width) : 0);
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
			if (sum == T(0)) {
				return;
			}
			for (size_t i = 0; i < this->histogram_definition.bins; i++) {
				this->data[i] /= sum;
			}
		}

		/** Clears the histogram by setting all bins to 0 */
		void clear() {
			std::fill(this->data, this->data + this->histogram_definition.bins, T(0));
		}
	};

	/** Owning version of the HistogramVoxel class. This class owns the data it points to, and will automatically deallocate the data when the OwningHistogramVoxel is destroyed.
	*/
	template<typename T = float>
	class OwningHistogramVoxel : public HistogramVoxel<T> {
	public:
		using typename HistogramVoxel<T>::HistogramDefinition;

		OwningHistogramVoxel(size_t bins = 0, const T& histogram_bin_width = T(0)) : HistogramVoxel<T>(bins, histogram_bin_width, (bins > 0) ? new T[bins] : nullptr) {}

		OwningHistogramVoxel(size_t bins, const T& histogram_bin_width, T* buffer) : HistogramVoxel<T>(bins, histogram_bin_width, (bins > 0) ? new T[bins] : nullptr) {
			if (bins > 0)
				memcpy(this->data, buffer, bins * sizeof(T));
		}

		OwningHistogramVoxel(const OwningHistogramVoxel<T>& buffer) : HistogramVoxel<T>(buffer) {
			if (this->histogram_definition.bins > 0) {
				this->data = new T[this->histogram_definition.bins];
				memcpy(this->data, buffer.data, this->histogram_definition.bins * sizeof(T));
			}
		}

		OwningHistogramVoxel(OwningHistogramVoxel<T>&& buffer) noexcept : HistogramVoxel<T>(buffer) {
			if (this->histogram_definition.bins > 0) {
				this->data = new T[this->histogram_definition.bins];
				memcpy(this->data, buffer.data, this->histogram_definition.bins * sizeof(T));
			}
		}

		~OwningHistogramVoxel() {
			if (this->data != nullptr)
				delete[] this->data;
		}

		virtual void init_from_header(const void* header) override {
			this->histogram_definition = *(HistogramDefinition*)header;
			if (this->data != nullptr) {
				delete[] this->data;
			}
			this->data = new T[this->histogram_definition.bins];
		}

		virtual void set_data(void* data) override {
			memcpy(this->data, data, this->histogram_definition.bins * sizeof(T));
		}
	};

	template<typename T = float>
	class AngularResolvedVoxel : public ScalarVoxel<T> {
		using typename IVoxel::VoxelBaseHeader;
	public:
#pragma pack(push, 4)
		struct AngularDefinition : public IVoxel::VoxelBaseHeader {
			glm::uvec2 segments;

			/**
			* @param segments (phi, theta)
			*/
			AngularDefinition(const glm::uvec2& segments) : segments(segments) {}
		};
#pragma pack(pop)

	protected:
		AngularDefinition angular_definition;

		/** Compute flat index from phi/theta grid indices */
		inline size_t calc_segment_idx(size_t phi_idx, size_t theta_idx) const {
			return theta_idx * this->angular_definition.segments.x + phi_idx;
		}

		/** Compute flat index from phi/theta in standard spherical coordinates (linear mapping)
		* @param phi Azimuthal angle in radians [0, 2*pi]
		* @param theta Polar angle in radians [0, pi]
		*/
		inline size_t calc_segment_idx_by_coord(float phi, float theta) const {
			// Wrap phi to [0, 2*pi]
			if (phi < 0.f) phi += 2.f * std::numbers::pi_v<float>;
			if (phi >= 2.f * std::numbers::pi_v<float>) phi -= 2.f * std::numbers::pi_v<float>;
			// Clamp theta to [0, pi]
			if (theta < 0.f) theta = 0.f;
			if (theta > std::numbers::pi_v<float>) theta = std::numbers::pi_v<float>;

			size_t phi_idx = static_cast<size_t>(phi / (2.f * std::numbers::pi_v<float>) * this->angular_definition.segments.x);
			size_t theta_idx = static_cast<size_t>(theta / std::numbers::pi_v<float> * this->angular_definition.segments.y);

			if (phi_idx >= this->angular_definition.segments.x) phi_idx = this->angular_definition.segments.x - 1;
			if (theta_idx >= this->angular_definition.segments.y) theta_idx = this->angular_definition.segments.y - 1;

			return calc_segment_idx(phi_idx, theta_idx);
		}

	public:
		/** Create a new AngularResolvedVoxel with an empty data buffer */
		AngularResolvedVoxel() noexcept : ScalarVoxel<T>(nullptr), angular_definition(AngularDefinition(glm::uvec2(0))) {}

		/** Create a new AngularResolvedVoxel with the given data buffer */
		AngularResolvedVoxel(const glm::uvec2& segments, T* buffer) : ScalarVoxel<T>(buffer), angular_definition(segments) {}

		/** Move constructor */
		AngularResolvedVoxel(AngularResolvedVoxel&& buffer) noexcept : ScalarVoxel<T>(buffer.data), angular_definition(buffer.angular_definition) {
			buffer.data = nullptr;
		}

		/** Copy constructor */
		AngularResolvedVoxel(const AngularResolvedVoxel& buffer) noexcept : ScalarVoxel<T>(buffer.data), angular_definition(buffer.angular_definition) {}

		/** Returns the total number of segments (phi * theta) */
		inline size_t get_total_segments() const { return this->angular_definition.segments.x * this->angular_definition.segments.y; }

		/** Returns the number of phi segments */
		inline size_t get_phi_segments() const { return this->angular_definition.segments.x; }

		/** Returns the number of theta segments */
		inline size_t get_theta_segments() const { return this->angular_definition.segments.y; }

		inline const glm::uvec2& get_segments() const{ return this->angular_definition.segments; }

		/** Returns a span containing all segment data */
		inline std::span<T> get_segments_data() const {
			return std::span<T>(this->data, this->get_total_segments());
		}

		/** Access a segment value by spherical coordinates
		* @param phi The phi coordinate in radians
		* @param theta The theta coordinate in radians
		* @return Reference to the segment value
		*/
		inline T& get_value_by_coord(float phi, float theta) const {
			return this->data[this->calc_segment_idx_by_coord(phi, theta)];
		}

		/** Access a segment value by spherical coordinates
		* @param coords The (phi, theta) coordinate in radians
		* @return Reference to the segment value
		*/
		inline T& get_value_by_coord(const glm::vec2& coords) const {
			return this->get_value_by_coord(coords.x, coords.y);
		}

		/** Access a segment value by grid indices
		* @param phi_idx The phi index [0, phi_segments - 1]
		* @param theta_idx The theta index [0, theta_segments - 1]
		* @return Reference to the segment value
		*/
		inline T& get_value(size_t phi_idx, size_t theta_idx) const {
			return this->data[this->calc_segment_idx(phi_idx, theta_idx)];
		}

		/** Access a segment value by grid indices
		* @param phi_idx The phi index [0, phi_segments - 1]
		* @param theta_idx The theta index [0, theta_segments - 1]
		* @return Reference to the segment value
		*/
		inline T& get_value(const glm::uvec2& idx_coord) const {
			return this->get_value(idx_coord.x, idx_coord.y);
		}

		/** Returns the type of the voxel as a string */
		virtual std::string get_type() const override {
			return "spherical";
		}

		/** Returns the size of the voxel object in bytes, excluding the data */
		virtual size_t get_voxel_bytes() const override { return sizeof(AngularResolvedVoxel<T>); }

		/** Returns the size of the data in bytes */
		virtual size_t get_bytes() const override { return sizeof(T) * this->get_total_segments(); }

		/** Returns the reference to the first element in the data buffer */
		inline T& get_data() const { return *this->data; }

		/** Returns a raw pointer to the data buffer */
		virtual void* get_raw() const override { return this->data; }

		/** Default assignment operator */
		AngularResolvedVoxel<T>& operator=(const AngularResolvedVoxel<T>& other) = default;

		AngularResolvedVoxel<T>& operator=(const T& val) {
			for (size_t i = 0; i < this->get_total_segments(); ++i) {
				this->data[i] = val;
			}
			return *this;
		}

		AngularResolvedVoxel<T>& operator+=(const T& rhs) {
			for (size_t i = 0; i < this->get_total_segments(); ++i) {
				this->data[i] += rhs;
			}
			return *this;
		}

		AngularResolvedVoxel<T>& operator-=(const T& rhs) {
			for (size_t i = 0; i < this->get_total_segments(); ++i) {
				this->data[i] -= rhs;
			}
			return *this;
		}

		AngularResolvedVoxel<T>& operator*=(const T& rhs) {
			for (size_t i = 0; i < this->get_total_segments(); ++i) {
				this->data[i] *= rhs;
			}
			return *this;
		}

		AngularResolvedVoxel<T>& operator/=(const T& rhs) {
			for (size_t i = 0; i < this->get_total_segments(); ++i) {
				this->data[i] /= rhs;
			}
			return *this;
		}

		AngularResolvedVoxel<T>& operator/=(const AngularResolvedVoxel<T>& rhs) {
			for (size_t i = 0; i < this->get_total_segments(); ++i) {
				this->data[i] /= rhs.data[i];
			}
			return *this;
		}

		AngularResolvedVoxel<T>& operator-=(const AngularResolvedVoxel<T>& rhs) {
			for (size_t i = 0; i < this->get_total_segments(); ++i) {
				this->data[i] -= rhs.data[i];
			}
			return *this;
		}

		AngularResolvedVoxel<T>& operator+=(const AngularResolvedVoxel<T>& rhs) {
			for (size_t i = 0; i < this->get_total_segments(); ++i) {
				this->data[i] += rhs.data[i];
			}
			return *this;
		}

		AngularResolvedVoxel<T>& operator*=(const AngularResolvedVoxel<T>& rhs) {
			for (size_t i = 0; i < this->get_total_segments(); ++i) {
				this->data[i] *= rhs.data[i];
			}
			return *this;
		}

		friend AngularResolvedVoxel<T> operator*(const AngularResolvedVoxel<T>& lhs, T scalar) {
			AngularResolvedVoxel<T> result(lhs);
			for (size_t i = 0; i < result.get_total_segments(); ++i) {
				result.data[i] *= scalar;
			}
			return result;
		}

		friend AngularResolvedVoxel<T> operator+(const AngularResolvedVoxel<T>& lhs, T scalar) {
			AngularResolvedVoxel<T> result(lhs);
			for (size_t i = 0; i < result.get_total_segments(); ++i) {
				result.data[i] += scalar;
			}
			return result;
		}

		friend AngularResolvedVoxel<T> operator-(const AngularResolvedVoxel<T>& lhs, T scalar) {
			AngularResolvedVoxel<T> result(lhs);
			for (size_t i = 0; i < result.get_total_segments(); ++i) {
				result.data[i] -= scalar;
			}
			return result;
		}

		friend AngularResolvedVoxel<T> operator/(const AngularResolvedVoxel<T>& lhs, T scalar) {
			AngularResolvedVoxel<T> result(lhs);
			for (size_t i = 0; i < result.get_total_segments(); ++i) {
				result.data[i] /= scalar;
			}
			return result;
		}

		friend AngularResolvedVoxel<T> operator*(const AngularResolvedVoxel<T>& lhs, const AngularResolvedVoxel<T>& rhs) {
			AngularResolvedVoxel<T> result(lhs);
			for (size_t i = 0; i < lhs.get_total_segments(); ++i) {
				result.data[i] *= rhs.data[i];
			}
			return result;
		}

		friend AngularResolvedVoxel<T> operator+(const AngularResolvedVoxel<T>& lhs, const AngularResolvedVoxel<T>& rhs) {
			AngularResolvedVoxel<T> result(lhs);
			for (size_t i = 0; i < lhs.get_total_segments(); ++i) {
				result.data[i] += rhs.data[i];
			}
			return result;
		}

		friend AngularResolvedVoxel<T> operator-(const AngularResolvedVoxel<T>& lhs, const AngularResolvedVoxel<T>& rhs) {
			AngularResolvedVoxel<T> result(lhs);
			for (size_t i = 0; i < lhs.get_total_segments(); ++i) {
				result.data[i] -= rhs.data[i];
			}
			return result;
		}

		friend AngularResolvedVoxel<T> operator/(const AngularResolvedVoxel<T>& lhs, const AngularResolvedVoxel<T>& rhs) {
			AngularResolvedVoxel<T> result(lhs);
			for (size_t i = 0; i < lhs.get_total_segments(); ++i) {
				result.data[i] /= rhs.data[i];
			}
			return result;
		}

		/** Returns the header for serialization */
		virtual IVoxel::VoxelBaseHeader get_header() const override {
			return IVoxel::VoxelBaseHeader(sizeof(AngularResolvedVoxel<T>::AngularDefinition), (void*)&this->angular_definition);
		}

		/** Initializes the Voxel from a header block (deserialization) */
		virtual void init_from_header(const void* header) override {
			this->angular_definition = *(AngularResolvedVoxel<T>::AngularDefinition*)header;
		}

		/** Adds a value at the given spherical direction
		* @param phi The phi coordinate in radians
		* @param theta The theta coordinate in radians
		* @param value The value to add
		*/
		void add_value(float phi, float theta, float value = 1.f) {
			this->data[this->calc_segment_idx_by_coord(phi, theta)] += value;
		}

		/** Adds a value at the given spherical direction
		* @param phi The phi coordinate in radians
		* @param theta The theta coordinate in radians
		* @param value The value to add
		*/
		void add_value(const glm::vec2& coord, float value = 1.f) {
			this->add_value(coord.x, coord.y, value);
		}

		/** Clears all segments to 0 */
		void clear() {
			std::fill(this->data, this->data + this->get_total_segments(), 0.0f);
		}
	};

	/** Owning version of the AngularResolvedVoxel class. Owns the data buffer and frees it on destruction. */
	template<typename T = float>
	class OwningAngularResolvedVoxel : public AngularResolvedVoxel<T> {
	public:
		using typename AngularResolvedVoxel<T>::AngularDefinition;

		OwningAngularResolvedVoxel(const glm::uvec2& segments = glm::uvec2(0)) : AngularResolvedVoxel<T>(segments, (segments.x * segments.y > 0) ? new T[segments.x * segments.y]() : nullptr) {}

		OwningAngularResolvedVoxel(const glm::uvec2& segments, T* buffer) : AngularResolvedVoxel<T>(segments, (segments.x * segments.y > 0) ? new T[segments.x * segments.y] : nullptr) {
			if (segments.x * segments.y > 0)
				memcpy(this->data, buffer, segments.x * segments.y * sizeof(T));
		}

		OwningAngularResolvedVoxel(const OwningAngularResolvedVoxel<T>& buffer) : AngularResolvedVoxel<T>(buffer) {
			size_t total = this->get_total_segments();
			if (total > 0) {
				this->data = new T[total];
				memcpy(this->data, buffer.data, total * sizeof(T));
			}
		}

		OwningAngularResolvedVoxel(OwningAngularResolvedVoxel<T>&& buffer) noexcept : AngularResolvedVoxel<T>(buffer) {
			size_t total = this->get_total_segments();
			if (total > 0) {
				this->data = new T[total];
				memcpy(this->data, buffer.data, total * sizeof(T));
			}
		}

		~OwningAngularResolvedVoxel() {
			if (this->data != nullptr)
				delete[] this->data;
		}

		virtual void init_from_header(const void* header) override {
			this->angular_definition = *(AngularDefinition*)header;
			if (this->data != nullptr)
				delete[] this->data;
			this->data = new T[this->angular_definition.segments.x * this->angular_definition.segments.y];
		}

		virtual void set_data(void* data) override {
			memcpy(this->data, data, this->get_total_segments() * sizeof(T));
		}
	};
};