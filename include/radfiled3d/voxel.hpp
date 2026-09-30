#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/geometric.hpp>
#include "radfiled3d/helpers/typing.hpp"
#include <cstring>
#include <cmath>
#include <span>
#include <numbers>
#include <algorithm>


namespace radfiled3d {
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

		// Voxels have no virtual destructor on purpose: they must stay trivially destructible so a
		// VoxelBuffer can hold them in char* arrays and free them with delete[] (char*). A heap-allocated
		// voxel handed out as a base pointer (e.g. by FieldAccessor) is therefore freed through this
		// virtual override, which deletes at the concrete type, instead of `delete base_ptr`.
		virtual void selfDestruct() { delete this; }

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
			return typing::Helper::get_plain_type_name<T>();
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

		virtual void selfDestruct() override { delete this; }

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

	// (voxel_header is unused; its old initializer VoxelBaseHeader(std::string) had no matching ctor
	// and only compiled because it was never instantiated. Default-construct so the explicit
	// instantiations below are valid. The type name is provided by get_type()/get_plain_type_name.)
	template<typename T>
	IVoxel::VoxelBaseHeader ScalarVoxel<T>::voxel_header = IVoxel::VoxelBaseHeader();


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

		virtual void selfDestruct() override { delete this; }
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

		virtual void selfDestruct() override { delete this; }

		/** Compares the bin definition and all bins of two HistogramVoxels */
		bool operator==(const HistogramVoxel<T>& other) const {
			return this->histogram_definition.bins == other.histogram_definition.bins
				&& this->histogram_definition.histogram_bin_width == other.histogram_definition.histogram_bin_width
				&& std::equal(this->data, this->data + this->histogram_definition.bins, other.data);
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

		// Copy-assignment: OwningHistogramVoxel owns its `data` buffer (new[]/delete[]),
		// but the inherited ScalarVoxel/HistogramVoxel operator= is `= default` and only
		// shallow-copies the `data` pointer. Without this deep-copy override, assigning
		// (e.g. std::fill over a freshly-allocated voxel array in VoxelLayer::Construct*)
		// makes multiple voxels share one buffer, which is then freed multiple times →
		// "double free or corruption" when the field is destroyed. (rule-of-five fix)
		OwningHistogramVoxel<T>& operator=(const OwningHistogramVoxel<T>& buffer) {
			if (this == &buffer)
				return *this;
			T* old = this->data;
			this->histogram_definition = buffer.histogram_definition;
			if (this->histogram_definition.bins > 0 && buffer.data != nullptr) {
				this->data = new T[this->histogram_definition.bins];
				memcpy(this->data, buffer.data, this->histogram_definition.bins * sizeof(T));
			} else {
				this->data = nullptr;
			}
			if (old != nullptr)
				delete[] old;
			return *this;
		}

		// Move-assignment: steal the buffer and null the source so its destructor does
		// not free it (the inherited default would shallow-copy → double free).
		OwningHistogramVoxel<T>& operator=(OwningHistogramVoxel<T>&& buffer) noexcept {
			if (this == &buffer)
				return *this;
			T* old = this->data;
			this->histogram_definition = buffer.histogram_definition;
			this->data = buffer.data;
			buffer.data = nullptr;
			if (old != nullptr)
				delete[] old;
			return *this;
		}

		~OwningHistogramVoxel() {
			if (this->data != nullptr)
				delete[] this->data;
		}

		virtual void selfDestruct() override { delete this; }

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

		virtual void selfDestruct() override { delete this; }

		/** Compares the segmentation and all segments of two AngularResolvedVoxels */
		bool operator==(const AngularResolvedVoxel<T>& other) const {
			return this->angular_definition.segments == other.angular_definition.segments
				&& std::equal(this->data, this->data + this->get_total_segments(), other.data);
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

		virtual void selfDestruct() override { delete this; }

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

	/** A VMFMixtureVoxel stores the directional distribution of the radiation within a voxel as a mixture of
	* von Mises-Fisher (spherical Gaussian) lobes. Each lobe is stored as 5 contiguous values:
	* [weight, mean_x, mean_y, mean_z, kappa]
	* weight: share of the voxel's radiation carried by the lobe (weights sum to 1, or are all 0 for an empty voxel)
	* mean: unit vector of the direction of travel of the radiation
	* kappa: concentration (>= 0, 0 means isotropic)
	* Element-wise arithmetic is not defined for a mixture, use VMFMixtureVoxel::merge to combine two mixtures.
	*/
	template<typename T = float>
	class VMFMixtureVoxel : public ScalarVoxel<T> {
		using typename IVoxel::VoxelBaseHeader;
	public:
		static constexpr size_t values_per_lobe = 5;

#pragma pack(push, 4)
		struct VMFDefinition : public IVoxel::VoxelBaseHeader {
			uint32_t lobes;

			VMFDefinition(uint32_t lobes = 0) : lobes(lobes) {}
		};
#pragma pack(pop)

	protected:
		VMFDefinition vmf_definition;

		struct MergeLobe {
			double weight;
			glm::dvec3 mean;
			double kappa;
		};

		/** Normalized vMF density per steradian at cos(angle between mean and direction) */
		static double lobe_density(double cos_angle, double kappa) {
			constexpr double inv_4pi = 1.0 / (4.0 * std::numbers::pi);
			if (kappa < 1e-4)
				return inv_4pi * (1.0 + kappa * cos_angle);
			return kappa / (2.0 * std::numbers::pi * -std::expm1(-2.0 * kappa)) * std::exp(kappa * (cos_angle - 1.0));
		}

		/** Mean resultant length A(kappa) = coth(kappa) - 1/kappa */
		static double mean_resultant_length(double kappa) {
			if (kappa < 1e-3)
				return kappa / 3.0 - kappa * kappa * kappa / 45.0;
			return 1.0 / std::tanh(kappa) - 1.0 / kappa;
		}

		/** Inverts A(kappa) = R. The closed-form approximation R(3 - R^2)/(1 - R^2) overestimates kappa by a few
		* percent in the mid range, which would inflate kappa on every repeated join, so it is refined by Newton steps.
		*/
		static double kappa_from_mean_resultant_length(double R) {
			R = std::clamp(R, 1e-6, 0.999999);
			double kappa = R * (3.0 - R * R) / (1.0 - R * R);
			for (int i = 0; i < 4; i++) {
				const double A = mean_resultant_length(kappa);
				const double dA = 1.0 - A * A - 2.0 * A / kappa;
				if (!(dA > 0.0))
					break;
				const double next = kappa - (A - R) / dA;
				kappa = (next > 0.0) ? next : kappa * 0.5;
			}
			return kappa;
		}

	public:
		/** Create a new VMFMixtureVoxel with 0 lobes and an empty data buffer */
		VMFMixtureVoxel() noexcept : ScalarVoxel<T>(nullptr), vmf_definition(VMFDefinition(0)) {}

		/** Create a new VMFMixtureVoxel with the given data buffer
		* @param lobes The number of lobes of the mixture
		* @param buffer The data buffer holding 5 * lobes values
		*/
		VMFMixtureVoxel(uint32_t lobes, T* buffer) : ScalarVoxel<T>(buffer), vmf_definition(lobes) {}

		/** Move constructor */
		VMFMixtureVoxel(VMFMixtureVoxel&& buffer) noexcept : ScalarVoxel<T>(buffer.data), vmf_definition(buffer.vmf_definition) {
			buffer.data = nullptr;
		}

		/** Copy constructor */
		VMFMixtureVoxel(const VMFMixtureVoxel& buffer) noexcept : ScalarVoxel<T>(buffer.data), vmf_definition(buffer.vmf_definition) {}

		/** Returns the number of lobes of the mixture */
		inline uint32_t get_lobes() const { return this->vmf_definition.lobes; }

		/** Returns the weight of lobe k */
		inline T get_weight(size_t k) const { return this->data[k * values_per_lobe]; }

		/** Returns the mean direction of lobe k */
		inline glm::vec3 get_mean(size_t k) const {
			const T* lobe = this->data + k * values_per_lobe;
			return glm::vec3(lobe[1], lobe[2], lobe[3]);
		}

		/** Returns the concentration kappa of lobe k */
		inline T get_kappa(size_t k) const { return this->data[k * values_per_lobe + 4]; }

		/** Sets all parameters of lobe k
		* @param k The lobe index [0, lobes - 1]
		* @param weight The share of the voxel's radiation carried by the lobe
		* @param mean The unit direction of travel
		* @param kappa The concentration (>= 0)
		*/
		void set_lobe(size_t k, T weight, const glm::vec3& mean, T kappa) {
			T* lobe = this->data + k * values_per_lobe;
			lobe[0] = weight;
			lobe[1] = static_cast<T>(mean.x);
			lobe[2] = static_cast<T>(mean.y);
			lobe[3] = static_cast<T>(mean.z);
			lobe[4] = kappa;
		}

		/** Returns a span over all 5 * lobes values [weight, mean_x, mean_y, mean_z, kappa] per lobe */
		inline std::span<T> get_lobes_data() const {
			return std::span<T>(this->data, values_per_lobe * this->vmf_definition.lobes);
		}

		/** Evaluates the mixture density per steradian in the given direction.
		* Each lobe integrates to 1 over the sphere, so the mixture integrates to the sum of the weights.
		* @param direction The direction to evaluate (normalized internally)
		* @return The density per steradian
		*/
		T density(const glm::vec3& direction) const {
			const glm::dvec3 d(direction);
			const double d_len = glm::length(d);
			if (!(d_len > 0.0))
				return T(0);
			const glm::dvec3 dir = d / d_len;
			double sum = 0.0;
			for (size_t k = 0; k < this->vmf_definition.lobes; k++) {
				const T* lobe = this->data + k * values_per_lobe;
				const double weight = static_cast<double>(lobe[0]);
				if (weight == 0.0)
					continue;
				glm::dvec3 mean(static_cast<double>(lobe[1]), static_cast<double>(lobe[2]), static_cast<double>(lobe[3]));
				const double mean_len = glm::length(mean);
				if (mean_len > 0.0)
					mean /= mean_len;
				sum += weight * lobe_density(glm::dot(mean, dir), std::max(0.0, static_cast<double>(lobe[4])));
			}
			return static_cast<T>(sum);
		}

		/** Builds the mixture a * weight_a + b * weight_b (both inputs normalized to their own weight sum, the result
		* renormalized to 1) and reduces it to out.get_lobes() lobes by repeatedly merging the two lobes with the smallest
		* angle between their means. Merging preserves the first moment: the mean resultant vectors A(kappa) * mean are
		* averaged by weight and the merged kappa is recovered from the resulting length.
		* An input whose weights are all 0 does not contribute; if both are empty, out is cleared.
		* out may alias a or b.
		*/
		static void merge(const VMFMixtureVoxel<T>& a, double weight_a, const VMFMixtureVoxel<T>& b, double weight_b, VMFMixtureVoxel<T>& out) {
			// a and b are reduced onto out's lobes, so out's lobe count is the target of the
			// merge. Zero lobes leave the result nowhere to go, which used to pass silently.
			if (out.get_lobes() == 0)
				throw std::invalid_argument("VMFMixtureVoxel::merge: the output voxel has 0 lobes, so the merged mixture has nowhere to go. Give it the number of lobes the result should be reduced to.");

			std::vector<MergeLobe> lobes;
			lobes.reserve(a.get_lobes() + b.get_lobes());

			const auto collect = [&lobes](const VMFMixtureVoxel<T>& src, double src_weight) {
				double sum = 0.0;
				for (size_t k = 0; k < src.get_lobes(); k++)
					sum += std::max(0.0, static_cast<double>(src.get_weight(k)));
				if (!(sum > 0.0) || !(src_weight > 0.0))
					return;
				for (size_t k = 0; k < src.get_lobes(); k++) {
					const double w = static_cast<double>(src.get_weight(k));
					if (!(w > 0.0))
						continue;
					glm::dvec3 mean(src.get_mean(k));
					const double mean_len = glm::length(mean);
					if (mean_len > 0.0)
						mean /= mean_len;
					lobes.push_back({ w / sum * src_weight, mean, std::max(0.0, static_cast<double>(src.get_kappa(k))) });
				}
			};
			collect(a, weight_a);
			collect(b, weight_b);

			double total = 0.0;
			for (const auto& lobe : lobes)
				total += lobe.weight;
			for (auto& lobe : lobes)
				lobe.weight /= total;

			const size_t target = out.get_lobes();
			while (lobes.size() > target && lobes.size() >= 2) {
				size_t best_i = 0, best_j = 1;
				double best_cos = -2.0;
				for (size_t i = 0; i < lobes.size(); i++) {
					for (size_t j = i + 1; j < lobes.size(); j++) {
						const double c = glm::dot(lobes[i].mean, lobes[j].mean);
						if (c > best_cos) {
							best_cos = c;
							best_i = i;
							best_j = j;
						}
					}
				}

				const MergeLobe& l1 = lobes[best_i];
				const MergeLobe& l2 = lobes[best_j];
				const double w = l1.weight + l2.weight;
				const glm::dvec3 r = (l1.weight * mean_resultant_length(l1.kappa) * l1.mean + l2.weight * mean_resultant_length(l2.kappa) * l2.mean) / w;
				const double R = glm::length(r);
				MergeLobe merged;
				merged.weight = w;
				merged.mean = (R > 1e-12) ? r / R : ((l1.weight >= l2.weight) ? l1.mean : l2.mean);
				merged.kappa = kappa_from_mean_resultant_length(R);
				lobes[best_i] = merged;
				lobes.erase(lobes.begin() + best_j);
			}

			std::vector<T> result(values_per_lobe * target, T(0));
			for (size_t k = 0; k < std::min(lobes.size(), target); k++) {
				T* lobe = result.data() + k * values_per_lobe;
				lobe[0] = static_cast<T>(lobes[k].weight);
				lobe[1] = static_cast<T>(lobes[k].mean.x);
				lobe[2] = static_cast<T>(lobes[k].mean.y);
				lobe[3] = static_cast<T>(lobes[k].mean.z);
				lobe[4] = static_cast<T>(lobes[k].kappa);
			}
			std::copy(result.begin(), result.end(), out.data);
		}

		/** Returns the type of the voxel as a string */
		virtual std::string get_type() const override {
			return "vmf_mixture";
		}

		/** Returns the size of the voxel object in bytes, excluding the data */
		virtual size_t get_voxel_bytes() const override { return sizeof(VMFMixtureVoxel<T>); }

		/** Returns the size of the data in bytes */
		virtual size_t get_bytes() const override { return sizeof(T) * values_per_lobe * this->vmf_definition.lobes; }

		/** Returns the reference to the first element in the data buffer */
		inline T& get_data() const { return *this->data; }

		/** Returns a raw pointer to the data buffer */
		virtual void* get_raw() const override { return this->data; }

		/** Default assignment operator */
		VMFMixtureVoxel<T>& operator=(const VMFMixtureVoxel<T>& other) = default;

		// Element-wise arithmetic has no meaning for a mixture. The inherited ScalarVoxel operators would silently act on the first weight only.
		VMFMixtureVoxel<T>& operator=(const T& val) = delete;
		VMFMixtureVoxel<T>& operator+=(const T& rhs) = delete;
		VMFMixtureVoxel<T>& operator-=(const T& rhs) = delete;
		VMFMixtureVoxel<T>& operator*=(const T& rhs) = delete;
		VMFMixtureVoxel<T>& operator/=(const T& rhs) = delete;
		VMFMixtureVoxel<T>& operator+=(const VMFMixtureVoxel<T>& rhs) = delete;
		VMFMixtureVoxel<T>& operator-=(const VMFMixtureVoxel<T>& rhs) = delete;
		VMFMixtureVoxel<T>& operator*=(const VMFMixtureVoxel<T>& rhs) = delete;
		VMFMixtureVoxel<T>& operator/=(const VMFMixtureVoxel<T>& rhs) = delete;

		/** Returns the header for serialization */
		virtual IVoxel::VoxelBaseHeader get_header() const override {
			return IVoxel::VoxelBaseHeader(sizeof(VMFMixtureVoxel<T>::VMFDefinition), (void*)&this->vmf_definition);
		}

		/** Initializes the Voxel from a header block (deserialization) */
		virtual void init_from_header(const void* header) override {
			this->vmf_definition = *(VMFMixtureVoxel<T>::VMFDefinition*)header;
		}

		virtual void selfDestruct() override { delete this; }

		/** Compares the lobe count and all lobe values of two VMFMixtureVoxels */
		bool operator==(const VMFMixtureVoxel<T>& other) const {
			return this->vmf_definition.lobes == other.vmf_definition.lobes
				&& std::equal(this->data, this->data + values_per_lobe * this->vmf_definition.lobes, other.data);
		}

		/** Sets all lobes to 0 */
		void clear() {
			std::fill(this->data, this->data + values_per_lobe * this->vmf_definition.lobes, T(0));
		}
	};

	/** Owning version of the VMFMixtureVoxel class. Owns the data buffer and frees it on destruction. */
	template<typename T = float>
	class OwningVMFMixtureVoxel : public VMFMixtureVoxel<T> {
	public:
		using typename VMFMixtureVoxel<T>::VMFDefinition;
		using VMFMixtureVoxel<T>::values_per_lobe;

		OwningVMFMixtureVoxel(uint32_t lobes = 0) : VMFMixtureVoxel<T>(lobes, (lobes > 0) ? new T[values_per_lobe * lobes]() : nullptr) {}

		OwningVMFMixtureVoxel(uint32_t lobes, T* buffer) : VMFMixtureVoxel<T>(lobes, (lobes > 0) ? new T[values_per_lobe * lobes] : nullptr) {
			if (lobes > 0)
				memcpy(this->data, buffer, values_per_lobe * lobes * sizeof(T));
		}

		OwningVMFMixtureVoxel(const OwningVMFMixtureVoxel<T>& buffer) : VMFMixtureVoxel<T>(buffer) {
			const size_t count = values_per_lobe * this->get_lobes();
			this->data = nullptr;
			if (count > 0 && buffer.data != nullptr) {
				this->data = new T[count];
				memcpy(this->data, buffer.data, count * sizeof(T));
			}
		}

		OwningVMFMixtureVoxel(OwningVMFMixtureVoxel<T>&& buffer) noexcept : VMFMixtureVoxel<T>(buffer) {
			buffer.data = nullptr;
		}

		// Deep copy / buffer stealing: the inherited defaulted assignment only copies the data pointer (see OwningHistogramVoxel).
		OwningVMFMixtureVoxel<T>& operator=(const OwningVMFMixtureVoxel<T>& buffer) {
			if (this == &buffer)
				return *this;
			T* old = this->data;
			this->vmf_definition = buffer.vmf_definition;
			const size_t count = values_per_lobe * this->get_lobes();
			if (count > 0 && buffer.data != nullptr) {
				this->data = new T[count];
				memcpy(this->data, buffer.data, count * sizeof(T));
			} else {
				this->data = nullptr;
			}
			if (old != nullptr)
				delete[] old;
			return *this;
		}

		OwningVMFMixtureVoxel<T>& operator=(OwningVMFMixtureVoxel<T>&& buffer) noexcept {
			if (this == &buffer)
				return *this;
			T* old = this->data;
			this->vmf_definition = buffer.vmf_definition;
			this->data = buffer.data;
			buffer.data = nullptr;
			if (old != nullptr)
				delete[] old;
			return *this;
		}

		~OwningVMFMixtureVoxel() {
			if (this->data != nullptr)
				delete[] this->data;
		}

		virtual void selfDestruct() override { delete this; }

		virtual void init_from_header(const void* header) override {
			this->vmf_definition = *(VMFDefinition*)header;
			if (this->data != nullptr)
				delete[] this->data;
			this->data = (this->get_lobes() > 0) ? new T[values_per_lobe * this->get_lobes()]() : nullptr;
		}

		virtual void set_data(void* data) override {
			memcpy(this->data, data, values_per_lobe * this->get_lobes() * sizeof(T));
		}
	};
};