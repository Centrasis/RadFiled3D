#pragma once
#include <string>
#include <cstdint>
#include <stdexcept>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

// Half precision (_Float16) requires a recent toolchain (GCC >= 12, modern Clang/MSVC-clang).
// Older compilers (e.g. the gcc-toolset-10 in some manylinux images) don't provide the type, so
// float16 support is compiled out there and any attempt to use it raises a clear runtime error.
#if defined(__FLT16_MAX__)
#define RADFILED3D_HAS_FLOAT16 1
#else
#define RADFILED3D_HAS_FLOAT16 0
#endif

namespace RadFiled3D {
	namespace Typing {
#if RADFILED3D_HAS_FLOAT16
		using float16 = _Float16;
#endif

		enum class FieldShape : uint8_t {
			Cone = 0,
			Rectangle = 1,
			Ellipsis = 2
		};

		enum class DType {
			Float,
			Double,
			Int,
			Char,
			Vec2,
			Vec3,
			Vec4,
			Hist,
			AngularResolved,
			UInt64,
			UInt32,
			Byte,
			Float16
		};

		class Helper {
		public:
			// Type name used as the on-disk dtype tag. The supported voxel types are pinned via the
			// explicit specializations below (compiler-independent, no RTTI). Any other type has no
			// stable name and is not serializable (incl. to the Python/numpy bindings), so the
			// primary template throws rather than relying on typeid().
			template<typename T>
			static inline std::string get_plain_type_name() {
				throw std::runtime_error("RadFiled3D: voxel type has no get_plain_type_name specialization and cannot be serialized. Add a Typing::Helper::get_plain_type_name<T> overload (and a DType) for it.");
			}

			/** Converts a data type string to the datatype enumeration
			* @param dtype The data type string
			* @return The data type enumeration
			*/
			static Typing::DType get_dtype(const std::string& dtype);

			/** Fetches the number of bytes of a data type */
			static size_t get_bytes_of_dtype(Typing::DType dtype);
		};

		// Static, compiler-independent type names. The primary template above uses typeid(), whose
		// std::type_info::name() is implementation-defined (differs MSVC vs Itanium, and demangled
		// glm names overflow the 32-char on-disk dtype field). These overloads pin the name per type
		// so it is identical on every OS/compiler. Scalar names match what existing files already
		// store ("float"/"unsigned char"/"unsigned int"/…); glm uses short names that fit the field
		// (older files store the long, truncated glm name and still load via get_dtype's "glm::vec<"
		// prefix fallback). float16 additionally has no typeinfo, so it could not use typeid at all.
		template<> inline std::string Helper::get_plain_type_name<float>()         { return "float"; }
		template<> inline std::string Helper::get_plain_type_name<double>()        { return "double"; }
		template<> inline std::string Helper::get_plain_type_name<int>()           { return "int"; }
		template<> inline std::string Helper::get_plain_type_name<char>()          { return "char"; }
		template<> inline std::string Helper::get_plain_type_name<unsigned char>() { return "unsigned char"; }
		template<> inline std::string Helper::get_plain_type_name<unsigned int>()  { return "unsigned int"; }
		// 64-bit unsigned is stored under the canonical fixed-width name. unsigned long is 64-bit on
		// LP64 (Linux/macOS) but 32-bit on LLP64 (Windows), so it only canonicalises to uint64_t when
		// it actually is 64 bits; otherwise it keeps its legacy spelling (read as a 32-bit unsigned).
		template<> inline std::string Helper::get_plain_type_name<unsigned long>() { return (sizeof(unsigned long) == 8) ? "uint64_t" : "unsigned long"; }
		template<> inline std::string Helper::get_plain_type_name<unsigned long long>() { return "uint64_t"; }
#if RADFILED3D_HAS_FLOAT16
		template<> inline std::string Helper::get_plain_type_name<float16>()       { return "float16"; }
#endif
		template<> inline std::string Helper::get_plain_type_name<glm::vec2>()     { return "glm::vec2"; }
		template<> inline std::string Helper::get_plain_type_name<glm::vec3>()     { return "glm::vec3"; }
		template<> inline std::string Helper::get_plain_type_name<glm::vec4>()     { return "glm::vec4"; }
	}
}