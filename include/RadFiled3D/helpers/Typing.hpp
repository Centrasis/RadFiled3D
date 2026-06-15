#pragma once
#include <string>
#include <cstdint>
#include <stdexcept>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace RadFiled3D {
	namespace Typing {
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
			Byte
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
		// prefix fallback). _Float16 additionally has no typeinfo, so it could not use typeid at all.
		template<> inline std::string Helper::get_plain_type_name<float>()         { return "float"; }
		template<> inline std::string Helper::get_plain_type_name<double>()        { return "double"; }
		template<> inline std::string Helper::get_plain_type_name<int>()           { return "int"; }
		template<> inline std::string Helper::get_plain_type_name<char>()          { return "char"; }
		template<> inline std::string Helper::get_plain_type_name<unsigned char>() { return "unsigned char"; }
		template<> inline std::string Helper::get_plain_type_name<unsigned int>()  { return "unsigned int"; }
		template<> inline std::string Helper::get_plain_type_name<unsigned long>() { return "unsigned long"; }
		template<> inline std::string Helper::get_plain_type_name<_Float16>()      { return "float16"; }
		template<> inline std::string Helper::get_plain_type_name<glm::vec2>()     { return "glm::vec2"; }
		template<> inline std::string Helper::get_plain_type_name<glm::vec3>()     { return "glm::vec3"; }
		template<> inline std::string Helper::get_plain_type_name<glm::vec4>()     { return "glm::vec4"; }
	}
}