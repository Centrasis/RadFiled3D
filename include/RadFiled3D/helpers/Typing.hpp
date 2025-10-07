#pragma once
#include <typeinfo>
#include <string>
#ifndef _WIN32 
	#include <cxxabi.h>
#endif

namespace RadFiled3D {
	namespace Typing {
		enum FieldShape : uint8_t {
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
			UInt64,
			UInt32
		};

		class Helper {
		public:
			template<typename T>
			static inline std::string get_plain_type_name() {
				std::string name = typeid(T).name();
		#ifdef _WIN32 
				return name;
		#else
				int status;
				char* demangled = abi::__cxa_demangle(name.c_str(), 0, 0, &status);
				std::string result = demangled;
				free(demangled);
				return result;
		#endif
			}

			/** Converts a data type string to the datatype enumeration
			* @param dtype The data type string
			* @return The data type enumeration
			*/
			static Typing::DType get_dtype(const std::string& dtype);

			/** Fetches the number of bytes of a data type */
			static size_t get_bytes_of_dtype(Typing::DType dtype);
		};
	}
}