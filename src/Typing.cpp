#include <RadFiled3D/helpers/Typing.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <stdexcept>

using namespace RadFiled3D;


Typing::DType Typing::Helper::get_dtype(const std::string& dtype)
{
	if (dtype == Typing::Helper::get_plain_type_name<float>()) {
		return Typing::DType::Float;
	}
	if (dtype == Typing::Helper::get_plain_type_name<double>()) {
		return Typing::DType::Double;
	}
	if (dtype == Typing::Helper::get_plain_type_name<int>()) {
		return Typing::DType::Int;
	}
	if (dtype == Typing::Helper::get_plain_type_name<char>()) {
		return Typing::DType::Char;
	}
	if (dtype == Typing::Helper::get_plain_type_name<uint64_t>()) {
		return Typing::DType::UInt64;
	}
	if (dtype == Typing::Helper::get_plain_type_name<unsigned long long>()) {
		return Typing::DType::UInt64;
	}
	if (dtype == Typing::Helper::get_plain_type_name<uint32_t>()) {
		return Typing::DType::UInt32;
	}
	if (dtype == Typing::Helper::get_plain_type_name<unsigned long>()) {
		return Typing::DType::UInt32;
	}
	if (dtype == Typing::Helper::get_plain_type_name<glm::vec3>()) {
		return Typing::DType::Vec3;
	}
	if (dtype == Typing::Helper::get_plain_type_name<glm::vec2>()) {
		return Typing::DType::Vec2;
	}
	if (dtype == Typing::Helper::get_plain_type_name<glm::vec4>()) {
		return Typing::DType::Vec4;
	}
	if (dtype == std::string("histogram")) {
		return Typing::DType::Hist;
	}

	const std::string vec_prefix = "glm::vec<";

	if (dtype.compare(0, vec_prefix.size(), vec_prefix) == 0) {
		// check for vector type, if the file was created by a different compiler
		const std::string vec3_prefix = "glm::vec<3, float";
		const std::string vec2_prefix = "glm::vec<2, float";
		const std::string vec4_prefix = "glm::vec<4, float";
		if (dtype.compare(0, vec3_prefix.size(), vec3_prefix) == 0) {
			return Typing::DType::Vec3;
		}
		if (dtype.compare(0, vec2_prefix.size(), vec2_prefix) == 0) {
			return Typing::DType::Vec2;
		}
		if (dtype.compare(0, vec4_prefix.size(), vec4_prefix) == 0) {
			return Typing::DType::Vec4;
		}
	}

	throw std::runtime_error("Unknown data type: " + dtype);
}

size_t RadFiled3D::Typing::Helper::get_bytes_of_dtype(Typing::DType dtype)
{
	switch (dtype) {
	case Typing::DType::Float:
		return sizeof(float);
	case Typing::DType::Double:
		return sizeof(double);
	case Typing::DType::Int:
		return sizeof(int);
	case Typing::DType::Char:
		return sizeof(char);
	case Typing::DType::UInt64:
		return sizeof(uint64_t);
	case Typing::DType::UInt32:
		return sizeof(uint32_t);
	case Typing::DType::Vec3:
		return sizeof(glm::vec3);
	case Typing::DType::Vec2:
		return sizeof(glm::vec2);
	case Typing::DType::Vec4:
		return sizeof(glm::vec4);
	case Typing::DType::Hist:
		return sizeof(float);
	default:
		throw std::runtime_error("Unknown data type");
	}
}