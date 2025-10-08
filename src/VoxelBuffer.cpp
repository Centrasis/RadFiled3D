#include "RadFiled3D/VoxelBuffer.hpp"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>


using namespace RadFiled3D;

template<typename T>
void add_layers_together(char* this_layer_data, char* other_layer_data, size_t element_count)
{
	T* this_data = (T*)this_layer_data;
	T* other_data = (T*)other_layer_data;

	for (size_t i = 0; i < element_count; i++)
	{
		this_data[i] += other_data[i];
	}
}

template<typename T>
void multiply_layers_together(char* this_layer_data, char* other_layer_data, size_t element_count)
{
	T* this_data = (T*)this_layer_data;
	T* other_data = (T*)other_layer_data;

	for (size_t i = 0; i < element_count; i++)
	{
		this_data[i] *= other_data[i];
	}
}

template<typename T>
void divide_layers_together(char* this_layer_data, char* other_layer_data, size_t element_count)
{
	T* this_data = (T*)this_layer_data;
	T* other_data = (T*)other_layer_data;

	for (size_t i = 0; i < element_count; i++)
	{
		this_data[i] /= other_data[i];
	}
}

template<typename T>
void subtract_layers_together(char* this_layer_data, char* other_layer_data, size_t element_count)
{
	T* this_data = (T*)this_layer_data;
	T* other_data = (T*)other_layer_data;

	for (size_t i = 0; i < element_count; i++)
	{
		this_data[i] -= other_data[i];
	}
}


VoxelBuffer::VoxelBuffer(size_t voxel_count)
	: voxel_count(voxel_count)
{
}

bool VoxelBuffer::operator==(VoxelBuffer const& other) const
{
	if (this->voxel_count != other.voxel_count)
		return false;

	auto layers = this->get_layers();
	
	if (layers.size() != other.get_layers().size())
		return false;

	for (auto& layer : layers)
	{
		auto other_layer_info = other.layers.find(layer);
		if (other_layer_info == other.layers.end())
			return false;

		auto layer_info = this->layers.find(layer);

		if (other_layer_info->second.bytes_per_data_element != layer_info->second.bytes_per_data_element)
			return false;

		char* other_layer_data = other.get_layer<char>(layer);
		char* this_layer_data = this->get_layer<char>(layer);

		if (memcmp(this_layer_data, other_layer_data, this->voxel_count * layer_info->second.bytes_per_data_element) != 0)
			return false;
	}

	return true;
}

VoxelLayer::VoxelLayer(size_t bytes_per_voxel, size_t bytes_per_data_element, char* voxels, char* data, const std::string& unit, float statistical_error, size_t voxel_count, bool shall_free_buffers)
{
	this->voxel_count = voxel_count;
	this->bytes_per_voxel = bytes_per_voxel;
	this->bytes_per_data_element = bytes_per_data_element;
	this->data = data;
	this->voxels = voxels;
	this->unit = unit;
	this->statistical_error = statistical_error;
	this->shall_free_buffers = shall_free_buffers;
}

VoxelLayer::VoxelLayer()
{
	this->voxel_count = 0;
	this->bytes_per_voxel = 0;
	this->bytes_per_data_element = 0;
	this->data = nullptr;
	this->voxels = nullptr;
	this->statistical_error = -1.f;
	this->shall_free_buffers = false;
}

VoxelLayer::~VoxelLayer()
{
	if (this->shall_free_buffers)
		this->free_buffers();
}

void VoxelLayer::free_buffers() noexcept
{
	if (this->data != nullptr)
		delete[] this->data;
	if (this->voxels != nullptr)
		delete[] this->voxels;

	this->data = nullptr;
	this->voxels = nullptr;
}

VoxelBuffer::~VoxelBuffer()
{
	for (auto& layer : this->layers)
	{
		layer.second.free_buffers();
	}
}

VoxelBuffer* VoxelBuffer::copy() const
{
	VoxelBuffer* copy = new VoxelBuffer(this->voxel_count);

	for (auto& layer : this->layers)
	{
		auto& layer_info = layer.second;
		auto data = new char[this->voxel_count * layer_info.bytes_per_data_element];
		auto voxels = new char[this->voxel_count * layer_info.bytes_per_voxel];
		memcpy(data, layer_info.data, this->voxel_count * layer_info.bytes_per_data_element);
		memcpy(voxels, layer_info.voxels, this->voxel_count * layer_info.bytes_per_voxel);
		copy->layers[layer.first] = VoxelLayer(layer_info.bytes_per_voxel, layer_info.bytes_per_data_element, voxels, data, layer_info.unit, layer_info.statistical_error, this->voxel_count);
	}

	return copy;
}


VoxelBuffer& VoxelBuffer::operator+=(const VoxelBuffer& other)
{
	if (this->voxel_count != other.voxel_count)
		throw std::runtime_error("Voxel count mismatch");

	for (auto& layer : this->layers)
	{
		auto other_layer_info = other.layers.find(layer.first);
		if (other_layer_info == other.layers.end())
			throw std::runtime_error("Layer: '" + layer.first + "' not found in other");
		if (layer.second.unit != other_layer_info->second.unit)
			throw std::runtime_error("Unit mismatch");
		auto dtype1 = Typing::Helper::get_dtype(this->get_type(layer.first));
		auto dtype2 = Typing::Helper::get_dtype(other.get_type(layer.first));
		if (dtype1 != dtype2)
			throw std::runtime_error("Data type mismatch");
		auto layer_info = this->layers.find(layer.first);
		if (other_layer_info->second.bytes_per_data_element != layer_info->second.bytes_per_data_element)
			throw std::runtime_error("Data element size mismatch");

		

		char* other_layer_data = other.get_layer<char>(layer.first);
		char* this_layer_data = this->get_layer<char>(layer.first);

		switch (dtype1)
		{
		case Typing::DType::Float:
			add_layers_together<float>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Double:
#if defined(__x86_64__) || defined(_M_X64)
			add_layers_together<double>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case Typing::DType::Int:
			add_layers_together<int>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Char:
			add_layers_together<char>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Byte:
			add_layers_together<uint8_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Vec2:
			add_layers_together<glm::vec2>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Vec3:
			add_layers_together<glm::vec3>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Vec4:
			add_layers_together<glm::vec4>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::UInt64:
#if defined(__x86_64__) || defined(_M_X64)
			add_layers_together<uint64_t>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case Typing::DType::UInt32:
			add_layers_together<uint32_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist1 = (HistogramVoxel*)(layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				auto hist2 = (HistogramVoxel*)(other_layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				*hist1 += *hist2;
			}
			break;
		}
	}

	return *this;
}

VoxelBuffer& VoxelBuffer::operator*=(const VoxelBuffer& other) {
	if (this->voxel_count != other.voxel_count)
		throw std::runtime_error("Voxel count mismatch");

	for (auto& layer : this->layers)
	{
		auto other_layer_info = other.layers.find(layer.first);
		if (other_layer_info == other.layers.end())
			throw std::runtime_error("Layer: '" + layer.first + "' not found in other");
		if (layer.second.unit != other_layer_info->second.unit)
			throw std::runtime_error("Unit mismatch");
		auto dtype1 = Typing::Helper::get_dtype(this->get_type(layer.first));
		auto dtype2 = Typing::Helper::get_dtype(other.get_type(layer.first));
		if (dtype1 != dtype2)
			throw std::runtime_error("Data type mismatch");
		auto layer_info = this->layers.find(layer.first);
		if (other_layer_info->second.bytes_per_data_element != layer_info->second.bytes_per_data_element)
			throw std::runtime_error("Data element size mismatch");

		char* other_layer_data = other.get_layer<char>(layer.first);
		char* this_layer_data = this->get_layer<char>(layer.first);

		switch (dtype1)
		{
		case Typing::DType::Float:
			multiply_layers_together<float>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Double:
#if defined(__x86_64__) || defined(_M_X64)
			multiply_layers_together<double>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case Typing::DType::Int:
			multiply_layers_together<int>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Char:
			multiply_layers_together<char>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Byte:
			multiply_layers_together<uint8_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Vec2:
			multiply_layers_together<glm::vec2>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Vec3:
			multiply_layers_together<glm::vec3>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Vec4:
			multiply_layers_together<glm::vec4>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::UInt64:
#if defined(__x86_64__) || defined(_M_X64)
			multiply_layers_together<uint64_t>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case Typing::DType::UInt32:
			multiply_layers_together<uint32_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist1 = (HistogramVoxel*)(layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				auto hist2 = (HistogramVoxel*)(other_layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				*hist1 *= *hist2;
			}
			break;
		}
	}

	return *this;
}

VoxelBuffer& VoxelBuffer::operator-=(const VoxelBuffer& other) {
	if (this->voxel_count != other.voxel_count)
		throw std::runtime_error("Voxel count mismatch");

	for (auto& layer : this->layers)
	{
		auto other_layer_info = other.layers.find(layer.first);
		if (other_layer_info == other.layers.end())
			throw std::runtime_error("Layer: '" + layer.first + "' not found in other");
		if (layer.second.unit != other_layer_info->second.unit)
			throw std::runtime_error("Unit mismatch");
		auto dtype1 = Typing::Helper::get_dtype(this->get_type(layer.first));
		auto dtype2 = Typing::Helper::get_dtype(other.get_type(layer.first));
		if (dtype1 != dtype2)
			throw std::runtime_error("Data type mismatch");
		auto layer_info = this->layers.find(layer.first);
		if (other_layer_info->second.bytes_per_data_element != layer_info->second.bytes_per_data_element)
			throw std::runtime_error("Data element size mismatch");

		char* other_layer_data = other.get_layer<char>(layer.first);
		char* this_layer_data = this->get_layer<char>(layer.first);

		switch (dtype1)
		{
		case Typing::DType::Float:
			subtract_layers_together<float>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Double:
#if defined(__x86_64__) || defined(_M_X64)
			subtract_layers_together<double>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case Typing::DType::Int:
			subtract_layers_together<int>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Char:
			subtract_layers_together<char>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Byte:
			subtract_layers_together<uint8_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Vec2:
			subtract_layers_together<glm::vec2>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Vec3:
			subtract_layers_together<glm::vec3>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Vec4:
			subtract_layers_together<glm::vec4>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::UInt64:
#if defined(__x86_64__) || defined(_M_X64)
			subtract_layers_together<uint64_t>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case Typing::DType::UInt32:
			subtract_layers_together<uint32_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist1 = (HistogramVoxel*)(layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				auto hist2 = (HistogramVoxel*)(other_layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				*hist1 -= *hist2;
			}
			break;
		}
	}

	return *this;
}

VoxelBuffer& VoxelBuffer::operator/=(const VoxelBuffer& other) {
	if (this->voxel_count != other.voxel_count)
		throw std::runtime_error("Voxel count mismatch");

	for (auto& layer : this->layers)
	{
		auto other_layer_info = other.layers.find(layer.first);
		if (other_layer_info == other.layers.end())
			throw std::runtime_error("Layer: '" + layer.first + "' not found in other");
		if (layer.second.unit != other_layer_info->second.unit)
			throw std::runtime_error("Unit mismatch");
		auto dtype1 = Typing::Helper::get_dtype(this->get_type(layer.first));
		auto dtype2 = Typing::Helper::get_dtype(other.get_type(layer.first));
		if (dtype1 != dtype2)
			throw std::runtime_error("Data type mismatch");
		auto layer_info = this->layers.find(layer.first);
		if (other_layer_info->second.bytes_per_data_element != layer_info->second.bytes_per_data_element)
			throw std::runtime_error("Data element size mismatch");

		char* other_layer_data = other.get_layer<char>(layer.first);
		char* this_layer_data = this->get_layer<char>(layer.first);

		switch (dtype1)
		{
		case Typing::DType::Float:
			divide_layers_together<float>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Double:
#if defined(__x86_64__) || defined(_M_X64)
			divide_layers_together<double>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case Typing::DType::Int:
			divide_layers_together<int>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Char:
			divide_layers_together<char>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Byte:
			divide_layers_together<uint8_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Vec2:
			divide_layers_together<glm::vec2>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Vec3:
			divide_layers_together<glm::vec3>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Vec4:
			divide_layers_together<glm::vec4>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::UInt64:
#if defined(__x86_64__) || defined(_M_X64)
			divide_layers_together<uint64_t>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case Typing::DType::UInt32:
			divide_layers_together<uint32_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case Typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist1 = (HistogramVoxel*)(layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				auto hist2 = (HistogramVoxel*)(other_layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				*hist1 /= *hist2;
			}
			break;
		}
	}
	return *this;
}

VoxelBuffer& VoxelBuffer::operator+=(const float& scalar) {
	for (auto& layer : this->layers)
	{
		auto layer_info = layer.second;
		char* this_layer_data = layer.second.data;

		switch (Typing::Helper::get_dtype(this->get_type(layer.first)))
		{
		case Typing::DType::Float:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				float* this_data = (float*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case Typing::DType::Double:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				double* this_data = (double*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case Typing::DType::Int:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				int* this_data = (int*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case Typing::DType::Char:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				char* this_data = (char*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case Typing::DType::Byte:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint8_t* this_data = (uint8_t*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case Typing::DType::Vec2:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec2* this_data = (glm::vec2*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case Typing::DType::Vec3:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec3* this_data = (glm::vec3*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case Typing::DType::Vec4:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec4* this_data = (glm::vec4*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case Typing::DType::UInt64:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint64_t* this_data = (uint64_t*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case Typing::DType::UInt32:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint32_t* this_data = (uint32_t*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case Typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist = (HistogramVoxel*)(layer_info.voxels + i * layer_info.bytes_per_voxel);
				*hist += scalar;
			}
			break;
		}
	}
	return *this;
}

VoxelBuffer& VoxelBuffer::operator-=(const float& scalar) {
	for (auto& layer : this->layers)
	{
		auto layer_info = layer.second;
		char* this_layer_data = layer.second.data;

		switch (Typing::Helper::get_dtype(this->get_type(layer.first)))
		{
		case Typing::DType::Float:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				float* this_data = (float*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case Typing::DType::Double:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				double* this_data = (double*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case Typing::DType::Int:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				int* this_data = (int*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case Typing::DType::Char:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				char* this_data = (char*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case Typing::DType::Byte:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint8_t* this_data = (uint8_t*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case Typing::DType::Vec2:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec2* this_data = (glm::vec2*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case Typing::DType::Vec3:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec3* this_data = (glm::vec3*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case Typing::DType::Vec4:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec4* this_data = (glm::vec4*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case Typing::DType::UInt64:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint64_t* this_data = (uint64_t*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case Typing::DType::UInt32:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint32_t* this_data = (uint32_t*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case Typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist = (HistogramVoxel*)(layer_info.voxels + i * layer_info.bytes_per_voxel);
				*hist -= scalar;
			}
			break;
		}
	}
	return *this;
}

VoxelBuffer& VoxelBuffer::operator*=(const float& scalar) {
	for (auto& layer : this->layers)
	{
		auto layer_info = layer.second;
		char* this_layer_data = layer.second.data;

		switch (Typing::Helper::get_dtype(this->get_type(layer.first)))
		{
		case Typing::DType::Float:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				float* this_data = (float*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case Typing::DType::Double:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				double* this_data = (double*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case Typing::DType::Int:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				int* this_data = (int*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case Typing::DType::Char:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				char* this_data = (char*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case Typing::DType::Byte:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint8_t* this_data = (uint8_t*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case Typing::DType::Vec2:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec2* this_data = (glm::vec2*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case Typing::DType::Vec3:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec3* this_data = (glm::vec3*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case Typing::DType::Vec4:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec4* this_data = (glm::vec4*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case Typing::DType::UInt64:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint64_t* this_data = (uint64_t*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case Typing::DType::UInt32:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint32_t* this_data = (uint32_t*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case Typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist = (HistogramVoxel*)(layer_info.voxels + i * layer_info.bytes_per_voxel);
				*hist *= scalar;
			}
			break;
		}
	}
	return *this;
}

VoxelBuffer& VoxelBuffer::operator/=(const float& scalar) {
	for (auto& layer : this->layers)
	{
		auto layer_info = layer.second;
		char* this_layer_data = layer.second.data;

		switch (Typing::Helper::get_dtype(this->get_type(layer.first)))
		{
		case Typing::DType::Float:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				float* this_data = (float*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case Typing::DType::Double:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				double* this_data = (double*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case Typing::DType::Int:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				int* this_data = (int*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case Typing::DType::Char:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				char* this_data = (char*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case Typing::DType::Byte:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint8_t* this_data = (uint8_t*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case Typing::DType::Vec2:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec2* this_data = (glm::vec2*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case Typing::DType::Vec3:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec3* this_data = (glm::vec3*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case Typing::DType::Vec4:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec4* this_data = (glm::vec4*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case Typing::DType::UInt64:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint64_t* this_data = (uint64_t*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case Typing::DType::UInt32:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint32_t* this_data = (uint32_t*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case Typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist = (HistogramVoxel*)(layer_info.voxels + i * layer_info.bytes_per_voxel);
				*hist /= scalar;
			}
			break;
		}
	}
	return *this;
}
