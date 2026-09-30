#include "radfiled3d/voxel_buffer.hpp"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>


using namespace radfiled3d;

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


static void throw_vmf_arithmetic(const std::string& operation, const std::string& layer_name)
{
	throw std::runtime_error("Element-wise " + operation + " is not defined for vMF mixture layers (layer: '" + layer_name + "'). Combine mixtures with VMFMixtureVoxel::merge or a field join instead.");
}

// Checked before any layer is modified, so a failing operation leaves the buffer untouched.
static void ensure_no_vmf_layers(const VoxelBuffer& buffer, const std::string& operation)
{
	for (const auto& layer_name : buffer.get_layers()) {
		if (buffer.get_type(layer_name) == "vmf_mixture")
			throw_vmf_arithmetic(operation, layer_name);
	}
}


VoxelBuffer::VoxelBuffer(size_t voxel_count)
	: voxel_count(voxel_count)
{
}

// Headers derive from IVoxel::VoxelBaseHeader, whose own members are bookkeeping and not part of the voxel definition.
static bool voxel_headers_equal(const IVoxel& a, const IVoxel& b)
{
	const IVoxel::VoxelBaseHeader header_a = a.get_header();
	const IVoxel::VoxelBaseHeader header_b = b.get_header();
	if (header_a.header_bytes != header_b.header_bytes)
		return false;
	const size_t skip = (header_a.header_bytes >= sizeof(IVoxel::VoxelBaseHeader)) ? sizeof(IVoxel::VoxelBaseHeader) : 0;
	if (header_a.header_bytes == skip)
		return true;
	return memcmp((const char*)header_a.header + skip, (const char*)header_b.header + skip, header_a.header_bytes - skip) == 0;
}

bool VoxelBuffer::operator==(VoxelBuffer const& other) const
{
	if (this->voxel_count != other.voxel_count || this->layers.size() != other.layers.size())
		return false;

	// Compare the cheap layout first, so the data is only compared for structurally equal buffers
	for (auto& [name, layer] : this->layers)
	{
		auto other_layer = other.layers.find(name);
		if (other_layer == other.layers.end())
			return false;
		if (layer.unit != other_layer->second.unit || layer.bytes_per_data_element != other_layer->second.bytes_per_data_element || layer.bytes_per_voxel != other_layer->second.bytes_per_voxel)
			return false;
		if (this->voxel_count == 0)
			continue;
		const IVoxel& vx = *layer.get_voxel_flat_raw(0);
		const IVoxel& other_vx = *other_layer->second.get_voxel_flat_raw(0);
		if (vx.get_type() != other_vx.get_type() || vx.get_bytes() != other_vx.get_bytes() || !voxel_headers_equal(vx, other_vx))
			return false;
	}

	for (auto& [name, layer] : this->layers)
	{
		if (this->voxel_count == 0)
			break;
		const size_t data_bytes = this->voxel_count * layer.get_voxel_flat_raw(0)->get_bytes();
		if (memcmp(layer.data, other.layers.find(name)->second.data, data_bytes) != 0)
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

void VoxelBuffer::copy_layers_to(VoxelBuffer& target) const
{
	if (target.voxel_count != this->voxel_count)
		throw VoxelBufferException("Voxel count mismatch in copy");

	for (auto& layer : this->layers)
	{
		auto& layer_info = layer.second;
		// Composite voxels (histogram, angular, vMF) hold several data elements, so the data size is taken from the voxel itself
		const size_t bytes_per_voxel_databuffer = (this->voxel_count > 0) ? layer_info.get_voxel_flat_raw(0)->get_bytes() : 0;
		auto data = new char[this->voxel_count * bytes_per_voxel_databuffer];
		auto voxels = new char[this->voxel_count * layer_info.bytes_per_voxel];
		memcpy(data, layer_info.data, this->voxel_count * bytes_per_voxel_databuffer);
		memcpy(voxels, layer_info.voxels, this->voxel_count * layer_info.bytes_per_voxel);
		for (size_t i = 0; i < this->voxel_count; i++)
		{
			IVoxel* vx = (IVoxel*)(voxels + i * layer_info.bytes_per_voxel);
			vx->set_data((void*)(data + i * bytes_per_voxel_databuffer));
		}
		target.layers[layer.first] = VoxelLayer(layer_info.bytes_per_voxel, layer_info.bytes_per_data_element, voxels, data, layer_info.unit, layer_info.statistical_error, this->voxel_count);
	}
}

VoxelBuffer* VoxelBuffer::copy() const
{
	VoxelBuffer* copy = new VoxelBuffer(this->voxel_count);
	this->copy_layers_to(*copy);
	return copy;
}


VoxelBuffer& VoxelBuffer::operator+=(const VoxelBuffer& other)
{
	ensure_no_vmf_layers(*this, "addition");
	if (this->voxel_count != other.voxel_count)
		throw std::runtime_error("Voxel count mismatch");

	for (auto& layer : this->layers)
	{
		auto other_layer_info = other.layers.find(layer.first);
		if (other_layer_info == other.layers.end())
			throw std::runtime_error("Layer: '" + layer.first + "' not found in other");
		if (layer.second.unit != other_layer_info->second.unit)
			throw std::runtime_error("Unit mismatch");
		auto dtype1 = typing::Helper::get_dtype(this->get_type(layer.first));
		auto dtype2 = typing::Helper::get_dtype(other.get_type(layer.first));
		if (dtype1 != dtype2)
			throw std::runtime_error("Data type mismatch");
		auto layer_info = this->layers.find(layer.first);
		if (other_layer_info->second.bytes_per_data_element != layer_info->second.bytes_per_data_element)
			throw std::runtime_error("Data element size mismatch");

		

		char* other_layer_data = other.get_layer<char>(layer.first);
		char* this_layer_data = this->get_layer<char>(layer.first);

		switch (dtype1)
		{
		case typing::DType::Float:
			add_layers_together<float>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Double:
#if RADFILED3D_HAS_64BIT
			add_layers_together<double>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case typing::DType::Int:
			add_layers_together<int>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Char:
			add_layers_together<char>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Byte:
			add_layers_together<uint8_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Vec2:
			add_layers_together<glm::vec2>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Vec3:
			add_layers_together<glm::vec3>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Vec4:
			add_layers_together<glm::vec4>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::UInt64:
#if RADFILED3D_HAS_64BIT
			add_layers_together<uint64_t>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case typing::DType::Int64:
#if RADFILED3D_HAS_64BIT
			add_layers_together<int64_t>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case typing::DType::UInt32:
			add_layers_together<uint32_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist1 = (HistogramVoxel<float>*)(layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				auto hist2 = (HistogramVoxel<float>*)(other_layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				*hist1 += *hist2;
			}
			break;
		case typing::DType::AngularResolved:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto sph1 = (AngularResolvedVoxel<float>*)(layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				auto sph2 = (AngularResolvedVoxel<float>*)(other_layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				*sph1 += *sph2;
			}
			break;
		case typing::DType::VMFMixture:
			throw_vmf_arithmetic("addition", layer.first);
		}
	}

	return *this;
}

VoxelBuffer& VoxelBuffer::operator*=(const VoxelBuffer& other) {
	ensure_no_vmf_layers(*this, "multiplication");
	if (this->voxel_count != other.voxel_count)
		throw std::runtime_error("Voxel count mismatch");

	for (auto& layer : this->layers)
	{
		auto other_layer_info = other.layers.find(layer.first);
		if (other_layer_info == other.layers.end())
			throw std::runtime_error("Layer: '" + layer.first + "' not found in other");
		if (layer.second.unit != other_layer_info->second.unit)
			throw std::runtime_error("Unit mismatch");
		auto dtype1 = typing::Helper::get_dtype(this->get_type(layer.first));
		auto dtype2 = typing::Helper::get_dtype(other.get_type(layer.first));
		if (dtype1 != dtype2)
			throw std::runtime_error("Data type mismatch");
		auto layer_info = this->layers.find(layer.first);
		if (other_layer_info->second.bytes_per_data_element != layer_info->second.bytes_per_data_element)
			throw std::runtime_error("Data element size mismatch");

		char* other_layer_data = other.get_layer<char>(layer.first);
		char* this_layer_data = this->get_layer<char>(layer.first);

		switch (dtype1)
		{
		case typing::DType::Float:
			multiply_layers_together<float>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Double:
#if RADFILED3D_HAS_64BIT
			multiply_layers_together<double>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case typing::DType::Int:
			multiply_layers_together<int>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Char:
			multiply_layers_together<char>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Byte:
			multiply_layers_together<uint8_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Vec2:
			multiply_layers_together<glm::vec2>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Vec3:
			multiply_layers_together<glm::vec3>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Vec4:
			multiply_layers_together<glm::vec4>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::UInt64:
#if RADFILED3D_HAS_64BIT
			multiply_layers_together<uint64_t>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case typing::DType::Int64:
#if RADFILED3D_HAS_64BIT
			multiply_layers_together<int64_t>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case typing::DType::UInt32:
			multiply_layers_together<uint32_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist1 = (HistogramVoxel<float>*)(layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				auto hist2 = (HistogramVoxel<float>*)(other_layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				*hist1 *= *hist2;
			}
			break;
		case typing::DType::AngularResolved:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto sph1 = (AngularResolvedVoxel<float>*)(layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				auto sph2 = (AngularResolvedVoxel<float>*)(other_layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				*sph1 *= *sph2;
			}
			break;
		case typing::DType::VMFMixture:
			throw_vmf_arithmetic("multiplication", layer.first);
		}
	}

	return *this;
}

VoxelBuffer& VoxelBuffer::operator-=(const VoxelBuffer& other) {
	ensure_no_vmf_layers(*this, "subtraction");
	if (this->voxel_count != other.voxel_count)
		throw std::runtime_error("Voxel count mismatch");

	for (auto& layer : this->layers)
	{
		auto other_layer_info = other.layers.find(layer.first);
		if (other_layer_info == other.layers.end())
			throw std::runtime_error("Layer: '" + layer.first + "' not found in other");
		if (layer.second.unit != other_layer_info->second.unit)
			throw std::runtime_error("Unit mismatch");
		auto dtype1 = typing::Helper::get_dtype(this->get_type(layer.first));
		auto dtype2 = typing::Helper::get_dtype(other.get_type(layer.first));
		if (dtype1 != dtype2)
			throw std::runtime_error("Data type mismatch");
		auto layer_info = this->layers.find(layer.first);
		if (other_layer_info->second.bytes_per_data_element != layer_info->second.bytes_per_data_element)
			throw std::runtime_error("Data element size mismatch");

		char* other_layer_data = other.get_layer<char>(layer.first);
		char* this_layer_data = this->get_layer<char>(layer.first);

		switch (dtype1)
		{
		case typing::DType::Float:
			subtract_layers_together<float>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Double:
#if RADFILED3D_HAS_64BIT
			subtract_layers_together<double>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case typing::DType::Int:
			subtract_layers_together<int>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Char:
			subtract_layers_together<char>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Byte:
			subtract_layers_together<uint8_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Vec2:
			subtract_layers_together<glm::vec2>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Vec3:
			subtract_layers_together<glm::vec3>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Vec4:
			subtract_layers_together<glm::vec4>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::UInt64:
#if RADFILED3D_HAS_64BIT
			subtract_layers_together<uint64_t>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case typing::DType::Int64:
#if RADFILED3D_HAS_64BIT
			subtract_layers_together<int64_t>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case typing::DType::UInt32:
			subtract_layers_together<uint32_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist1 = (HistogramVoxel<float>*)(layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				auto hist2 = (HistogramVoxel<float>*)(other_layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				*hist1 -= *hist2;
			}
			break;
		case typing::DType::AngularResolved:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto sph1 = (AngularResolvedVoxel<float>*)(layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				auto sph2 = (AngularResolvedVoxel<float>*)(other_layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				*sph1 -= *sph2;
			}
			break;
		case typing::DType::VMFMixture:
			throw_vmf_arithmetic("subtraction", layer.first);
		}
	}

	return *this;
}

VoxelBuffer& VoxelBuffer::operator/=(const VoxelBuffer& other) {
	ensure_no_vmf_layers(*this, "division");
	if (this->voxel_count != other.voxel_count)
		throw std::runtime_error("Voxel count mismatch");

	for (auto& layer : this->layers)
	{
		auto other_layer_info = other.layers.find(layer.first);
		if (other_layer_info == other.layers.end())
			throw std::runtime_error("Layer: '" + layer.first + "' not found in other");
		if (layer.second.unit != other_layer_info->second.unit)
			throw std::runtime_error("Unit mismatch");
		auto dtype1 = typing::Helper::get_dtype(this->get_type(layer.first));
		auto dtype2 = typing::Helper::get_dtype(other.get_type(layer.first));
		if (dtype1 != dtype2)
			throw std::runtime_error("Data type mismatch");
		auto layer_info = this->layers.find(layer.first);
		if (other_layer_info->second.bytes_per_data_element != layer_info->second.bytes_per_data_element)
			throw std::runtime_error("Data element size mismatch");

		char* other_layer_data = other.get_layer<char>(layer.first);
		char* this_layer_data = this->get_layer<char>(layer.first);

		switch (dtype1)
		{
		case typing::DType::Float:
			divide_layers_together<float>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Double:
#if RADFILED3D_HAS_64BIT
			divide_layers_together<double>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case typing::DType::Int:
			divide_layers_together<int>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Char:
			divide_layers_together<char>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Byte:
			divide_layers_together<uint8_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Vec2:
			divide_layers_together<glm::vec2>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Vec3:
			divide_layers_together<glm::vec3>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Vec4:
			divide_layers_together<glm::vec4>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::UInt64:
#if RADFILED3D_HAS_64BIT
			divide_layers_together<uint64_t>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case typing::DType::Int64:
#if RADFILED3D_HAS_64BIT
			divide_layers_together<int64_t>(this_layer_data, other_layer_data, this->voxel_count);
#else
			throw std::runtime_error("Can't use 64-bit data type in 32-bit system!");
#endif
			break;
		case typing::DType::UInt32:
			divide_layers_together<uint32_t>(this_layer_data, other_layer_data, this->voxel_count);
			break;
		case typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist1 = (HistogramVoxel<float>*)(layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				auto hist2 = (HistogramVoxel<float>*)(other_layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				*hist1 /= *hist2;
			}
			break;
		case typing::DType::AngularResolved:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto sph1 = (AngularResolvedVoxel<float>*)(layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				auto sph2 = (AngularResolvedVoxel<float>*)(other_layer_info->second.voxels + i * layer_info->second.bytes_per_voxel);
				*sph1 /= *sph2;
			}
			break;
		case typing::DType::VMFMixture:
			throw_vmf_arithmetic("division", layer.first);
		}
	}
	return *this;
}

VoxelBuffer& VoxelBuffer::operator+=(const float& scalar) {
	ensure_no_vmf_layers(*this, "scalar addition");
	for (auto& layer : this->layers)
	{
		auto layer_info = layer.second;
		char* this_layer_data = layer.second.data;

		switch (typing::Helper::get_dtype(this->get_type(layer.first)))
		{
		case typing::DType::Float:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				float* this_data = (float*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case typing::DType::Double:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				double* this_data = (double*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case typing::DType::Int:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				int* this_data = (int*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case typing::DType::Char:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				char* this_data = (char*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case typing::DType::Byte:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint8_t* this_data = (uint8_t*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case typing::DType::Vec2:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec2* this_data = (glm::vec2*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case typing::DType::Vec3:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec3* this_data = (glm::vec3*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case typing::DType::Vec4:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec4* this_data = (glm::vec4*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case typing::DType::UInt64:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint64_t* this_data = (uint64_t*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case typing::DType::Int64:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				int64_t* this_data = (int64_t*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case typing::DType::UInt32:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint32_t* this_data = (uint32_t*)this_layer_data;
				this_data[i] += scalar;
			}
			break;
		case typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist = (HistogramVoxel<float>*)(layer_info.voxels + i * layer_info.bytes_per_voxel);
				*hist += scalar;
			}
			break;
		case typing::DType::AngularResolved:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto sph = (AngularResolvedVoxel<float>*)(layer_info.voxels + i * layer_info.bytes_per_voxel);
				*sph += scalar;
			}
			break;
		case typing::DType::VMFMixture:
			throw_vmf_arithmetic("scalar addition", layer.first);
		}
	}
	return *this;
}

VoxelBuffer& VoxelBuffer::operator-=(const float& scalar) {
	ensure_no_vmf_layers(*this, "scalar subtraction");
	for (auto& layer : this->layers)
	{
		auto layer_info = layer.second;
		char* this_layer_data = layer.second.data;

		switch (typing::Helper::get_dtype(this->get_type(layer.first)))
		{
		case typing::DType::Float:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				float* this_data = (float*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case typing::DType::Double:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				double* this_data = (double*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case typing::DType::Int:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				int* this_data = (int*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case typing::DType::Char:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				char* this_data = (char*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case typing::DType::Byte:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint8_t* this_data = (uint8_t*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case typing::DType::Vec2:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec2* this_data = (glm::vec2*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case typing::DType::Vec3:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec3* this_data = (glm::vec3*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case typing::DType::Vec4:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec4* this_data = (glm::vec4*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case typing::DType::UInt64:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint64_t* this_data = (uint64_t*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case typing::DType::Int64:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				int64_t* this_data = (int64_t*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case typing::DType::UInt32:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint32_t* this_data = (uint32_t*)this_layer_data;
				this_data[i] -= scalar;
			}
			break;
		case typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist = (HistogramVoxel<float>*)(layer_info.voxels + i * layer_info.bytes_per_voxel);
				*hist -= scalar;
			}
			break;
		case typing::DType::AngularResolved:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto sph = (AngularResolvedVoxel<float>*)(layer_info.voxels + i * layer_info.bytes_per_voxel);
				*sph -= scalar;
			}
			break;
		case typing::DType::VMFMixture:
			throw_vmf_arithmetic("scalar subtraction", layer.first);
		}
	}
	return *this;
}

VoxelBuffer& VoxelBuffer::operator*=(const float& scalar) {
	ensure_no_vmf_layers(*this, "scalar multiplication");
	for (auto& layer : this->layers)
	{
		auto layer_info = layer.second;
		char* this_layer_data = layer.second.data;

		switch (typing::Helper::get_dtype(this->get_type(layer.first)))
		{
		case typing::DType::Float:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				float* this_data = (float*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case typing::DType::Double:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				double* this_data = (double*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case typing::DType::Int:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				int* this_data = (int*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case typing::DType::Char:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				char* this_data = (char*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case typing::DType::Byte:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint8_t* this_data = (uint8_t*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case typing::DType::Vec2:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec2* this_data = (glm::vec2*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case typing::DType::Vec3:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec3* this_data = (glm::vec3*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case typing::DType::Vec4:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec4* this_data = (glm::vec4*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case typing::DType::UInt64:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint64_t* this_data = (uint64_t*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case typing::DType::Int64:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				int64_t* this_data = (int64_t*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case typing::DType::UInt32:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint32_t* this_data = (uint32_t*)this_layer_data;
				this_data[i] *= scalar;
			}
			break;
		case typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist = (HistogramVoxel<float>*)(layer_info.voxels + i * layer_info.bytes_per_voxel);
				*hist *= scalar;
			}
			break;
		case typing::DType::AngularResolved:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto sph = (AngularResolvedVoxel<float>*)(layer_info.voxels + i * layer_info.bytes_per_voxel);
				*sph *= scalar;
			}
			break;
		case typing::DType::VMFMixture:
			throw_vmf_arithmetic("scalar multiplication", layer.first);
		}
	}
	return *this;
}

VoxelBuffer& VoxelBuffer::operator/=(const float& scalar) {
	ensure_no_vmf_layers(*this, "scalar division");
	for (auto& layer : this->layers)
	{
		auto layer_info = layer.second;
		char* this_layer_data = layer.second.data;

		switch (typing::Helper::get_dtype(this->get_type(layer.first)))
		{
		case typing::DType::Float:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				float* this_data = (float*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case typing::DType::Double:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				double* this_data = (double*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case typing::DType::Int:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				int* this_data = (int*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case typing::DType::Char:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				char* this_data = (char*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case typing::DType::Byte:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint8_t* this_data = (uint8_t*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case typing::DType::Vec2:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec2* this_data = (glm::vec2*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case typing::DType::Vec3:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec3* this_data = (glm::vec3*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case typing::DType::Vec4:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				glm::vec4* this_data = (glm::vec4*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case typing::DType::UInt64:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint64_t* this_data = (uint64_t*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case typing::DType::Int64:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				int64_t* this_data = (int64_t*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case typing::DType::UInt32:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				uint32_t* this_data = (uint32_t*)this_layer_data;
				this_data[i] /= scalar;
			}
			break;
		case typing::DType::Hist:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto hist = (HistogramVoxel<float>*)(layer_info.voxels + i * layer_info.bytes_per_voxel);
				*hist /= scalar;
			}
			break;
		case typing::DType::AngularResolved:
			for (size_t i = 0; i < this->voxel_count; i++)
			{
				auto sph = (AngularResolvedVoxel<float>*)(layer_info.voxels + i * layer_info.bytes_per_voxel);
				*sph /= scalar;
			}
			break;
		case typing::DType::VMFMixture:
			throw_vmf_arithmetic("scalar division", layer.first);
		}
	}
	return *this;
}
