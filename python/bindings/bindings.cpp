#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <glm/glm.hpp>
#include <RadFiled3D/storage/RadiationFieldStore.hpp>
#include <RadFiled3D/RadiationField.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <stdexcept>
#include "RadFiled3D/storage/FieldSerializer.hpp"
#include "RadFiled3D/GridTracer.hpp"
#include <fstream>
#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <iostream>
#include <RadFiled3D/dataset/helpers.hpp>


namespace py = pybind11;
using namespace pybind11::detail;
using namespace RadFiled3D;
using namespace RadFiled3D::Storage;
using namespace RadFiled3D::Dataset;

struct NonDeletingDeleter {
    void operator()(void*) const {
        // Do nothing
    }
};

// This macro is used to return a shared_ptr that does not delete the object. Used for returning regular voxel pointers from radiation field buffers that are not holding their own data.
#define VOXEL_REFERENCE(vx) std::shared_ptr<IVoxel>(static_cast<IVoxel*>(vx), NonDeletingDeleter())
#define VOXEL_CAPSULE(vx, T) std::static_pointer_cast<IVoxel>(std::shared_ptr<T>(static_cast<T*>(vx)))

#define DECLARE_SCALAR_VOXEL(m, dT, name, parent) \
    py::class_<ScalarVoxel<dT>, std::shared_ptr<ScalarVoxel<dT>>, parent>(m, name)\
        .def("get_data", &ScalarVoxel<dT>::get_data, py::return_value_policy::reference)\
        .def("set_data", [](ScalarVoxel<dT>& v, dT value) {\
            v = value;\
        })\
        .def(py::self == py::self)\
        .def(py::self /= py::self)\
        .def(py::self *= py::self)\
        .def(py::self += py::self)\
        .def(py::self -= py::self)\
        .def("__repr__",\
            [](const ScalarVoxel<dT>& a) {\
                return "<RadFiled3D." + std::string(name) + " (" + std::to_string(a.get_data()) + ")>";\
            }\
        )

#define DECLARE_OWNING_SCALAR_VOXEL(m, dT, name, parent) \
    py::class_<OwningScalarVoxel<dT>, std::shared_ptr<OwningScalarVoxel<dT>>, parent>(m, name)\
        .def("get_data", &OwningScalarVoxel<dT>::get_data, py::return_value_policy::reference)\
        .def("set_data", [](OwningScalarVoxel<dT>& v, dT value) {\
            v.set_data(&value);\
        })\
        .def(py::self == py::self)\
        .def(py::self /= py::self)\
        .def(py::self *= py::self)\
        .def(py::self += py::self)\
        .def(py::self -= py::self)\
        .def("__repr__",\
            [](const OwningScalarVoxel<dT>& a) {\
                return "<RadFiled3D." + std::string(name) + " (" + std::to_string(a.get_data()) + ")>";\
            }\
        )


std::shared_ptr<IVoxel> encapsulate_voxel(IVoxel* vx) {
    const Typing::DType type = Typing::Helper::get_dtype(vx->get_type());
    switch (type) {
    case Typing::DType::Float:
        return VOXEL_CAPSULE(vx, ScalarVoxel<float>);
    case Typing::DType::Double:
        return VOXEL_CAPSULE(vx, ScalarVoxel<double>);
    case Typing::DType::Int:
        return VOXEL_CAPSULE(vx, ScalarVoxel<int>);
    case Typing::DType::Char:
        return VOXEL_CAPSULE(vx, ScalarVoxel<char>);
    case Typing::DType::Vec2:
        return VOXEL_CAPSULE(vx, ScalarVoxel<glm::vec2>);
    case Typing::DType::Vec3:
        return VOXEL_CAPSULE(vx, ScalarVoxel<glm::vec3>);
    case Typing::DType::Vec4:
        return VOXEL_CAPSULE(vx, ScalarVoxel<glm::vec4>);
    case Typing::DType::Hist:
        return VOXEL_CAPSULE(vx, HistogramVoxel);
    case Typing::DType::UInt64:
        return VOXEL_CAPSULE(vx, ScalarVoxel<uint64_t>);
    case Typing::DType::UInt32:
        return VOXEL_CAPSULE(vx, ScalarVoxel<uint32_t>);
    }
    throw std::runtime_error("Unsupported voxel type: " + std::to_string(static_cast<int>(type)));
}

std::map<void*, std::pair<std::atomic<size_t>, std::shared_ptr<void>>> shared_ptrs;

typedef std::tuple<std::string, std::string, std::string, std::string, std::string> MetadataHeaderSoftwarePickleTuple;
typedef std::tuple<
    float,
    std::tuple<
    float,
    float,
    float
    >,
    std::tuple<
    float,
    float,
    float
    >,
    std::string
> MetadataHeaderSimulationXRayTubePickleTuple;
typedef std::tuple<size_t, std::string, std::string, MetadataHeaderSimulationXRayTubePickleTuple> MetadataHeaderSimulationPickleTuple;
typedef std::tuple<MetadataHeaderSimulationPickleTuple, MetadataHeaderSoftwarePickleTuple> MetadataHeaderPickleTuple;

typedef std::tuple<float, float> Vec2PickleTuple;
typedef std::tuple<float, float, float> Vec3PickleTuple;
typedef std::tuple<float, float, float, float> Vec4PickleTuple;
typedef std::tuple<unsigned int, unsigned int, unsigned int, unsigned int> UVec4PickleTuple;
typedef std::tuple<unsigned int, unsigned int, unsigned int> UVec3PickleTuple;
typedef std::tuple<unsigned int, unsigned int> UVec2PickleTuple;
typedef std::tuple<FieldType, std::vector<char>> FieldAccessorPickleTuple;

class PyGridTracerFactory {
public:
    static std::shared_ptr<GridTracer> construct(std::shared_ptr<CartesianRadiationField> field, GridTracerAlgorithm algorithm) {
		if (field->get_typename() != "CartesianRadiationField") {
			throw std::invalid_argument("Field is not a CartesianRadiationField");
		}

        auto channels = field->get_channels();

        if (channels.size() == 0) {
            throw std::invalid_argument("No channels in field");
        }

        VoxelGridBuffer& grid = *static_cast<VoxelGridBuffer*>(channels[0].second.get());
        switch (algorithm) {
        case GridTracerAlgorithm::SAMPLING:
            return std::make_shared<SamplingGridTracer>(grid);
        case GridTracerAlgorithm::BRESENHAM:
            return std::make_shared<BresenhamGridTracer>(grid);
        case GridTracerAlgorithm::LINETRACING:
            return std::make_shared<LinetracingGridTracer>(grid);
        default:
            throw std::invalid_argument("Unknown algorithm");
        }
    }
};

template<typename T>
py::array create_py_array_generic(const T* data, const glm::uvec3& shape, size_t element_size, std::shared_ptr<void> ptr) {
    auto memory_safe_struct = shared_ptrs.find((void*)data);
    if (memory_safe_struct != shared_ptrs.end()) {
        memory_safe_struct->second.first++;
    }
    else {
        shared_ptrs[(void*)data] = std::make_pair(1, ptr);
    }

    const size_t components = static_cast<size_t>(element_size / sizeof(T));
    return static_cast<py::array>(py::array_t<T>(
        { static_cast<size_t>(shape.x), static_cast<size_t>(shape.y), static_cast<size_t>(shape.z), components },  // shape
        { static_cast<size_t>(element_size * shape.y * shape.z), static_cast<size_t>(shape.z * element_size), static_cast<size_t>(element_size), sizeof(T) },  // strides
        data,
        py::capsule(data, [](void* data) {
            auto memory_safe_struct = shared_ptrs.find((void*)data);
            if (memory_safe_struct != shared_ptrs.end()) {
                memory_safe_struct->second.first--;
                if (memory_safe_struct->second.first == 0) {
                    shared_ptrs.erase(memory_safe_struct);
                }
            }
        })
    ));
}

template<typename T>
py::array create_py_array_generic(const T* data, size_t len, size_t element_size = sizeof(T)) {
    const size_t components = static_cast<size_t>(element_size / sizeof(T));
    return (components > 1) ? static_cast<py::array>(py::array_t<T>(
		{ len, components },  // shape
		{ element_size, sizeof(T) },  // strides
		data
	)) : static_cast<py::array>(py::array_t<T>(
		{ len },  // shape
		{ sizeof(T) },  // strides
		data,
        py::capsule(data, [](void* data) {
			delete[] static_cast<T*>(data);
        })
	));
}

template<typename T>
py::array create_py_array_generic(const T* data, const glm::uvec2& shape, size_t element_size, std::shared_ptr<void> ptr) {
    auto memory_safe_struct = shared_ptrs.find((void*)data);
	if (memory_safe_struct != shared_ptrs.end()) {
		memory_safe_struct->second.first++;
	}
	else {
		shared_ptrs[(void*)data] = std::make_pair(1, ptr);
	}

    const size_t components = static_cast<size_t>(element_size / sizeof(T));
    return static_cast<py::array>(py::array_t<T>(
        { static_cast<size_t>(shape.x), static_cast<size_t>(shape.y), components },  // shape
        { static_cast<size_t>(element_size * shape.y), static_cast<size_t>(element_size), sizeof(T) },  // strides
        data,
        py::capsule(data, [](void* data) {
            auto memory_safe_struct = shared_ptrs.find((void*)data);
			if (memory_safe_struct != shared_ptrs.end()) {
				memory_safe_struct->second.first--;
				if (memory_safe_struct->second.first == 0) {
					shared_ptrs.erase(memory_safe_struct);
				}
			}
        })
    ));
}

template<typename T>
py::array create_py_array_as(const T* data, const glm::uvec3& shape, std::shared_ptr<void> ptr) {
    auto memory_safe_struct = shared_ptrs.find((void*)data);
    if (memory_safe_struct != shared_ptrs.end()) {
        memory_safe_struct->second.first++;
    }
    else {
        shared_ptrs[(void*)data] = std::make_pair(1, ptr);
    }

    return static_cast<py::array>(py::array_t<T>(
        { shape.x, shape.y, shape.z },  // shape
        { sizeof(T) * shape.y * shape.z, shape.z * sizeof(T), sizeof(T) },  // strides
        data,
        py::capsule(data, [](void* data) {
            auto memory_safe_struct = shared_ptrs.find((void*)data);
            if (memory_safe_struct != shared_ptrs.end()) {
                memory_safe_struct->second.first--;
                if (memory_safe_struct->second.first == 0) {
                    shared_ptrs.erase(memory_safe_struct);
                }
            }
        })
    ));
}

template<typename T>
py::array create_py_array_as(const T* data, const glm::uvec2& shape, std::shared_ptr<void> ptr) {
    auto memory_safe_struct = shared_ptrs.find((void*)data);
    if (memory_safe_struct != shared_ptrs.end()) {
        memory_safe_struct->second.first++;
    }
    else {
        shared_ptrs[(void*)data] = std::make_pair(1, ptr);
    }

	return static_cast<py::array>(py::array_t<T>(
		{ static_cast<size_t>(shape.x), static_cast<size_t>(shape.y) },  // shape
		{ static_cast<size_t>(sizeof(T) * shape.y), sizeof(T) },  // strides
		data,
        py::capsule(data, [](void* data) {
            auto memory_safe_struct = shared_ptrs.find((void*)data);
            if (memory_safe_struct != shared_ptrs.end()) {
                memory_safe_struct->second.first--;
                if (memory_safe_struct->second.first == 0) {
                    shared_ptrs.erase(memory_safe_struct);
                }
            }
        })
	));
}


PYBIND11_MODULE(RadFiled3D, m) {
    m.doc() = R"pbdoc(
        RadFiled3D for Python loading of RadiationFieldStores
        -----------------------
        .. currentmodule:: RadFiled3D
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    py::register_exception<std::runtime_error>(m, "RuntimeError");
	py::register_exception<std::invalid_argument>(m, "InvalidArgument");
	py::register_exception<std::out_of_range>(m, "OutOfRange");
	py::register_exception<std::exception>(m, "Exception");
	py::register_exception<RadFiled3D::VoxelBufferException>(m, "VoxelBufferException");
	py::register_exception<RadFiled3D::RadiationFieldStoreException>(m, "RadiationFieldStoreException");

    py::class_<glm::uvec4>(m, "uvec4")
        .def(py::init<unsigned int, unsigned int, unsigned int, unsigned int>())
        .def_readwrite("x", &glm::uvec4::x)
        .def_readwrite("y", &glm::uvec4::y)
        .def_readwrite("z", &glm::uvec4::z)
        .def_readwrite("w", &glm::uvec4::w)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self / py::self)
        .def(py::self /= py::self)
        .def(py::self == py::self)
        .def("__mul__",
            [](const glm::uvec4& a, unsigned int scalar) {
                return a * scalar;
            })
        .def("__rmul__",
            [](const glm::uvec4& a, unsigned int scalar) {
                return a * scalar;
            })
        .def("__truediv__",
            [](const glm::uvec4& a, unsigned int scalar) {
                return a / scalar;
            })
        .def("__rtruediv__",
            [](const glm::uvec4& a, unsigned int scalar) {
                return a / scalar;
            })
        .def("__add__",
            [](const glm::uvec4& a, unsigned int scalar) {
                return a + scalar;
            })
        .def("__radd__",
            [](const glm::uvec4& a, unsigned int scalar) {
                return a + scalar;
            })
        .def("__sub__",
            [](const glm::uvec4& a, unsigned int scalar) {
                return a - scalar;
            })
        .def("__rsub__",
            [](const glm::uvec4& a, unsigned int scalar) {
                return a - scalar;
            })
		.def(py::pickle([](const glm::uvec4& a) {
		    return UVec4PickleTuple(a.x, a.y, a.z, a.w);
	    },
			[](const UVec4PickleTuple& t) {
				return glm::uvec4(std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t));
		}))
        .def(-py::self)
        .def(py::pickle([](const glm::uvec4& a) {
            return UVec4PickleTuple(a.x, a.y, a.z, a.w);
        },
            [](const UVec4PickleTuple& t) {
                return glm::uvec4(std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t));
        }))
        .def("__repr__",
            [](const glm::uvec4& a) {
                return "<glm.uvec4 (" + std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + ", " + std::to_string(a.w) + ")>";
            });

    py::class_<glm::vec4>(m, "vec4")
        .def(py::init<float, float, float, float>())
        .def_readwrite("x", &glm::vec4::x)
        .def_readwrite("y", &glm::vec4::y)
        .def_readwrite("z", &glm::vec4::z)
        .def_readwrite("w", &glm::vec4::w)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self / py::self)
        .def(py::self /= py::self)
        .def(py::self == py::self)
        .def(-py::self)
        .def("__mul__",
            [](const glm::vec4& a, float scalar) {
                return a * scalar;
            })
        .def("__rmul__",
            [](const glm::vec4& a, float scalar) {
                return a * scalar;
            })
        .def("__truediv__",
            [](const glm::vec4& a, float scalar) {
                return a / scalar;
            })
        .def("__rtruediv__",
            [](const glm::vec4& a, float scalar) {
                return a / scalar;
            })
        .def("__add__",
            [](const glm::vec4& a, float scalar) {
                return a + scalar;
            })
        .def("__radd__",
            [](const glm::vec4& a, float scalar) {
                return a + scalar;
            })
        .def("__sub__",
            [](const glm::vec4& a, float scalar) {
                return a - scalar;
            })
        .def("__rsub__",
            [](const glm::vec4& a, float scalar) {
                return a - scalar;
            })
		.def(py::pickle([](const glm::vec4& a) {
		    return Vec4PickleTuple(a.x, a.y, a.z, a.w);
		},
		[](const Vec4PickleTuple& t) {
			return glm::vec4(std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t));
		}))
        .def("__repr__",
            [](const glm::vec4& a) {
                return "<glm.vec4 (" + std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + ", " + std::to_string(a.w) + ")>";
            });

    py::class_<glm::uvec3>(m, "uvec3")
        .def(py::init<unsigned int, unsigned int, unsigned int>())
        .def_readwrite("x", &glm::uvec3::x)
        .def_readwrite("y", &glm::uvec3::y)
        .def_readwrite("z", &glm::uvec3::z)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self / py::self)
        .def(py::self /= py::self)
        .def(py::self == py::self)
        .def(-py::self)
        .def("__mul__",
            [](const glm::uvec3& a, unsigned int scalar) {
                return a * scalar;
            })
        .def("__rmul__",
            [](const glm::uvec3& a, unsigned int scalar) {
                return a * scalar;
            })
        .def("__truediv__",
            [](const glm::uvec3& a, unsigned int scalar) {
                return a / scalar;
            })
        .def("__rtruediv__",
            [](const glm::uvec3& a, unsigned int scalar) {
                return a / scalar;
            })
        .def("__add__",
            [](const glm::uvec3& a, unsigned int scalar) {
                return a + scalar;
            })
        .def("__radd__",
            [](const glm::uvec3& a, unsigned int scalar) {
                return a + scalar;
            })
        .def("__sub__",
            [](const glm::uvec3& a, unsigned int scalar) {
                return a - scalar;
            })
        .def("__rsub__",
            [](const glm::uvec3& a, unsigned int scalar) {
                return a - scalar;
            })
		.def(py::pickle([](const glm::uvec3& a) {
		    return UVec3PickleTuple(a.x, a.y, a.z);
		},
	    [](const UVec3PickleTuple& t) {
			return glm::uvec3(std::get<0>(t), std::get<1>(t), std::get<2>(t));
		}))
        .def("__repr__",
            [](const glm::uvec3& a) {
                return "<glm.uvec3 (" + std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + ")>";
            });

    py::class_<glm::vec3>(m, "vec3")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &glm::vec3::x)
        .def_readwrite("y", &glm::vec3::y)
        .def_readwrite("z", &glm::vec3::z)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self / py::self)
        .def(py::self /= py::self)
        .def(py::self == py::self)
        .def(-py::self)
        .def("__mul__",
            [](const glm::vec3& a, float scalar) {
                return a * scalar;
        })
        .def("__rmul__",
            [](const glm::vec3& a, float scalar) {
                return a * scalar;
        })
        .def("__truediv__",
            [](const glm::vec3& a, float scalar) {
                return a / scalar;
        })
        .def("__rtruediv__",
            [](const glm::vec3& a, float scalar) {
                return a / scalar;
        })
        .def("__add__",
            [](const glm::vec3& a, float scalar) {
				return a + scalar;
	    })
        .def("__radd__",
			[](const glm::vec3& a, float scalar) {
				return a + scalar;
	    })
        .def("__sub__",
			[](const glm::vec3& a, float scalar) {
                return a - scalar;
        })
        .def("__rsub__",
            [](const glm::vec3& a, float scalar) {
				return a - scalar;
		})
		.def(py::pickle([](const glm::vec3& a) {
		    return Vec3PickleTuple(a.x, a.y, a.z);
		},
		[](const Vec3PickleTuple& t) {
                return glm::vec3(std::get<0>(t), std::get<1>(t), std::get<2>(t));
	    }))
        .def("__repr__",
            [](const glm::vec3& a) {
                return "<glm.vec3 (" + std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + ")>";
        });

    py::class_<glm::uvec2>(m, "uvec2")
        .def(py::init<unsigned int, unsigned int>())
        .def_readwrite("x", &glm::uvec2::x)
        .def_readwrite("y", &glm::uvec2::y)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self / py::self)
        .def(py::self /= py::self)
        .def(py::self == py::self)
        .def(-py::self)
        .def("__mul__",
            [](const glm::uvec2& a, unsigned int scalar) {
                return a * scalar;
            })
        .def("__rmul__",
            [](const glm::uvec2& a, unsigned int scalar) {
                return a * scalar;
            })
        .def("__truediv__",
            [](const glm::uvec2& a, unsigned int scalar) {
                return a / scalar;
            })
        .def("__rtruediv__",
            [](const glm::uvec2& a, unsigned int scalar) {
                return a / scalar;
            })
        .def("__add__",
            [](const glm::uvec2& a, unsigned int scalar) {
                return a + scalar;
            })
        .def("__radd__",
            [](const glm::uvec2& a, unsigned int scalar) {
                return a + scalar;
            })
        .def("__sub__",
            [](const glm::uvec2& a, unsigned int scalar) {
                return a - scalar;
            })
        .def("__rsub__",
            [](const glm::uvec2& a, unsigned int scalar) {
                return a - scalar;
            })
		.def(py::pickle([](const glm::uvec2& a) {
		    return UVec2PickleTuple(a.x, a.y);
		},
		[](const UVec2PickleTuple& t) {
			return glm::uvec2(std::get<0>(t), std::get<1>(t));
		}))
        .def("__repr__",
            [](const glm::uvec2& a) {
                return "<glm.uvec2 (" + std::to_string(a.x) + ", " + std::to_string(a.y) + ")>";
            });

    py::class_<glm::vec2>(m, "vec2")
        .def(py::init<float, float>())
        .def_readwrite("x", &glm::vec2::x)
        .def_readwrite("y", &glm::vec2::y)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self / py::self)
        .def(py::self /= py::self)
        .def(py::self == py::self)
        .def(-py::self)
        .def("__mul__",
            [](const glm::vec2& a, float scalar) {
                return a * scalar;
            })
        .def("__rmul__",
            [](const glm::vec2& a, float scalar) {
                return a * scalar;
            })
        .def("__truediv__",
            [](const glm::vec2& a, float scalar) {
                return a / scalar;
            })
        .def("__rtruediv__",
            [](const glm::vec2& a, float scalar) {
                return a / scalar;
            })
        .def("__add__",
            [](const glm::vec2& a, float scalar) {
                return a + scalar;
            })
        .def("__radd__",
            [](const glm::vec2& a, float scalar) {
                return a + scalar;
            })
        .def("__sub__",
            [](const glm::vec2& a, float scalar) {
                return a - scalar;
            })
        .def("__rsub__",
            [](const glm::vec2& a, float scalar) {
                return a - scalar;
            })
		.def(py::pickle([](const glm::vec2& a) {
		    return Vec2PickleTuple(a.x, a.y);
	    },
		[](const Vec2PickleTuple& t) {
		    return glm::vec2(std::get<0>(t), std::get<1>(t));
	    }))
        .def("__repr__",
            [](const glm::vec2& a) {
                return "<glm.vec2 (" + std::to_string(a.x) + ", " + std::to_string(a.y) + ")>";
            });

    py::class_<FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube>(m, "RadiationFieldXRayTubeMetadataV1")
		.def(py::init<const glm::vec3&, const glm::vec3&, float, const std::string&>(), py::arg("radiation_direction"), py::arg("radiation_origin"), py::arg("max_energy_eV"), py::arg("tube_id"))
		.def_readwrite("radiation_direction", &FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube::radiation_direction)
		.def_readwrite("radiation_origin", &FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube::radiation_origin)
		.def_readwrite("max_energy_eV", &FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube::max_energy_eV)
		.def_property("tube_id",
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube& self) -> std::string {
                return std::string(self.tube_id);
            },
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube& self, const std::string& s) {
                strncpy(self.tube_id, s.c_str(), std::min(sizeof(self.tube_id), s.length()));
            }
        );

    py::class_<FiledTypes::V1::RadiationFieldMetadataHeader::Simulation>(m, "RadiationFieldSimulationMetadataV1")
        .def(py::init<size_t, const std::string&, const std::string&, const FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube&>(), py::arg("primary_particle_count"), py::arg("geometry"), py::arg("physics_list"), py::arg("tube"))
		.def_readwrite("primary_particle_count", &FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::primary_particle_count)
        .def_property("geometry",
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Simulation& self) -> std::string {
                return std::string(self.geometry);
            },
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Simulation& self, const std::string& s) {
                strncpy(self.geometry, s.c_str(), std::min(sizeof(self.geometry), s.length()));
            }
        ).def_property("physics_list",
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Simulation& self) -> std::string {
                return std::string(self.physics_list);
            },
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Simulation& self, const std::string& s) {
                strncpy(self.physics_list, s.c_str(), std::min(sizeof(self.physics_list), s.length()));
            }
        )
		.def(py::pickle([](FiledTypes::V1::RadiationFieldMetadataHeader::Simulation& self) {
		    return MetadataHeaderSimulationPickleTuple(
			    self.primary_particle_count,
			    std::string(self.geometry),
			    std::string(self.physics_list),
			    std::tuple(
				    self.tube.max_energy_eV,
				    std::tuple(
					    self.tube.radiation_direction.x,
					    self.tube.radiation_direction.y,
					    self.tube.radiation_direction.z
				    ),
				    std::tuple(
					    self.tube.radiation_origin.x,
					    self.tube.radiation_origin.y,
					    self.tube.radiation_origin.z
				    ),
				    std::string(self.tube.tube_id)
			    )
		    );
		},
			[](const MetadataHeaderSimulationPickleTuple& t) {
                FiledTypes::V1::RadiationFieldMetadataHeader::Simulation simulation;
				simulation.primary_particle_count = std::get<0>(t);
				strncpy(simulation.geometry, std::get<1>(t).c_str(), sizeof(simulation.geometry));
				strncpy(simulation.physics_list, std::get<2>(t).c_str(), sizeof(simulation.physics_list));
				auto tube = std::get<3>(t);
				simulation.tube.max_energy_eV = std::get<0>(tube);
				simulation.tube.radiation_direction.x = std::get<0>(std::get<1>(tube));
				simulation.tube.radiation_direction.y = std::get<1>(std::get<1>(tube));
				simulation.tube.radiation_direction.z = std::get<2>(std::get<1>(tube));
				simulation.tube.radiation_origin.x = std::get<0>(std::get<2>(tube));
				simulation.tube.radiation_origin.y = std::get<1>(std::get<2>(tube));
				simulation.tube.radiation_origin.z = std::get<2>(std::get<2>(tube));
				strncpy(simulation.tube.tube_id, std::get<3>(tube).c_str(), sizeof(simulation.tube.tube_id));
				return simulation;
		}))
        .def_readwrite("tube", &FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::tube);

    py::class_<FiledTypes::V1::RadiationFieldMetadataHeader::Software>(m, "RadiationFieldSoftwareMetadataV1")
		.def(py::init([](const std::string& name, const std::string& version, const std::string& repository, const std::string& commit, const std::string& doi = "") { return FiledTypes::V1::RadiationFieldMetadataHeader::Software(name, version, repository, commit, doi); }), py::arg("name"), py::arg("version"), py::arg("repository"), py::arg("commit"), py::arg("doi") = std::string(""))
        .def_property("name",
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Software& self) -> std::string {
                return std::string(self.name);
            },
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Software& self, const std::string& s) {
                strncpy(self.name, s.c_str(), std::min(sizeof(self.name), s.length()));
            }
        )
        .def_property("version",
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Software& self) -> std::string {
                return std::string(self.version);
            },
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Software& self, const std::string& s) {
                strncpy(self.version, s.c_str(), std::min(sizeof(self.version), s.length()));
            }
        )
        .def_property("repository",
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Software& self) -> std::string {
                return std::string(self.repository);
            },
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Software& self, const std::string& s) {
                strncpy(self.repository, s.c_str(), std::min(sizeof(self.repository), s.length()));
            }
        )
        .def_property("commit",
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Software& self) -> std::string {
                return std::string(self.commit);
            },
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Software& self, const std::string& s) {
                strncpy(self.commit, s.c_str(), std::min(sizeof(self.commit), s.length()));
            }
        )
		.def(py::pickle([](FiledTypes::V1::RadiationFieldMetadataHeader::Software& self) {
		    return MetadataHeaderSoftwarePickleTuple(
			    std::tuple(
				    std::string(self.name),
				    std::string(self.version),
				    std::string(self.repository),
				    std::string(self.commit),
				    std::string(self.doi)
			    )
		    );
		},
			[](const MetadataHeaderSoftwarePickleTuple& t) {
                FiledTypes::V1::RadiationFieldMetadataHeader::Software software;
				strncpy(software.name, std::get<0>(t).c_str(), std::min(sizeof(software.name), std::get<0>(t).length()));
				strncpy(software.version, std::get<1>(t).c_str(), std::min(sizeof(software.version), std::get<1>(t).length()));
				strncpy(software.repository, std::get<2>(t).c_str(), std::min(sizeof(software.repository), std::get<2>(t).length()));
				strncpy(software.commit, std::get<3>(t).c_str(), std::min(sizeof(software.commit), std::get<3>(t).length()));
				strncpy(software.doi, std::get<4>(t).c_str(), std::min(sizeof(software.doi), std::get<4>(t).length()));
				return software;
	    }))
        .def_property("doi",
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Software& self) -> std::string {
                return std::string(self.doi);
            },
            [](FiledTypes::V1::RadiationFieldMetadataHeader::Software& self, const std::string& s) {
                strncpy(self.doi, s.c_str(), std::min(sizeof(self.doi), s.length()));
            }
        );

    py::class_<FiledTypes::V1::RadiationFieldMetadataHeader>(m, "RadiationFieldMetadataHeaderV1")
		.def(py::init<FiledTypes::V1::RadiationFieldMetadataHeader::Simulation, FiledTypes::V1::RadiationFieldMetadataHeader::Software>(), py::arg("simulation"), py::arg("software"))
		.def_readwrite("simulation", &FiledTypes::V1::RadiationFieldMetadataHeader::simulation)
        .def_readwrite("software", &FiledTypes::V1::RadiationFieldMetadataHeader::software)
        .def(py::pickle([](FiledTypes::V1::RadiationFieldMetadataHeader& header) {
		    return MetadataHeaderPickleTuple(
				std::tuple(
                    header.simulation.primary_particle_count,
                    std::string(header.simulation.geometry),
                    std::string(header.simulation.physics_list),
                    std::tuple(
                        header.simulation.tube.max_energy_eV,
                        std::tuple(
                            header.simulation.tube.radiation_direction.x,
                            header.simulation.tube.radiation_direction.y,
                            header.simulation.tube.radiation_direction.z
                        ),
                        std::tuple(
                            header.simulation.tube.radiation_origin.x,
                            header.simulation.tube.radiation_origin.y,
                            header.simulation.tube.radiation_origin.z
                        ),
						std::string(header.simulation.tube.tube_id)
                    )
                ),
                std::tuple(
                    std::string(header.software.name),
                    std::string(header.software.version),
                    std::string(header.software.repository),
                    std::string(header.software.commit),
                    std::string(header.software.doi)
                )
            );
        },
		[](const MetadataHeaderPickleTuple& t) {
            FiledTypes::V1::RadiationFieldMetadataHeader header;
		    auto simulation = std::get<0>(t);
		    auto software = std::get<1>(t);
		    header.simulation.primary_particle_count = std::get<0>(simulation);
		    strncpy(header.simulation.geometry, std::get<1>(simulation).c_str(), sizeof(header.simulation.geometry));
		    strncpy(header.simulation.physics_list, std::get<2>(simulation).c_str(), sizeof(header.simulation.physics_list));
            auto tube = std::get<3>(simulation);
		    header.simulation.tube.max_energy_eV = std::get<0>(tube);
		    header.simulation.tube.radiation_direction.x = std::get<0>(std::get<1>(tube));
			header.simulation.tube.radiation_direction.y = std::get<1>(std::get<1>(tube));
			header.simulation.tube.radiation_direction.z = std::get<2>(std::get<1>(tube));
			header.simulation.tube.radiation_origin.x = std::get<0>(std::get<2>(tube));
			header.simulation.tube.radiation_origin.y = std::get<1>(std::get<2>(tube));
			header.simulation.tube.radiation_origin.z = std::get<2>(std::get<2>(tube));
		    strncpy(header.simulation.tube.tube_id, std::get<3>(tube).c_str(), sizeof(header.simulation.tube.tube_id));
		    strncpy(header.software.name, std::get<0>(software).c_str(), sizeof(header.software.name));
		    strncpy(header.software.version, std::get<1>(software).c_str(), sizeof(header.software.version));
		    strncpy(header.software.repository, std::get<2>(software).c_str(), sizeof(header.software.repository));
		    strncpy(header.software.commit, std::get<3>(software).c_str(), sizeof(header.software.commit));
		    strncpy(header.software.doi, std::get<4>(software).c_str(), sizeof(header.software.doi));
            return header;
	    }));

    py::class_<IVoxel, std::shared_ptr<IVoxel>>(m, "Voxel")
		.def("__repr__",
			[](const IVoxel& a) {
				return "<RadFiled3D.IVoxel>";
			});

    py::class_<Storage::RadiationFieldMetadata, std::shared_ptr<Storage::RadiationFieldMetadata>>(m, "RadiationFieldMetadata");

    py::class_<Storage::V1::RadiationFieldMetadata, std::shared_ptr<Storage::V1::RadiationFieldMetadata>, Storage::RadiationFieldMetadata>(m, "RadiationFieldMetadataV1")
        .def(py::init<Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation, Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software>(), py::arg("simulation"), py::arg("software"))
        .def("get_header", &Storage::V1::RadiationFieldMetadata::get_header)
        .def("set_header", &Storage::V1::RadiationFieldMetadata::set_header)
        .def("get_dynamic_metadata", [](Storage::V1::RadiationFieldMetadata& self, const std::string& key) {
            IVoxel* voxel = self.get_dynamic_metadata().at(key);
            return VOXEL_REFERENCE(voxel);
        }, py::arg("key"), py::return_value_policy::reference)
        .def("get_dynamic_metadata_keys", &Storage::V1::RadiationFieldMetadata::get_dynamic_metadata_keys)
        .def("add_dynamic_histogram_metadata", [](Storage::V1::RadiationFieldMetadata& self, const std::string& key, size_t bins, float bin_width) {
		    RadFiled3D::OwningHistogramVoxel hist(bins, bin_width);
            self.set_dynamic_metadata<RadFiled3D::HistogramVoxel>(key, hist);
			IVoxel* voxel = self.get_dynamic_metadata().at(key);
            return VOXEL_REFERENCE(voxel);
		}, py::arg("key"), py::arg("bins"), py::arg("bin_width"), py::return_value_policy::reference)
        .def("add_dynamic_metadata", [](Storage::V1::RadiationFieldMetadata& self, const std::string& key, Typing::DType dtype) {
            switch (dtype) {
		    case Typing::DType::Float:
		        self.add_dynamic_metadata<float>(key, 0.f);
                break;
			case Typing::DType::UInt32:
				self.add_dynamic_metadata<uint32_t>(key, 0u);
				break;
            case Typing::DType::Int:
				self.add_dynamic_metadata<int>(key, 0);
				break;
			case Typing::DType::UInt64:
				self.add_dynamic_metadata<uint64_t>(key, 0ull);
				break;
            case Typing::DType::Char:
				self.add_dynamic_metadata<char>(key, 0);
                break;
            case Typing::DType::Double:
				self.add_dynamic_metadata<double>(key, 0.0);
				break;
			case Typing::DType::Vec2:
				self.add_dynamic_metadata<glm::vec2>(key, glm::vec2(0.f, 0.f));
                break;
			case Typing::DType::Vec3:
				self.add_dynamic_metadata<glm::vec3>(key, glm::vec3(0.f, 0.f, 0.f));
                break;
			case Typing::DType::Vec4:
				self.add_dynamic_metadata<glm::vec4>(key, glm::vec4(0.f, 0.f, 0.f, 0.f));
                break;
            case Typing::DType::Hist:
				throw py::value_error("Histograms are not supported by this method please use the explicit histogram metadata method.");
            }

            IVoxel* voxel = self.get_dynamic_metadata().at(key);
            return VOXEL_REFERENCE(voxel);
        }, py::arg("key"), py::arg("dtype"), py::return_value_policy::reference);

    // TODO: SWITCH TO USING DECLARE_SCALAR_VOXEL(...) makro

    DECLARE_SCALAR_VOXEL(m, float, "Float32Voxel", IVoxel);
    DECLARE_OWNING_SCALAR_VOXEL(m, float, "OwningFloat32Voxel", ScalarVoxel<float>);

#if defined(__x86_64__) || defined(_M_X64)
	DECLARE_SCALAR_VOXEL(m, uint64_t, "UInt64Voxel", IVoxel);
	DECLARE_OWNING_SCALAR_VOXEL(m, uint64_t, "OwningUInt64Voxel", ScalarVoxel<uint64_t>);
#endif

	DECLARE_SCALAR_VOXEL(m, uint32_t, "UInt32Voxel", IVoxel);
	DECLARE_OWNING_SCALAR_VOXEL(m, uint32_t, "OwningUInt32Voxel", ScalarVoxel<uint32_t>);

	DECLARE_SCALAR_VOXEL(m, char, "ByteVoxel", IVoxel);
	DECLARE_OWNING_SCALAR_VOXEL(m, char, "OwningByteVoxel", ScalarVoxel<char>);

	DECLARE_SCALAR_VOXEL(m, int, "Int32Voxel", IVoxel);
	DECLARE_OWNING_SCALAR_VOXEL(m, int, "OwningInt32Voxel", ScalarVoxel<int>);

#if defined(__x86_64__) || defined(_M_X64)
	DECLARE_SCALAR_VOXEL(m, int64_t, "Int64Voxel", IVoxel);
	DECLARE_OWNING_SCALAR_VOXEL(m, int64_t, "OwningInt64Voxel", ScalarVoxel<int64_t>);
#endif

    py::class_<ScalarVoxel<glm::vec2>, std::shared_ptr<ScalarVoxel<glm::vec2>>, IVoxel>(m, "Vec2Voxel")
        .def("get_data", &ScalarVoxel<glm::vec2>::get_data, py::return_value_policy::reference)
        .def("set_data", [](ScalarVoxel<glm::vec2>& v, const glm::vec2& value) {
            v = value;
        })
        .def(py::self == py::self)
        .def(py::self /= py::self)
        .def(py::self *= py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def("__repr__",
            [](const ScalarVoxel<glm::vec2>& a) {
                return "<RadFiled3D.Vec2Voxel (" + std::to_string(a.get_data().x) + ", " + std::to_string(a.get_data().y) + ")>";
            }
        );

	py::class_<OwningScalarVoxel<glm::vec2>, std::shared_ptr<OwningScalarVoxel<glm::vec2>>, ScalarVoxel<glm::vec2>>(m, "OwningVec2Voxel")
		.def("get_data", &OwningScalarVoxel<glm::vec2>::get_data, py::return_value_policy::reference)
		.def("set_data", &OwningScalarVoxel<glm::vec2>::set_data)
		.def(py::self == py::self)
		.def(py::self /= py::self)
		.def(py::self *= py::self)
		.def(py::self += py::self)
		.def(py::self -= py::self)
		.def("__repr__",
			[](const OwningScalarVoxel<glm::vec2>& a) {
				return "<RadFiled3D.OwningVec2Voxel (" + std::to_string(a.get_data().x) + ", " + std::to_string(a.get_data().y) + ")>";
			}
		);

    py::class_<ScalarVoxel<glm::vec3>, std::shared_ptr<ScalarVoxel<glm::vec3>>, IVoxel>(m, "Vec3Voxel")
        .def("get_data", &ScalarVoxel<glm::vec3>::get_data, py::return_value_policy::reference)
        .def("set_data", [](ScalarVoxel<glm::vec3>& v, const glm::vec3& value) {
            v = value;
        })
        .def(py::self == py::self)
        .def(py::self /= py::self)
        .def(py::self *= py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def("__repr__",
            [](const ScalarVoxel<glm::vec3>& a) {
                return "<RadFiled3D.Vec3Voxel (" + std::to_string(a.get_data().x) + ", " + std::to_string(a.get_data().y) + ", " + std::to_string(a.get_data().z) + ")>";
            }
        );

	py::class_<OwningScalarVoxel<glm::vec3>, std::shared_ptr<OwningScalarVoxel<glm::vec3>>, ScalarVoxel<glm::vec3>>(m, "OwningVec3Voxel")
		.def("get_data", &OwningScalarVoxel<glm::vec3>::get_data, py::return_value_policy::reference)
		.def("set_data", &OwningScalarVoxel<glm::vec3>::set_data)
		.def(py::self == py::self)
		.def(py::self /= py::self)
		.def(py::self *= py::self)
		.def(py::self += py::self)
		.def(py::self -= py::self)
		.def("__repr__",
			[](const OwningScalarVoxel<glm::vec3>& a) {
				return "<RadFiled3D.OwningVec3Voxel (" + std::to_string(a.get_data().x) + ", " + std::to_string(a.get_data().y) + ", " + std::to_string(a.get_data().z) + ")>";
			}
		);

    py::class_<ScalarVoxel<glm::vec4>, std::shared_ptr<ScalarVoxel<glm::vec4>>, IVoxel>(m, "Vec4Voxel")
        .def("get_data", &ScalarVoxel<glm::vec4>::get_data, py::return_value_policy::reference)
        .def("set_data", [](ScalarVoxel<glm::vec4>& v, const glm::vec4& value) {
            v = value;
        })
        .def(py::self == py::self)
        .def(py::self /= py::self)
        .def(py::self *= py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def("__repr__",
            [](const ScalarVoxel<glm::vec4>& a) {
                return "<RadFiled3D.Vec4Voxel (" + std::to_string(a.get_data().x) + ", " + std::to_string(a.get_data().y) + ", " + std::to_string(a.get_data().z) + ", " + std::to_string(a.get_data().w) + ")>";
            }
        );

	py::class_<OwningScalarVoxel<glm::vec4>, std::shared_ptr<OwningScalarVoxel<glm::vec4>>, ScalarVoxel<glm::vec4>>(m, "OwningVec4Voxel")
		.def("get_data", &OwningScalarVoxel<glm::vec4>::get_data, py::return_value_policy::reference)
		.def("set_data", &OwningScalarVoxel<glm::vec4>::set_data)
		.def(py::self == py::self)
		.def(py::self /= py::self)
		.def(py::self *= py::self)
		.def(py::self += py::self)
		.def(py::self -= py::self)
		.def("__repr__",
			[](const OwningScalarVoxel<glm::vec4>& a) {
				return "<RadFiled3D.OwningVec4Voxel (" + std::to_string(a.get_data().x) + ", " + std::to_string(a.get_data().y) + ", " + std::to_string(a.get_data().z) + ", " + std::to_string(a.get_data().w) + ")>";
			}
		);

    py::class_<HistogramVoxel, std::shared_ptr<HistogramVoxel>, IVoxel>(m, "HistogramVoxel")
        .def("get_histogram_bin_width", &HistogramVoxel::get_histogram_bin_width)
        .def("get_bins", &HistogramVoxel::get_bins)
        .def("get_histogram", [](const HistogramVoxel& a) {
            auto histogram = a.get_histogram();
            py::capsule cap(histogram.data(), [](void* data) { /* No deletion */ });
            return py::array_t<float>(
                { static_cast<size_t>(histogram.size()) },  // shape
                { sizeof(float) },  // strides
                histogram.data(),
                cap
            );
        }, py::return_value_policy::reference)
        .def("get_data", [](const HistogramVoxel& a) {
            auto histogram = a.get_histogram();
            py::capsule cap(histogram.data(), [](void* data) { /* No deletion */ });
            return py::array_t<float>(
                { static_cast<size_t>(histogram.size()) },  // shape
                { sizeof(float) },  // strides
                histogram.data(),
                cap
            );
        }, py::return_value_policy::reference)
        .def("add_value", &HistogramVoxel::add_value)
        .def("normalize", &HistogramVoxel::normalize)
        .def(py::self == py::self)
        .def("__repr__",
            [](const HistogramVoxel& a) {
                const size_t bins = a.get_bins();
                const float bin_width = a.get_histogram_bin_width();
                return "<RadFiled3D.HistogramVoxel (" + std::to_string(bins) + "bins @ " + std::to_string(bin_width) + " width" + ")>";
            }
        );

	py::class_<OwningHistogramVoxel, std::shared_ptr<OwningHistogramVoxel>, HistogramVoxel>(m, "OwningHistogramVoxel")
		.def("get_histogram_bin_width", &OwningHistogramVoxel::get_histogram_bin_width)
		.def("get_bins", &OwningHistogramVoxel::get_bins)
		.def("get_histogram", [](const OwningHistogramVoxel& a) {
		    auto histogram = a.get_histogram();
		    py::capsule cap(histogram.data(), [](void* data) { /* No deletion */ });
		    return py::array_t<float>(
			    { static_cast<size_t>(histogram.size()) },  // shape
			    { sizeof(float) },  // strides
			    histogram.data(),
			    cap
		    );
		}, py::return_value_policy::reference)
        .def("get_data", [](const OwningHistogramVoxel& a) {
            auto histogram = a.get_histogram();
            py::capsule cap(histogram.data(), [](void* data) { /* No deletion */ });
            return py::array_t<float>(
                { static_cast<size_t>(histogram.size()) },  // shape
                { sizeof(float) },  // strides
                histogram.data(),
                cap
            );
        }, py::return_value_policy::reference)
		.def("add_value", &OwningHistogramVoxel::add_value)
		.def("normalize", &OwningHistogramVoxel::normalize)
		.def(py::self == py::self)
		.def("__repr__",
			[](const OwningHistogramVoxel& a) {
				const size_t bins = a.get_bins();
				const float bin_width = a.get_histogram_bin_width();
				return "<RadFiled3D.OwningHistogramVoxel (" + std::to_string(bins) + "bins @ " + std::to_string(bin_width) + " width" + ")>";
			}
		);

    py::enum_<GridTracerAlgorithm>(m, "GridTracerAlgorithm")
        .value("SAMPLING", GridTracerAlgorithm::SAMPLING)
		.value("BRESENHAM", GridTracerAlgorithm::BRESENHAM)
        .value("LINETRACING", GridTracerAlgorithm::LINETRACING);

    py::enum_<Typing::DType>(m, "DType")
        .value("FLOAT32", Typing::DType::Float)
        .value("FLOAT64", Typing::DType::Double)
        .value("INT32", Typing::DType::Int)
        .value("BYTE", Typing::DType::Char)
        .value("VEC2", Typing::DType::Vec2)
        .value("VEC3", Typing::DType::Vec3)
        .value("VEC4", Typing::DType::Vec4)
        .value("HISTOGRAM", Typing::DType::Hist)
        .value("UINT64", Typing::DType::UInt64)
        .value("UINT32", Typing::DType::UInt32);

    py::enum_<FieldJoinMode>(m, "FieldJoinMode")
        .value("IDENTITY", FieldJoinMode::Identity)
        .value("ADD", FieldJoinMode::Add)
        .value("MEAN", FieldJoinMode::Mean)
        .value("SUBTRACT", FieldJoinMode::Subtract)
        .value("DIVIDE", FieldJoinMode::Divide)
        .value("MULTIPLY", FieldJoinMode::Multiply)
        .value("ADD_WEIGHTED", FieldJoinMode::AddWeighted);

    py::enum_<FieldType>(m, "FieldType")
        .value("CARTESIAN", FieldType::Cartesian)
        .value("POLAR", FieldType::Polar);

    py::enum_<FieldJoinCheckMode>(m, "FieldJoinCheckMode")
        .value("STRICT", FieldJoinCheckMode::Strict)
        .value("METADATA_SIMULATION_SIMILAR", FieldJoinCheckMode::MetadataSimulationSimilar)
        .value("METADATA_SOFTWARE_EQUAL", FieldJoinCheckMode::MetadataSoftwareEqual)
        .value("METADATA_SOFTWARE_SIMILAR", FieldJoinCheckMode::MetadataSoftwareSimilar)
        .value("FIELD_STRUCTURE_ONLY", FieldJoinCheckMode::FieldStructureOnly)
        .value("FIELD_UNITS_ONLY", FieldJoinCheckMode::FieldUnitsOnly)
        .value("NO_CHECKS", FieldJoinCheckMode::NoChecks);

    py::class_<VoxelBuffer, std::shared_ptr<VoxelBuffer>>(m, "VoxelBuffer")
        .def("get_voxel_count", &VoxelBuffer::get_voxel_count)
        .def("get_layers", &VoxelBuffer::get_layers)
		.def("has_layer", &VoxelBuffer::has_layer)
        .def("get_layer_unit", &VoxelBuffer::get_layer_unit)
        .def("get_statistical_error", &VoxelBuffer::get_statistical_error)
		.def("set_statistical_error", &VoxelBuffer::set_statistical_error)
        .def("get_layer_voxel_type", [](VoxelBuffer& self, const std::string& layer_name) {
            return self.get_voxel_flat<IVoxel>(layer_name, 0).get_type();
            })
        .def("add_layer", [](VoxelBuffer& self, const std::string& name, const std::string& unit, Typing::DType dtype) {
            switch (dtype) {
                case Typing::DType::Float:
                    self.add_layer<float>(name, 0.f, unit);
                    break;
                case Typing::DType::Double:
                    self.add_layer<double>(name, 0.0, unit);
                    break;
                case Typing::DType::Int:
                    self.add_layer<int>(name, 0, unit);
                    break;
                case Typing::DType::Char:
                    self.add_layer<char>(name, 0, unit);
                    break;
                case Typing::DType::Vec3:
                    self.add_layer<glm::vec3>(name, glm::vec3(0.f), unit);
                    break;
                case Typing::DType::Vec2:
                    self.add_layer<glm::vec2>(name, glm::vec2(0.f), unit);
                    break;
                case Typing::DType::Vec4:
                    self.add_layer<glm::vec4>(name, glm::vec4(0.f), unit);
                    break;
                case Typing::DType::UInt64:
                    self.add_layer<uint64_t>(name, 0, unit);
					break;
				case Typing::DType::UInt32:
					self.add_layer<unsigned long>(name, 0, unit);
					break;
                default:
                    throw std::runtime_error("Unsupported voxel type: " + std::to_string(static_cast<int>(dtype)));
            }
            }, py::arg("name"), py::arg("unit"), py::arg("dtype"))
            .def("add_histogram_layer", [](VoxelBuffer& self, const std::string& name, size_t bins, float bin_width, const std::string& unit) {
                self.add_custom_layer<HistogramVoxel>(name, HistogramVoxel(bins, bin_width, nullptr), 0.f, unit);
            }, py::arg("name"), py::arg("bins"), py::arg("bin_width"), py::arg("unit"));

        py::class_<VoxelGridBuffer, std::shared_ptr<VoxelGridBuffer>, VoxelBuffer>(m, "VoxelGridBuffer")
            .def("get_voxel_counts", &VoxelGridBuffer::get_voxel_counts)
            .def("get_voxel_dimensions", &VoxelGridBuffer::get_voxel_dimensions)
			.def("get_voxel_idx_by_coord", &VoxelGridBuffer::get_voxel_idx_by_coord)
			.def("get_voxel_idx", &VoxelGridBuffer::get_voxel_idx, py::arg("x"), py::arg("y"), py::arg("z"))
            .def("get_voxel_flat", [](VoxelGridBuffer& self, const std::string& layer_name, size_t idx) {
                const Typing::DType type = Typing::Helper::get_dtype(self.get_voxel_flat<IVoxel>(layer_name, 0).get_type());
                switch (type) {
                    case Typing::DType::Float:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<float>>(layer_name, idx));
                    case Typing::DType::Double:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<double>>(layer_name, idx));
                    case Typing::DType::Int:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<int>>(layer_name, idx));
                    case Typing::DType::Char:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<char>>(layer_name, idx));
                    case Typing::DType::Vec2:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<glm::vec2>>(layer_name, idx));
                    case Typing::DType::Vec3:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<glm::vec3>>(layer_name, idx));
                    case Typing::DType::Vec4:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<glm::vec4>>(layer_name, idx));
                    case Typing::DType::Hist:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<HistogramVoxel>(layer_name, idx));
                    case Typing::DType::UInt64:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<uint64_t>>(layer_name, idx));
                    case Typing::DType::UInt32:
						return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<unsigned long>>(layer_name, idx));
                    default:
                        throw std::runtime_error("Unsupported voxel type: " + std::to_string(static_cast<int>(type)));
                }
            }, py::arg("layer_name"), py::arg("idx"), py::return_value_policy::reference)
            .def("get_voxel", [](VoxelGridBuffer& self, const std::string& layer_name, size_t x, size_t y, size_t z) {
                const Typing::DType type = Typing::Helper::get_dtype(self.get_voxel_flat<IVoxel>(layer_name, 0).get_type());
                switch (type) {
                    case Typing::DType::Float:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<float>>(layer_name, x, y, z));
                    case Typing::DType::Double:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<double>>(layer_name, x, y, z));
                    case Typing::DType::Int:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<int>>(layer_name, x, y, z));
                    case Typing::DType::Char:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<char>>(layer_name, x, y, z));
                    case Typing::DType::Vec2:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<glm::vec2>>(layer_name, x, y, z));
                    case Typing::DType::Vec3:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<glm::vec3>>(layer_name, x, y, z));
                    case Typing::DType::Vec4:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<glm::vec4>>(layer_name, x, y, z));
                    case Typing::DType::Hist:
                        return VOXEL_REFERENCE(&self.get_voxel<HistogramVoxel>(layer_name, x, y, z));
                    case Typing::DType::UInt64:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<uint64_t>>(layer_name, x, y, z));
                    case Typing::DType::UInt32:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<unsigned long>>(layer_name, x, y, z));
                    default:
                        throw std::runtime_error("Unsupported voxel type: " + std::to_string(static_cast<int>(type)));
                }
            }, py::return_value_policy::reference)
            .def("get_voxel_by_coord", [](VoxelGridBuffer& self, const std::string& layer_name, float x, float y, float z) {
                const Typing::DType type = Typing::Helper::get_dtype(self.get_voxel_flat<IVoxel>(layer_name, 0).get_type());
                switch (type) {
                    case Typing::DType::Float:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<float>>(layer_name, x, y, z));
                    case Typing::DType::Double:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<double>>(layer_name, x, y, z));
                    case Typing::DType::Int:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<int>>(layer_name, x, y, z));
                    case Typing::DType::Char:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<char>>(layer_name, x, y, z));
                    case Typing::DType::Vec2:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<glm::vec2>>(layer_name, x, y, z));
                    case Typing::DType::Vec3:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<glm::vec3>>(layer_name, x, y, z));
                    case Typing::DType::Vec4:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<glm::vec4>>(layer_name, x, y, z));
                    case Typing::DType::Hist:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<HistogramVoxel>(layer_name, x, y, z));
                    case Typing::DType::UInt64:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<uint64_t>>(layer_name, x, y, z));
                    case Typing::DType::UInt32:
						return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<unsigned long>>(layer_name, x, y, z));
                    default:
                        throw std::runtime_error("Unsupported voxel type: " + std::to_string(static_cast<int>(type)));
                }
            }, py::return_value_policy::reference)
            .def("get_layer_as_ndarray", [](std::shared_ptr<VoxelGridBuffer>& self, const std::string& layer) {
                try {
                    const auto& layer_info = self->get_voxel_flat<IVoxel>(layer, 0);
                    const Typing::DType type = Typing::Helper::get_dtype(layer_info.get_type());

                    switch (type) {
                        case Typing::DType::Float:
							return create_py_array_as<float>(self->get_layer<float>(layer), self->get_voxel_counts(), self);
                        case Typing::DType::Double:
							return create_py_array_as<double>(self->get_layer<double>(layer), self->get_voxel_counts(), self);
                        case Typing::DType::Int:
							return create_py_array_as<int>(self->get_layer<int>(layer), self->get_voxel_counts(), self);
                        case Typing::DType::Char:
							return create_py_array_as<char>(self->get_layer<char>(layer), self->get_voxel_counts(), self);
                        case Typing::DType::UInt64:
							return create_py_array_as<uint64_t>(self->get_layer<uint64_t>(layer), self->get_voxel_counts(), self);
                        case Typing::DType::UInt32:
                            return create_py_array_as<unsigned long>(self->get_layer<unsigned long>(layer), self->get_voxel_counts(), self);
                    }

                    const size_t element_size = layer_info.get_bytes();
                    const float* data = self->get_layer<float>(layer);

                    return create_py_array_generic<float>(data, self->get_voxel_counts(), element_size, self);
                }
				catch (const std::exception& e) {
					throw std::runtime_error("Failed to get layer as ndarray: " + std::string(e.what()));
				}
            });

            py::class_<VoxelLayer, std::shared_ptr<VoxelLayer>>(m, "VoxelLayer")
                .def("get_voxel_flat", [](VoxelLayer& self, size_t idx) {
                    const Typing::DType type = Typing::Helper::get_dtype(self.get_voxel_flat<IVoxel>(0).get_type());
                    switch (type) {
                    case Typing::DType::Float:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<float>>(idx));
                    case Typing::DType::Double:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<double>>(idx));
                    case Typing::DType::Int:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<int>>(idx));
                    case Typing::DType::Char:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<char>>(idx));
                    case Typing::DType::Vec2:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<glm::vec2>>(idx));
                    case Typing::DType::Vec3:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<glm::vec3>>(idx));
                    case Typing::DType::Vec4:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<glm::vec4>>(idx));
                    case Typing::DType::Hist:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<HistogramVoxel>(idx));
                    case Typing::DType::UInt64:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<uint64_t>>(idx));
                    case Typing::DType::UInt32:
                        return VOXEL_REFERENCE(&self.get_voxel_flat<ScalarVoxel<unsigned long>>(idx));
                    default:
                        throw std::runtime_error("Unsupported voxel type: " + std::to_string(static_cast<int>(type)));
                    }
                }, py::return_value_policy::reference)
                .def("get_unit", &VoxelLayer::get_unit)
                .def("get_statistical_error", &VoxelLayer::get_statistical_error)
                .def("get_voxel_count", &VoxelLayer::get_voxel_count);

            py::class_<VoxelGrid, std::shared_ptr<VoxelGrid>>(m, "VoxelGrid")
                .def(py::init([](const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions, std::shared_ptr<VoxelLayer> layer) {
                    return std::make_shared<VoxelGrid>(field_dimensions, voxel_dimensions, layer);
                }), py::arg("field_dimensions"), py::arg("voxel_dimensions"), py::arg("dimensions") = std::shared_ptr<VoxelLayer>(nullptr))
                .def("get_voxel_dimensions", &VoxelGrid::get_voxel_dimensions)
                .def("get_voxel_counts", &VoxelGrid::get_voxel_counts)
                .def("get_voxel_idx", &VoxelGrid::get_voxel_idx, py::arg("x"), py::arg("y"), py::arg("z"))
                .def("get_voxel_idx_by_coord", &VoxelGrid::get_voxel_idx_by_coord, py::arg("x"), py::arg("y"), py::arg("z"))
                .def("get_voxel", [](const VoxelGrid& self, size_t x, size_t y, size_t z) {
                    const Typing::DType type = Typing::Helper::get_dtype(self.get_layer()->get_voxel_flat<IVoxel>(0).get_type());
                    switch (type) {
                    case Typing::DType::Float:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<float>>(x, y, z));
                    case Typing::DType::Double:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<double>>(x, y, z));
                    case Typing::DType::Int:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<int>>(x, y, z));
                    case Typing::DType::Char:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<char>>(x, y, z));
                    case Typing::DType::Vec2:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<glm::vec2>>(x, y, z));
                    case Typing::DType::Vec3:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<glm::vec3>>(x, y, z));
                    case Typing::DType::Vec4:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<glm::vec4>>(x, y, z));
                    case Typing::DType::Hist:
                        return VOXEL_REFERENCE(&self.get_voxel<HistogramVoxel>(x, y, z));
                    case Typing::DType::UInt64:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<uint64_t>>(x, y, z));
                    case Typing::DType::UInt32:
                        return VOXEL_REFERENCE(&self.get_voxel<ScalarVoxel<unsigned long>>(x, y, z));
                    default:
                        throw std::runtime_error("Unsupported voxel type: " + std::to_string(static_cast<int>(type)));
                    }
                }, py::arg("x"), py::arg("y"), py::arg("z"), py::return_value_policy::reference)
                .def("get_voxel_by_coord", [](const VoxelGrid& self, float x, float y, float z) {
                    const Typing::DType type = Typing::Helper::get_dtype(self.get_layer()->get_voxel_flat<IVoxel>(0).get_type());
                    switch (type) {
                    case Typing::DType::Float:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<float>>(x, y, z));
                    case Typing::DType::Double:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<double>>(x, y, z));
                    case Typing::DType::Int:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<int>>(x, y, z));
                    case Typing::DType::Char:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<char>>(x, y, z));
                    case Typing::DType::Vec2:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<glm::vec2>>(x, y, z));
                    case Typing::DType::Vec3:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<glm::vec3>>(x, y, z));
                    case Typing::DType::Vec4:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<glm::vec4>>(x, y, z));
                    case Typing::DType::Hist:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<HistogramVoxel>(x, y, z));
                    case Typing::DType::UInt64:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<uint64_t>>(x, y, z));
                    case Typing::DType::UInt32:
                        return VOXEL_REFERENCE(&self.get_voxel_by_coord<ScalarVoxel<unsigned long>>(x, y, z));
                    default:
                        throw std::runtime_error("Unsupported voxel type: " + std::to_string(static_cast<int>(type)));
                    }
                }, py::arg("x"), py::arg("y"), py::arg("z"), py::return_value_policy::reference)
                .def("get_layer", &VoxelGrid::get_layer)
				.def("__enter__", [](std::shared_ptr<VoxelGrid>& self) { return self; })
                .def("__exit__", [](std::shared_ptr<VoxelGrid>& r, py::object exc_type, py::object exc_value, py::object traceback) {
                    r.reset();
			    })
                .def("get_as_ndarray", [](std::shared_ptr<VoxelGrid>& self) {
				    try {
					    const auto& layer_info = self->get_layer()->get_voxel_flat<IVoxel>(0);
					    const Typing::DType type = Typing::Helper::get_dtype(layer_info.get_type());

					    switch (type) {
					    case Typing::DType::Float:
						    return create_py_array_as<float>((float*)self->get_layer()->get_raw_data(), self->get_voxel_counts(), self);
					    case Typing::DType::Double:
						    return create_py_array_as<double>((double*)self->get_layer()->get_raw_data(), self->get_voxel_counts(), self);
					    case Typing::DType::Int:
						    return create_py_array_as<int>((int*)self->get_layer()->get_raw_data(), self->get_voxel_counts(), self);
					    case Typing::DType::Char:
						    return create_py_array_as<char>((char*)self->get_layer()->get_raw_data(), self->get_voxel_counts(), self);
					    case Typing::DType::UInt64:
						    return create_py_array_as<uint64_t>((uint64_t*)self->get_layer()->get_raw_data(), self->get_voxel_counts(), self);
					    case Typing::DType::UInt32:
						    return create_py_array_as<unsigned long>((unsigned long*)self->get_layer()->get_raw_data(), self->get_voxel_counts(), self);
					    }

					    const size_t element_size = layer_info.get_bytes();
					    const float* data = (float*)self->get_layer()->get_raw_data();

					    return create_py_array_generic<float>(data, self->get_voxel_counts(), element_size, self);
				    }
				    catch (const std::exception& e) {
					    throw std::runtime_error("Failed to get layer as ndarray: " + std::string(e.what()));
				    }
                });

        py::class_<PolarSegments, std::shared_ptr<PolarSegments>>(m, "PolarSegments")
			.def(py::init([](const glm::uvec2& segments_counts, std::shared_ptr<VoxelLayer> layer) {
		        return std::make_shared<PolarSegments>(segments_counts, layer);
			}), py::arg("segments_counts"), py::arg("layer") = std::shared_ptr<VoxelLayer>(nullptr))
			.def("get_segments_count", &PolarSegments::get_segments_count)
			.def("get_segment_idx_by_coord", &PolarSegments::get_segment_idx_by_coord)
			.def("get_segment_idx", &PolarSegments::get_segment_idx)
			.def("get_segment", [](PolarSegments& self, size_t x, size_t y) {
			    const Typing::DType type = Typing::Helper::get_dtype(self.get_layer()->get_voxel_flat<IVoxel>(0).get_type());
			    switch (type) {
			    case Typing::DType::Float:
				    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<float>>(x, y));
			    case Typing::DType::Double:
				    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<double>>(x, y));
			    case Typing::DType::Int:
				    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<int>>(x, y));
			    case Typing::DType::Char:
				    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<char>>(x, y));
			    case Typing::DType::Vec2:
				    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<glm::vec2>>(x, y));
			    case Typing::DType::Vec3:
				    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<glm::vec3>>(x, y));
			    case Typing::DType::Vec4:
				    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<glm::vec4>>(x, y));
			    case Typing::DType::Hist:
				    return VOXEL_REFERENCE(&self.get_segment<HistogramVoxel>(x, y));
			    case Typing::DType::UInt64:
				    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<uint64_t>>(x, y));
			    case Typing::DType::UInt32:
				    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<unsigned long>>(x, y));
			    default:
				    throw std::runtime_error("Unsupported voxel type: " + std::to_string(static_cast<int>(type)));
			    }
			}, py::return_value_policy::reference)
			.def("get_segment_by_coord", [](PolarSegments& self, float phi, float theta) {
			    const Typing::DType type = Typing::Helper::get_dtype(self.get_layer()->get_voxel_flat<IVoxel>(0).get_type());
			    switch (type) {
			    case Typing::DType::Float:
				    return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<float>>(phi, theta));
			    case Typing::DType::Double:
				    return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<double>>(phi, theta));
			    case Typing::DType::Int:
				    return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<int>>(phi, theta));
			    case Typing::DType::Char:
				    return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<char>>(phi, theta));
			    case Typing::DType::Vec2:
				    return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<glm::vec2>>(phi, theta));
			    case Typing::DType::Vec3:
				    return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<glm::vec3>>(phi, theta));
			    case Typing::DType::Vec4:
				    return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<glm::vec4>>(phi, theta));
			    case Typing::DType::Hist:
				    return VOXEL_REFERENCE(&self.get_segment_by_coord<HistogramVoxel>(phi, theta));
			    case Typing::DType::UInt64:
				    return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<uint64_t>>(phi, theta));
			    case Typing::DType::UInt32:
				    return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<unsigned long>>(phi, theta));
			    default:
				    throw std::runtime_error("Unsupported voxel type: " + std::to_string(static_cast<int>(type)));
			    }
			}, py::return_value_policy::reference)
			.def("get_layer", &PolarSegments::get_layer)
			.def("__enter__", [](std::shared_ptr<PolarSegments>& self) { return self; })
			.def("__exit__", [](std::shared_ptr<PolarSegments>& r, py::object exc_type, py::object exc_value, py::object traceback) {
			    r.reset();
		    })
            .def("get_as_ndarray", [](std::shared_ptr<PolarSegments>& self) {
                try {
                    const auto& layer_info = self->get_layer()->get_voxel_flat<IVoxel>(0);
                    const Typing::DType type = Typing::Helper::get_dtype(layer_info.get_type());

                    switch (type) {
                    case Typing::DType::Float:
                        return create_py_array_as<float>((float*)self->get_layer()->get_raw_data(), self->get_segments_count(), self);
                    case Typing::DType::Double:
                        return create_py_array_as<double>((double*)self->get_layer()->get_raw_data(), self->get_segments_count(), self);
                    case Typing::DType::Int:
                        return create_py_array_as<int>((int*)self->get_layer()->get_raw_data(), self->get_segments_count(), self);
                    case Typing::DType::Char:
                        return create_py_array_as<char>((char*)self->get_layer()->get_raw_data(), self->get_segments_count(), self);
                    case Typing::DType::UInt64:
                        return create_py_array_as<uint64_t>((uint64_t*)self->get_layer()->get_raw_data(), self->get_segments_count(), self);
                    case Typing::DType::UInt32:
                        return create_py_array_as<unsigned long>((unsigned long*)self->get_layer()->get_raw_data(), self->get_segments_count(), self);
                    }

                    const size_t element_size = layer_info.get_bytes();
                    const float* data = (float*)self->get_layer()->get_raw_data();

                    return create_py_array_generic<float>(data, self->get_segments_count(), element_size, self);
                }
                catch (const std::exception& e) {
                    throw std::runtime_error("Failed to get layer as ndarray: " + std::string(e.what()));
                }
			});

        py::class_<PolarSegmentsBuffer, std::shared_ptr<PolarSegmentsBuffer>, VoxelBuffer>(m, "PolarSegmentsBuffer")
            .def("get_segments_count", &PolarSegmentsBuffer::get_segments_count)
            .def("get_segment_idx_by_coord", &PolarSegmentsBuffer::get_segment_idx_by_coord)
            .def("get_segment_idx", &PolarSegmentsBuffer::get_segment_idx)
            .def("get_segment_flat", [](const PolarSegmentsBuffer& self, const std::string& layer, size_t idx) {
                const Typing::DType type = Typing::Helper::get_dtype(self.get_segment_flat<IVoxel>(layer, 0).get_type());
                switch (type) {
                    case Typing::DType::Float:
                        return VOXEL_REFERENCE(&self.get_segment_flat<ScalarVoxel<float>>(layer, idx));
                    case Typing::DType::Double:
                        return VOXEL_REFERENCE(&self.get_segment_flat<ScalarVoxel<double>>(layer, idx));
                    case Typing::DType::Int:
                        return VOXEL_REFERENCE(&self.get_segment_flat<ScalarVoxel<int>>(layer, idx));
                    case Typing::DType::Char:
                        return VOXEL_REFERENCE(&self.get_segment_flat<ScalarVoxel<char>>(layer, idx));
                    case Typing::DType::Vec2:
                        return VOXEL_REFERENCE(&self.get_segment_flat<ScalarVoxel<glm::vec2>>(layer, idx));
                    case Typing::DType::Vec3:
                        return VOXEL_REFERENCE(&self.get_segment_flat<ScalarVoxel<glm::vec3>>(layer, idx));
                    case Typing::DType::Vec4:
                        return VOXEL_REFERENCE(&self.get_segment_flat<ScalarVoxel<glm::vec4>>(layer, idx));
                    case Typing::DType::Hist:
                        return VOXEL_REFERENCE(&self.get_segment_flat<HistogramVoxel>(layer, idx));
                    case Typing::DType::UInt64:
                        return VOXEL_REFERENCE(&self.get_segment_flat<ScalarVoxel<uint64_t>>(layer, idx));
                    case Typing::DType::UInt32:
                        return VOXEL_REFERENCE(&self.get_segment_flat<ScalarVoxel<unsigned long>>(layer, idx));
                    default:
                        throw std::runtime_error("Unsupported segment type: " + std::to_string(static_cast<int>(type)));
                }
            }, py::return_value_policy::reference)
            .def("get_segment_by_coord", [](const PolarSegmentsBuffer& self, const std::string& layer, float phi, float theta) {
                const Typing::DType type = Typing::Helper::get_dtype(self.get_segment_flat<IVoxel>(layer, 0).get_type());
                switch (type) {
                    case Typing::DType::Float:
						return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<float>>(layer, phi, theta));
                    case Typing::DType::Double:
                        return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<double>>(layer, phi, theta));
                    case Typing::DType::Int:
						return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<int>>(layer, phi, theta));
                    case Typing::DType::Char:
                        return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<char>>(layer, phi, theta));
                    case Typing::DType::Vec2:
						return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<glm::vec2>>(layer, phi, theta));
                    case Typing::DType::Vec3:
                        return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<glm::vec3>>(layer, phi, theta));
                    case Typing::DType::Vec4:
						return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<glm::vec4>>(layer, phi, theta));
                    case Typing::DType::Hist:
						return VOXEL_REFERENCE(&self.get_segment_by_coord<HistogramVoxel>(layer, phi, theta));
                    case Typing::DType::UInt64:
                        return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<uint64_t>>(layer, phi, theta));
                    case Typing::DType::UInt32:
                        return VOXEL_REFERENCE(&self.get_segment_by_coord<ScalarVoxel<unsigned long>>(layer, phi, theta));
                    default:
                        throw std::runtime_error("Unsupported segment type: " + std::to_string(static_cast<int>(type)));
                }
            }, py::return_value_policy::reference)
            .def("get_segment", [](const PolarSegmentsBuffer& self, const std::string& layer, size_t x, size_t y) {
                const Typing::DType type = Typing::Helper::get_dtype(self.get_voxel_flat<IVoxel>(layer, 0).get_type());
                switch (type) {
                case Typing::DType::Float:
                    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<float>>(layer, x, y));
                case Typing::DType::Double:
                    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<double>>(layer, x, y));
                case Typing::DType::Int:
                    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<int>>(layer, x, y));
                case Typing::DType::Char:
                    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<char>>(layer, x, y));
                case Typing::DType::Vec2:
                    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<glm::vec2>>(layer, x, y));
                case Typing::DType::Vec3:
                    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<glm::vec3>>(layer, x, y));
                case Typing::DType::Vec4:
                    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<glm::vec4>>(layer, x, y));
                case Typing::DType::Hist:
                    return VOXEL_REFERENCE(&self.get_segment<HistogramVoxel>(layer, x, y));
                case Typing::DType::UInt64:
                    return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<uint64_t>>(layer, x, y));
                case Typing::DType::UInt32:
					return VOXEL_REFERENCE(&self.get_segment<ScalarVoxel<unsigned long>>(layer, x, y));
                default:
                    throw std::runtime_error("Unsupported segment type: " + std::to_string(static_cast<int>(type)));
                }
            }, py::return_value_policy::reference)
            .def("get_layer_as_ndarray", [](std::shared_ptr<PolarSegmentsBuffer>& self, const std::string& layer) {
                const auto& layer_info = self->get_voxel_flat<IVoxel>(layer, 0);
                const Typing::DType type = Typing::Helper::get_dtype(layer_info.get_type());
            
                switch (type) {
                    case Typing::DType::Float:
						return create_py_array_as<float>(self->get_layer<float>(layer), self->get_segments_count(), self);
                    case Typing::DType::Double:
						return create_py_array_as<double>(self->get_layer<double>(layer), self->get_segments_count(), self);
                    case Typing::DType::Int:
						return create_py_array_as<int>(self->get_layer<int>(layer), self->get_segments_count(), self);
                    case Typing::DType::Char:
						return create_py_array_as<char>(self->get_layer<char>(layer), self->get_segments_count(), self);
                    case Typing::DType::UInt64:
						return create_py_array_as<uint64_t>(self->get_layer<uint64_t>(layer), self->get_segments_count(), self);
                    case Typing::DType::UInt32:
                        return create_py_array_as<unsigned long>(self->get_layer<unsigned long>(layer), self->get_segments_count(), self);
                }

				if (type == Typing::DType::Hist) {
					const size_t element_size = layer_info.get_bytes();
					const float* data = self->get_layer<float>(layer);

					return create_py_array_generic<float>(data, self->get_segments_count(), element_size, self);
                }
                else {
					throw std::runtime_error("Unsupported voxel type: " + std::to_string(static_cast<int>(type)));
				}

                const size_t element_size = layer_info.get_bytes();
                const float* data = self->get_layer<float>(layer);

				return create_py_array_generic<float>(data, self->get_segments_count(), element_size, self);
            });

        py::class_<IRadiationField, std::shared_ptr<IRadiationField>>(m, "RadiationField")
            .def("get_typename", &IRadiationField::get_typename)
            .def("get_channels", &IRadiationField::get_channels)
            .def("get_channel_names", &IRadiationField::get_channel_names)
			.def("has_channel", &IRadiationField::has_channel)
			.def("get_channel", &IRadiationField::get_generic_channel)
            .def("add_channel", &IRadiationField::add_channel)
            .def("__exit__",
                [&](std::shared_ptr<IRadiationField>& r,
                    const std::optional<pybind11::type>& exc_type,
                    const std::optional<pybind11::object>& exc_value,
                    const std::optional<pybind11::object>& traceback)
                {
                    r.reset();
                },
                "Exit the runtime context related to this object")
			.def("__enter__", [](std::shared_ptr<IRadiationField>& r) { return r; })
            .def("copy", &IRadiationField::copy);

        py::class_<CartesianRadiationField, std::shared_ptr<CartesianRadiationField>, IRadiationField>(m, "CartesianRadiationField")
            .def(py::init<const glm::vec3&, const glm::vec3&>())
            .def("add_channel", [](CartesianRadiationField& self, const std::string& name) {
                return std::static_pointer_cast<VoxelGridBuffer>(self.add_channel(name));
            })
            .def("get_channels", [](CartesianRadiationField& self) {
                auto channels = self.get_channels();
                std::vector<std::pair<std::string, std::shared_ptr<VoxelGridBuffer>>> result;
                for (auto& channel : channels) {
				    result.push_back(std::make_pair(channel.first, std::static_pointer_cast<VoxelGridBuffer>(channel.second)));
			    }
                return result;
             })
            .def("get_channel", [](CartesianRadiationField& self, const std::string& name) {
				return self.get_channel(name);
			})
            .def("get_voxel_dimensions", &CartesianRadiationField::get_voxel_dimensions)
            .def("get_voxel_counts", &CartesianRadiationField::get_voxel_counts)
            .def("get_field_dimensions", &CartesianRadiationField::get_field_dimensions)
            .def("copy", &CartesianRadiationField::copy)
            .def("__exit__",
                [&](std::shared_ptr<CartesianRadiationField>& r,
                    const std::optional<pybind11::type>& exc_type,
                    const std::optional<pybind11::object>& exc_value,
                    const std::optional<pybind11::object>& traceback)
                {
                    r.reset();
                },
                "Exit the runtime context related to this object")
			.def("__enter__", [](std::shared_ptr<CartesianRadiationField>& r) { return r; })
            .def("__repr__",
                [](const CartesianRadiationField& a) {
                    auto field_dim   = a.get_field_dimensions();
                    auto voxel_dim   = a.get_voxel_dimensions();
                    auto voxel_count = a.get_voxel_counts();
                    return "<RadFiled3D.CartesianRadiationField (" + std::to_string(field_dim.x) + " m, " + std::to_string(field_dim.y) + " m, " + std::to_string(field_dim.z) + " m) @ Voxels(" + std::to_string(voxel_dim.x) + " m, " + std::to_string(voxel_dim.y) + " m, " + std::to_string(voxel_dim.z) + " m) x (" + std::to_string(voxel_count.x) + ", " + std::to_string(voxel_count.y) + ", " + std::to_string(voxel_count.z) + ")>";
                }
             );

        py::class_<PolarRadiationField, std::shared_ptr<PolarRadiationField>, IRadiationField>(m, "PolarRadiationField")
            .def(py::init<const glm::uvec2&>())
            .def("get_channels", [](CartesianRadiationField& self) {
                auto channels = self.get_channels();
                std::vector<std::pair<std::string, std::shared_ptr<PolarSegmentsBuffer>>> result;
                for (auto& channel : channels) {
                    result.push_back(std::make_pair(channel.first, std::static_pointer_cast<PolarSegmentsBuffer>(channel.second)));
                }
                return result;
            })
            .def("add_channel", [](PolarRadiationField& self, const std::string& name) {
                return std::static_pointer_cast<PolarSegmentsBuffer>(self.add_channel(name));
            })
            .def("get_channel", [](PolarRadiationField& self, const std::string& name) {
                return self.get_channel(name);
            })
		    .def("get_segments_count", &PolarRadiationField::get_segments_count)
            .def("copy", &PolarRadiationField::copy)
            .def("__exit__",
                [&](std::shared_ptr<PolarRadiationField>& r,
                    const std::optional<pybind11::type>& exc_type,
                    const std::optional<pybind11::object>& exc_value,
                    const std::optional<pybind11::object>& traceback)
                {
                    r.reset();
                },
                "Exit the runtime context related to this object")
			.def("__enter__", [](std::shared_ptr<PolarRadiationField>& r) { return r; })
            .def("__repr__",
                [](const PolarRadiationField& a) {
                    auto segments = a.get_segments_count();
                    return "<RadFiled3D.PolarRadiationField (" + std::to_string(segments.x) + " x " + std::to_string(segments.y) + ")>";
                }
            );

        py::enum_<Storage::StoreVersion>(m, "StoreVersion")
            .value("V1", Storage::StoreVersion::V1);

        py::class_<RadFiled3D::Storage::FieldAccessor, std::shared_ptr<FieldAccessor>>(m, "FieldAccessor")
			.def(py::pickle(    // general fallback for all FieldAccessor types. No explicit testing if the type python is expecting matches the unpickle procedure loaded, but should be fine for future accessors.
                [](const Storage::FieldAccessor& self) {
                    auto data = FieldAccessor::Serialize(&self);
                    return FieldAccessorPickleTuple(self.getFieldType(), data);
                },
                [](const FieldAccessorPickleTuple& t) {
                    FieldType type = std::get<0>(t);
                    if (std::get<1>(t).size() == 0) {
                        throw std::runtime_error("Empty data");
                    }
                    return FieldAccessor::Deserialize(std::get<1>(t));
                }
            ))
            .def("get_field_type", [](const FieldAccessor& self) {
                return self.getFieldType();
            })
            .def("access_field_from_buffer", [](const FieldAccessor& self, const py::bytes& bytes) {
                std::istringstream stream(static_cast<std::string>(bytes));
                return self.accessField(stream);
            })
            .def("access_field", [](const FieldAccessor& self, const std::string& file) {
			    std::ifstream stream(file, std::ios::binary);
                return self.accessField(stream);
            })
			.def_static("get_store_version", [](const py::bytes& bytes) {
                std::istringstream stream(static_cast<std::string>(bytes));
			    return FieldAccessor::getStoreVersion(stream);
			})
            .def("get_voxel_count", [](const FieldAccessor& self) {
                return self.getVoxelCount();
            })
			.def("__repr__", [](const FieldAccessor& a) {
                std::string field_type = "";
				switch (a.getFieldType()) {
				case FieldType::Cartesian:
					field_type = "Cartesian";
					break;
				case FieldType::Polar:
					field_type = "Polar";
					break;
				default:
					field_type = "Unknown";
					break;
				}
			    return std::string("<RadFiled3D.FieldAccessor (") + field_type + std::string(")>");
		    })
            .def("access_voxel_flat", [](const FieldAccessor& self, const std::string& file, const std::string& channel_name, const std::string& layer_name, size_t idx) {
                std::ifstream stream(file, std::ios::binary);
                return encapsulate_voxel(self.accessVoxelRawFlat(stream, channel_name, layer_name, idx));
            })
            .def("access_voxel_flat_from_buffer", [](const FieldAccessor& self, const py::bytes& bytes, const std::string& channel_name, const std::string& layer_name, size_t idx) {
                std::istringstream stream(static_cast<std::string>(bytes));
                return encapsulate_voxel(self.accessVoxelRawFlat(stream, channel_name, layer_name, idx));
            });

        py::class_<Storage::CartesianFieldAccessor, std::shared_ptr<CartesianFieldAccessor>, RadFiled3D::Storage::FieldAccessor>(m, "CartesianFieldAccessor")
			.def(py::init([](const std::shared_ptr<FieldAccessor>& base) { return std::dynamic_pointer_cast<Storage::CartesianFieldAccessor>(base); }))
            .def("access_voxel_flat", [](const Storage::CartesianFieldAccessor& self, const std::string& file, const std::string& channel_name, const std::string& layer_name, size_t idx) {
			    std::ifstream stream(file, std::ios::binary);
			    return encapsulate_voxel(self.accessVoxelRawFlat(stream, channel_name, layer_name, idx));
			})
            .def("get_field_type", [](const CartesianFieldAccessor& self) {
                return self.getFieldType();
            })
            .def("get_voxel_count", [](const CartesianFieldAccessor& self) {
                return self.getVoxelCount();
            })
			.def("access_voxel_flat_from_buffer", [](const Storage::CartesianFieldAccessor& self, const py::bytes& bytes, const std::string& channel_name, const std::string& layer_name, size_t idx) {
			    std::istringstream stream(static_cast<std::string>(bytes));
                return encapsulate_voxel(self.accessVoxelRawFlat(stream, channel_name, layer_name, idx));
			})
            .def("access_field", [](const Storage::CartesianFieldAccessor& self, const std::string& file) {
			    std::ifstream stream(file, std::ios::binary);
			    return self.accessField(stream);
		    })
			.def("access_field_from_buffer", [](const Storage::CartesianFieldAccessor& self, const py::bytes& bytes) {
			    std::istringstream stream(static_cast<std::string>(bytes));
			    return self.accessField(stream);
			})
            .def("access_layer_from_buffer", [](const Storage::CartesianFieldAccessor& self, const py::bytes& bytes, const std::string& channel_name, const std::string& layer_name) {
                std::istringstream stream(static_cast<std::string>(bytes));
			    return self.accessLayer(stream, channel_name, layer_name);
			})
			.def("access_layer", [](const Storage::CartesianFieldAccessor& self, const std::string& file, const std::string& channel_name, const std::string& layer_name) {
			    std::ifstream stream(file, std::ios::binary);
			    return self.accessLayer(stream, channel_name, layer_name);
		    })
            .def("access_layer_across_channels", [](const Storage::CartesianFieldAccessor& self, const std::string& file, const std::string& layer_name) {
			    std::ifstream stream(file, std::ios::binary);
			    return self.accessLayerAcrossChannels(stream, layer_name);
			})
            .def("access_layer_across_channels_from_buffer", [](const Storage::CartesianFieldAccessor& self, const py::bytes& bytes, const std::string& layer_name) {
			    std::istringstream stream(static_cast<std::string>(bytes));
			    return self.accessLayerAcrossChannels(stream, layer_name);
			})
			.def("access_channel", [](const Storage::CartesianFieldAccessor& self, const std::string& file, const std::string& channel_name) {
			    std::ifstream stream(file, std::ios::binary);
			    return self.accessChannel(stream, channel_name);
			})
            .def("access_channel_from_buffer", [](const Storage::CartesianFieldAccessor& self, const py::bytes& bytes, const std::string& channel_name) {
                std::istringstream stream(static_cast<std::string>(bytes));
                return self.accessChannel(stream, channel_name);
            })
			.def("access_voxel", [](const Storage::CartesianFieldAccessor& self, const std::string& file, const std::string& channel_name, const std::string& layer_name, const glm::uvec3& coord) {
			    std::ifstream stream(static_cast<std::string>(file), std::ios::binary);
			    return encapsulate_voxel(self.accessVoxelRaw(stream, channel_name, layer_name, coord));
			})
            .def("access_voxel_from_buffer", [](const Storage::CartesianFieldAccessor& self, const py::bytes& bytes, const std::string& channel_name, const std::string& layer_name, const glm::uvec3& coord) {
                std::istringstream stream(static_cast<std::string>(bytes));
                return encapsulate_voxel(self.accessVoxelRaw(stream, channel_name, layer_name, coord));
            })
			.def("__repr__", [](const Storage::CartesianFieldAccessor& self) {
                auto voxels = self.getVoxelCount();
			    return std::string("<RadFiled3D.CartesianFieldAccessor (voxels: ") + std::to_string(voxels) + std::string(")>");
			})
			.def("access_voxel_by_coord_from_buffer", [](const Storage::CartesianFieldAccessor& self, const py::bytes& bytes, const std::string& channel_name, const std::string& layer_name, const glm::vec3& coord) {
                std::istringstream stream(static_cast<std::string>(bytes));
			    return encapsulate_voxel(self.accessVoxelRawByCoord(stream, channel_name, layer_name, coord));
			})
            .def("access_voxel_by_coord", [](const Storage::CartesianFieldAccessor& self, const std::string& file, const std::string& channel_name, const std::string& layer_name, const glm::vec3& coord) {
			    std::ifstream stream(file, std::ios::binary);
                return encapsulate_voxel(self.accessVoxelRawByCoord(stream, channel_name, layer_name, coord));
            });
        
		py::class_<Storage::V1::CartesianFieldAccessor, std::shared_ptr<Storage::V1::CartesianFieldAccessor>, Storage::CartesianFieldAccessor>(m, "CartesianFieldAccessorV1")
            .def(py::pickle(
                [](const Storage::V1::CartesianFieldAccessor& self) {
                    auto data = FieldAccessor::Serialize(&self);
                    return FieldAccessorPickleTuple(self.getFieldType(), data);
                },
                [](const FieldAccessorPickleTuple& t) {
                    FieldType type = std::get<0>(t);
                    if (type != FieldType::Cartesian) {
                        throw std::runtime_error("Unsupported field type: " + std::to_string(static_cast<int>(type)));
                    }
                    if (std::get<1>(t).size() == 0) {
                        throw std::runtime_error("Empty data");
                    }
                    return std::dynamic_pointer_cast<Storage::V1::CartesianFieldAccessor>(FieldAccessor::Deserialize(std::get<1>(t)));
                }
            ))
            .def(py::init([](const std::shared_ptr<FieldAccessor>& base) { return std::dynamic_pointer_cast<Storage::V1::CartesianFieldAccessor>(base); }))
            .def("get_voxel_count", [](const V1::CartesianFieldAccessor& self) {
			    return self.getVoxelCount();
			})
            .def("get_field_type", [](const V1::CartesianFieldAccessor& self) {
                return self.getFieldType();
            })
			.def("__repr__", [](const V1::CartesianFieldAccessor& self) {
			    auto voxels = self.getVoxelCount();
				return std::string("<RadFiled3D.CartesianFieldAccessorV1 (voxels: ") + std::to_string(voxels) + std::string(")>");
			});

		py::class_<Storage::PolarFieldAccessor, std::shared_ptr<PolarFieldAccessor>, Storage::FieldAccessor>(m, "PolarFieldAccessor")
            .def("get_voxel_count", [](const PolarFieldAccessor& self) {
                return self.getVoxelCount();
            })
            .def("get_field_type", [](const PolarFieldAccessor& self) {
                return self.getFieldType();
            })
            .def("access_layer", [](const PolarFieldAccessor& self, const py::bytes& bytes, const std::string& channel_name, const std::string& layer_name) {
                std::istringstream stream(static_cast<std::string>(bytes));
                return self.accessLayer(stream, channel_name, layer_name);
            })
			.def("access_voxel", [](const PolarFieldAccessor& self, const py::bytes& bytes, const std::string& channel_name, const std::string& layer_name, const glm::uvec2& coord) {
                std::istringstream stream(static_cast<std::string>(bytes));
			    return encapsulate_voxel(self.accessVoxelRaw(stream, channel_name, layer_name, coord));
		    })
			.def("__repr__", [](const PolarFieldAccessor& a) {
			    auto voxels = a.getVoxelCount();
			    return std::string("<RadFiled3D.PolarFieldAccessor (voxels: ") + std::to_string(voxels) + std::string(")>");
		    })
			.def("access_voxel_by_coord", [](const PolarFieldAccessor& self, const py::bytes& bytes, const std::string& channel_name, const std::string& layer_name, const glm::vec2& coord) {
                std::istringstream stream(static_cast<std::string>(bytes));
			    return encapsulate_voxel(self.accessVoxelRawByCoord(stream, channel_name, layer_name, coord));
			});

		py::class_<V1::PolarFieldAccessor, std::shared_ptr<V1::PolarFieldAccessor>, Storage::PolarFieldAccessor>(m, "PolarFieldAccessorV1")
            .def(py::pickle(
                [](const Storage::V1::PolarFieldAccessor& self) {
                    auto data = FieldAccessor::Serialize(&self);
                    return FieldAccessorPickleTuple(self.getFieldType(), data);
                },
                [](const FieldAccessorPickleTuple& t) {
                    FieldType type = std::get<0>(t);
                    if (type != FieldType::Polar) {
                        throw std::runtime_error("Unsupported field type: " + std::to_string(static_cast<int>(type)));
                    }
                    if (std::get<1>(t).size() == 0) {
                        throw std::runtime_error("Empty data");
                    }
                    return std::dynamic_pointer_cast<Storage::V1::PolarFieldAccessor>(FieldAccessor::Deserialize(std::get<1>(t)));
                }
            ))
            .def("get_voxel_count", [](const V1::PolarFieldAccessor& self) {
                return self.getVoxelCount();
            })
            .def("get_field_type", [](const V1::PolarFieldAccessor& self) {
                return self.getFieldType();
            })
			.def("__repr__", [](const V1::PolarFieldAccessor& a) {
			    auto voxels = a.getVoxelCount();
			    return std::string("<RadFiled3D.PolarFieldAccessorV1 (voxels: ") + std::to_string(voxels) + std::string(")>");
			});

        py::class_<Storage::FieldStore>(m, "FieldStore")
            .def_static("init_store_instance", &Storage::FieldStore::init_store_instance)
            .def_static("enable_file_lock_syncronization", &Storage::FieldStore::enable_file_lock_syncronization)
            .def_static("get_store_version", static_cast<Storage::StoreVersion(*)(const std::string&)>(&Storage::FieldStore::get_store_version))
            .def_static("load", static_cast<std::shared_ptr<IRadiationField>(*)(const std::string&)>(&FieldStore::load))
            .def_static("load_from_buffer", [](const std::string& bytes) {
                std::istringstream stream(bytes);
                return FieldStore::load(stream);
            })
            .def_static("load_metadata", static_cast<std::shared_ptr<Storage::RadiationFieldMetadata>(*)(const std::string&)>(&FieldStore::load_metadata))
            .def_static("peek_metadata", static_cast<std::shared_ptr<Storage::RadiationFieldMetadata>(*)(const std::string&)>(&FieldStore::peek_metadata))
            .def_static("load_metadata_from_buffer", [](const std::string& bytes) {
                std::istringstream stream(bytes);
                return FieldStore::load_metadata(stream);
            })
            .def_static("peek_metadata_from_buffer", [](const std::string& bytes) {
                std::istringstream stream(bytes);
                return FieldStore::peek_metadata(stream);
            })
            .def_static("store", &FieldStore::store, py::arg("field"), py::arg("metadata"), py::arg("file"), py::arg("version") = StoreVersion::V1)
            .def_static("join", &FieldStore::join, py::arg("field"), py::arg("metadata"), py::arg("file"), py::arg("join_mode") = FieldJoinMode::Add, py::arg("check_mode") = FieldJoinCheckMode::MetadataSimulationSimilar, py::arg("fallback_version") = StoreVersion::V1)
            .def_static("peek_field_type", &FieldStore::peek_field_type)
            .def_static("construct_field_accessor", [](const std::string& file) {
			    std::ifstream stream(file, std::ios::binary);
                return FieldStore::construct_accessor(stream);
            })
            .def_static("construct_field_accessor_from_buffer", [](const py::bytes& bytes) {
			    std::istringstream stream(static_cast<std::string>(bytes));
                return FieldStore::construct_accessor(stream);
            })
            .def_static("load_single_grid_layer", [](const std::string& file, const std::string& channel_name, const std::string& layer_name) -> std::shared_ptr<VoxelGrid> {
			    std::ifstream buffer(file, std::ios::binary);
                auto accessor = FieldStore::construct_accessor(buffer);

				if (accessor->getFieldType() != RadFiled3D::FieldType::Cartesian) {
					throw std::runtime_error("Field is not of type Cartesian");
				}

				return std::dynamic_pointer_cast<CartesianFieldAccessor>(accessor)->accessLayer(buffer, channel_name, layer_name);
            })
			.def_static("load_single_grid_layer_from_buffer", [](const std::string& bytes, const std::string& channel_name, const std::string& layer_name) -> std::shared_ptr<VoxelGrid> {
			    std::istringstream stream(bytes);
				auto accessor = FieldStore::construct_accessor(stream);

				if (accessor->getFieldType() != RadFiled3D::FieldType::Cartesian) {
					throw std::runtime_error("Field is not of type Cartesian");
				}

				return std::dynamic_pointer_cast<CartesianFieldAccessor>(accessor)->accessLayer(stream, channel_name, layer_name);
		    })
            .def_static("load_single_polar_layer", [](const std::string& file, const std::string& channel_name, const std::string& layer_name) -> std::shared_ptr<PolarSegments> {
			    std::ifstream buffer(file, std::ios::binary);
			    auto accessor = FieldStore::construct_accessor(buffer);

				if (accessor->getFieldType() != RadFiled3D::FieldType::Polar) {
					throw std::runtime_error("Field is not of type Polar");
				}

			    return std::dynamic_pointer_cast<PolarFieldAccessor>(accessor)->accessLayer(buffer, channel_name, layer_name);
            })
            .def_static("load_single_polar_layer_from_buffer", [](const std::string& bytes, const std::string& channel_name, const std::string& layer_name) -> std::shared_ptr<PolarSegments> {
			    std::istringstream stream(bytes);
			    auto accessor = FieldStore::construct_accessor(stream);

				if (accessor->getFieldType() != RadFiled3D::FieldType::Polar) {
					throw std::runtime_error("Field is not of type Polar");
				}

			    return std::dynamic_pointer_cast<PolarFieldAccessor>(accessor)->accessLayer(stream, channel_name, layer_name);
			}, py::arg("bytes"), py::arg("channel_name"), py::arg("layer_name"));


        // Datasets helper bindings
        py::class_<VoxelCollectionRequest>(m, "VoxelCollectionRequest")
            .def(py::init<const std::string&, const std::vector<size_t>&>(), py::arg("file_path"), py::arg("voxel_indices"))
            .def_readonly("file_path", &VoxelCollectionRequest::filePath)
            .def_readonly("voxel_indices", &VoxelCollectionRequest::voxelIndices);

        py::class_<VoxelCollection, std::shared_ptr<VoxelCollection>>(m, "VoxelCollection")
            .def("get_as_ndarray", [](std::shared_ptr<VoxelCollection>& self, const std::string& channel, const std::string& layer) {
			    auto channel_it = self->channels.find(channel);
                if (channel_it == self->channels.end())
					throw std::runtime_error("Channel '" + channel + "' not found in VoxelCollection");
				auto layer_it = channel_it->second.layers.find(layer);
				if (layer_it == channel_it->second.layers.end())
					throw std::runtime_error("Layer '" + layer + "' not found in channel '" + channel + "'");

                const Typing::DType type = Typing::Helper::get_dtype(layer_it->second.voxels[0]->get_type());
                char* data_buffer = self->extract_data_buffer_from(channel, layer);
                size_t voxel_count = self->channels.begin()->second.layers.begin()->second.voxels.size();

                switch (type) {
                    case Typing::DType::Float:
                        return create_py_array_generic<float>((float*)data_buffer, voxel_count);
                    case Typing::DType::Double:
                        return create_py_array_generic<double>((double*)data_buffer, voxel_count);
                    case Typing::DType::Int:
                        return create_py_array_generic<int>((int*)data_buffer, voxel_count);
                    case Typing::DType::Char:
                        return create_py_array_generic<char>(data_buffer, voxel_count);
                    case Typing::DType::UInt64:
                        return create_py_array_generic<uint64_t>((uint64_t*)data_buffer, voxel_count);
                    case Typing::DType::UInt32:
                        return create_py_array_generic<unsigned long>((unsigned long*)data_buffer, voxel_count);
                }

                const size_t element_size = layer_it->second.voxels[0]->get_bytes();
                return create_py_array_generic<float>((float*)data_buffer, voxel_count, element_size);
            }, py::arg("channel"), py::arg("layer"), py::return_value_policy::take_ownership);

        py::class_<VoxelCollectionAccessor>(m, "VoxelCollectionAccessor")
            .def(py::init<std::shared_ptr<Storage::FieldAccessor>, const std::vector<std::string>&, const std::vector<std::string>&>(), py::arg("accessor"), py::arg("channels"), py::arg("layers"))
            .def("access", &VoxelCollectionAccessor::access, py::arg("requests"));


        py::class_<GridTracer, std::shared_ptr<GridTracer>>(m, "GridTracer")
            .def("trace", &GridTracer::trace, py::arg("p1"), py::arg("p2"));

		py::class_<SamplingGridTracer, std::shared_ptr<SamplingGridTracer>, GridTracer>(m, "SamplingGridTracer")
			.def("trace", &SamplingGridTracer::trace, py::arg("p1"), py::arg("p2"));

		py::class_<BresenhamGridTracer, std::shared_ptr<BresenhamGridTracer>, GridTracer>(m, "BresenhamGridTracer")
			.def("trace", &BresenhamGridTracer::trace, py::arg("p1"), py::arg("p2"));

		py::class_<LinetracingGridTracer, std::shared_ptr<LinetracingGridTracer>, GridTracer>(m, "LinetracingGridTracer")
			.def("trace", &LinetracingGridTracer::trace, py::arg("p1"), py::arg("p2"));

		py::class_<PyGridTracerFactory>(m, "GridTracerFactory")
			.def_static("construct", &PyGridTracerFactory::construct, py::arg("field"), py::arg("algorithm") = GridTracerAlgorithm::SAMPLING);
}