#include "RadFiled3D/VoxelGrid.hpp"
#include "RadFiled3D/RadiationField.hpp"
#include <iostream>
#include "RadFiled3D/storage/RadiationFieldStore.hpp"
#include <memory>
#include <vector>
#include <chrono>
#include <fstream>
#include "gtest/gtest.h"
#include <fstream>
#include <cstdio>
#include <thread>
#include <shared_mutex>
#ifdef _WIN32
#include <Windows.h>
#include "psapi.h"
#else
#include "sys/types.h"
#include "sys/sysinfo.h"
#endif

using namespace RadFiled3D;
using namespace RadFiled3D::Storage;

namespace {
	class Storage : public ::testing::Test {
	protected:
		void TearDown() override {
			std::vector<std::string> files = { "test01.rf3" };
			for (auto& file : files) {
				if (std::ifstream(file).good())
					std::remove(file.c_str());
			}
		}
	};

	TEST(Field, CreationAndDestruction) {
		unsigned long long field_memory_consuption = 0;
		{
			std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.02f));
			auto channel = field->add_channel("test_channel");
			channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
			channel->add_layer<float>("doserate", 25.3f, "Gy/s");
			channel->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");
			if (field_memory_consuption == 0) {
				field_memory_consuption += sizeof(CartesianRadiationField);
				field_memory_consuption += sizeof(VoxelGridBuffer) * 3;
				field_memory_consuption += field->get_voxel_counts().x * field->get_voxel_counts().y * field->get_voxel_counts().z * sizeof(ScalarVoxel<float>);
				field_memory_consuption += field->get_voxel_counts().x * field->get_voxel_counts().y * field->get_voxel_counts().z * sizeof(ScalarVoxel<glm::vec3>);
				field_memory_consuption += field->get_voxel_counts().x * field->get_voxel_counts().y * field->get_voxel_counts().z * sizeof(float) * 26;
			}
			field_memory_consuption *= 1000;
		}

		unsigned long long totalVirtualUsed_begining = 0;
#ifdef _WIN32
		PROCESS_MEMORY_COUNTERS_EX pmc;
		GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
		totalVirtualUsed_begining = static_cast<unsigned long long>(pmc.PrivateUsage);
#else
		struct sysinfo memInfo;
		sysinfo(&memInfo);

		totalVirtualUsed_begining = static_cast<unsigned long long>(memInfo.totalram - memInfo.freeram);
		totalVirtualUsed_begining += static_cast<unsigned long long>(memInfo.totalswap - memInfo.freeswap);
		totalVirtualUsed_begining *= static_cast<unsigned long long>(memInfo.mem_unit);
#endif

		for (size_t i = 0; i < 500; i++) {
			auto field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.02f));
			auto channel = field->add_channel("test_channel");
			channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
			channel->add_layer<float>("doserate", 25.3f, "Gy/s");
			channel->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");
		}

		unsigned long long totalVirtualUsed_end = 0;
#ifdef _WIN32
		GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
		totalVirtualUsed_end = static_cast<unsigned long long>(pmc.PrivateUsage);
#else
		sysinfo(&memInfo);

		totalVirtualUsed_end = static_cast<unsigned long long>(memInfo.totalram - memInfo.freeram);
		totalVirtualUsed_end += static_cast<unsigned long long>(memInfo.totalswap - memInfo.freeswap);
		totalVirtualUsed_end *= static_cast<unsigned long long>(memInfo.mem_unit);
#endif

		std::cout << "Memory used in the begining: " << totalVirtualUsed_begining << std::endl;
		std::cout << "Memory used in the end: " << totalVirtualUsed_end << std::endl;
		unsigned long long memoryUsed = (totalVirtualUsed_end > totalVirtualUsed_begining) ? totalVirtualUsed_end - totalVirtualUsed_begining : 0;

		std::cout << "Memory used per field: " << memoryUsed / 1000 << std::endl;

		EXPECT_TRUE(memoryUsed / 1000 < field_memory_consuption + 1);
	}


	TEST(Storage, FileLoading) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_layer<float>("doserate", 25.3f, "Gy/s");
		channel->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");

		std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				100,
				"geom",
				"FTFP_BERT",
				RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(1.f, 0.f, 0.f),
					glm::vec3(0.f, 0.f, 0.f),
					100.f,
					"XRayTube"
				)
			),
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
				"test",
				"1.0",
				"repo",
				"commit"
			)
		);

		EXPECT_NO_THROW(FieldStore::store(field, metadata, "test01.rf3", StoreVersion::V1));

		// get memory consuption of this process under windows and linux
		unsigned long long totalVirtualUsed_begining = 0;
#ifdef _WIN32
		PROCESS_MEMORY_COUNTERS_EX pmc;
		GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
		totalVirtualUsed_begining = static_cast<unsigned long long>(pmc.PrivateUsage);
#else
		struct sysinfo memInfo;
		sysinfo(&memInfo);

		totalVirtualUsed_begining = static_cast<unsigned long long>(memInfo.totalram - memInfo.freeram);
		totalVirtualUsed_begining += static_cast<unsigned long long>(memInfo.totalswap - memInfo.freeswap);
		totalVirtualUsed_begining *= static_cast<unsigned long long>(memInfo.mem_unit);
#endif

		unsigned long long field_memory_consuption = 0;
		for (size_t i = 0; i < 1000; i++) {
			EXPECT_NO_THROW(auto field = FieldStore::load("test01.rf3"));
			if (field_memory_consuption == 0) {
				field_memory_consuption += sizeof(CartesianRadiationField);
				field_memory_consuption += sizeof(VoxelGridBuffer) * 3;
				field_memory_consuption += field->get_voxel_counts().x * field->get_voxel_counts().y * field->get_voxel_counts().z * sizeof(ScalarVoxel<float>);
				field_memory_consuption += field->get_voxel_counts().x * field->get_voxel_counts().y * field->get_voxel_counts().z * sizeof(ScalarVoxel<glm::vec3>);
				field_memory_consuption += field->get_voxel_counts().x * field->get_voxel_counts().y * field->get_voxel_counts().z * sizeof(float) * 26;
			}
		}

		unsigned long long totalVirtualUsed_end = 0;
#ifdef _WIN32
		GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
		totalVirtualUsed_end = static_cast<unsigned long long>(pmc.PrivateUsage);
#else
		sysinfo(&memInfo);

		totalVirtualUsed_end = static_cast<unsigned long long>(memInfo.totalram - memInfo.freeram);
		totalVirtualUsed_end += static_cast<unsigned long long>(memInfo.totalswap - memInfo.freeswap);
		totalVirtualUsed_end *= static_cast<unsigned long long>(memInfo.mem_unit);
#endif

		std::cout << "Memory used in the begining: " << totalVirtualUsed_begining << std::endl;
		std::cout << "Memory used in the end: " << totalVirtualUsed_end << std::endl;
		unsigned long long memoryUsed = (totalVirtualUsed_end > totalVirtualUsed_begining) ? totalVirtualUsed_end - totalVirtualUsed_begining : 0;

		std::cout << "Memory used per field: " << memoryUsed / 1000 << std::endl;

		EXPECT_TRUE(memoryUsed / 1000 < field_memory_consuption + 1);
	}

	TEST(Storage, FileLayerLoading) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_layer<float>("doserate", 25.3f, "Gy/s");
		channel->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");

		std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				100,
				"geom",
				"FTFP_BERT",
				RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(1.f, 0.f, 0.f),
					glm::vec3(0.f, 0.f, 0.f),
					100.f,
					"XRayTube"
				)
			),
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
				"test",
				"1.0",
				"repo",
				"commit"
			)
		);

		EXPECT_NO_THROW(FieldStore::store(field, metadata, "test01.rf3", StoreVersion::V1));

		// get memory consuption of this process under windows and linux
		unsigned long long totalVirtualUsed_begining = 0;
#ifdef _WIN32
		PROCESS_MEMORY_COUNTERS_EX pmc;
		GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
		totalVirtualUsed_begining = static_cast<unsigned long long>(pmc.PrivateUsage);
#else
		struct sysinfo memInfo;
		sysinfo(&memInfo);

		totalVirtualUsed_begining = static_cast<unsigned long long>(memInfo.totalram - memInfo.freeram);
		totalVirtualUsed_begining += static_cast<unsigned long long>(memInfo.totalswap - memInfo.freeswap);
		totalVirtualUsed_begining *= static_cast<unsigned long long>(memInfo.mem_unit);
#endif

		unsigned long long field_memory_consuption = 0;

		for (size_t i = 0; i < 1000; i++) {
			std::ifstream file("test01.rf3", std::ios::binary);
			std::shared_ptr<VoxelLayer> field;
			EXPECT_NO_THROW(field = FieldStore::load_single_layer(file, "test_channel", "spectra"));
			if (field_memory_consuption == 0) {
				field_memory_consuption += sizeof(VoxelLayer);
				field_memory_consuption += field->get_voxel_count() * sizeof(ScalarVoxel<float>);
				field_memory_consuption += field->get_voxel_count() * sizeof(float) * 26;
			}
		}

		unsigned long long totalVirtualUsed_end = 0;
#ifdef _WIN32
		GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
		totalVirtualUsed_end = static_cast<unsigned long long>(pmc.PrivateUsage);
#else
		sysinfo(&memInfo);

		totalVirtualUsed_end = static_cast<unsigned long long>(memInfo.totalram - memInfo.freeram);
		totalVirtualUsed_end += static_cast<unsigned long long>(memInfo.totalswap - memInfo.freeswap);
		totalVirtualUsed_end *= static_cast<unsigned long long>(memInfo.mem_unit);
#endif

		std::cout << "Memory used in the begining: " << totalVirtualUsed_begining << std::endl;
		std::cout << "Memory used in the end: " << totalVirtualUsed_end << std::endl;
		unsigned long long memoryUsed = (totalVirtualUsed_end > totalVirtualUsed_begining) ? totalVirtualUsed_end - totalVirtualUsed_begining : 0;

		std::cout << "Memory used per field: " << memoryUsed / 1000 << std::endl;

		EXPECT_TRUE(memoryUsed / 1000 < field_memory_consuption + 1);
	}
}