#include "radfiled3d/voxel_grid.hpp"
#include "radfiled3d/radiation_field.hpp"
#include "radfiled3d/storage/radiation_field_store.hpp"
#include "radfiled3d/storage/types.hpp"
#include "gtest/gtest.h"
#include <cstdio>
#include <memory>
#include <string>

using namespace radfiled3d;
using namespace radfiled3d::storage;

namespace {
	TEST(FieldCreationTest, ChannelsMatchTheFieldsVoxelCount) {
		// a loaded field is rebuilt from counts * voxel size, which is 59.99999 voxels of 0.02 m for 60
		for (unsigned n : { 27u, 30u, 54u, 60u, 96u, 100u }) {
			for (float voxel : { 0.005f, 0.01f, 0.02f }) {
				CartesianRadiationField field(glm::vec3(glm::uvec3(n)) * glm::vec3(voxel), glm::vec3(voxel));
				auto channel = std::static_pointer_cast<VoxelGridBuffer>(field.add_channel("c"));
				EXPECT_EQ(field.get_voxel_counts(), glm::uvec3(n)) << n << " x " << voxel;
				EXPECT_EQ(channel->get_voxel_counts(), glm::uvec3(n)) << n << " x " << voxel;
				EXPECT_EQ(channel->get_voxel_count(), static_cast<size_t>(n) * n * n) << n << " x " << voxel;
			}
		}
	}

	TEST(Storage, LoadsFieldsWhoseSizeIsNotExactInFloat) {
		const std::string file = "test_voxel_count_roundtrip.rf3";
		auto field = std::make_shared<CartesianRadiationField>(glm::vec3(1.2f), glm::vec3(0.02f));
		ASSERT_EQ(field->get_voxel_counts(), glm::uvec3(60));
		field->add_channel("c")->add_layer<float>("flux", 1.f, "");
		auto metadata = std::make_shared<v1::RadiationFieldMetadata>(
			filed_types::v1::RadiationFieldMetadataHeader::Simulation(1, "", "", filed_types::v1::RadiationFieldMetadataHeader::Simulation::XRayTube(glm::vec3(0.f), glm::vec3(0.f), 0.f, "")),
			filed_types::v1::RadiationFieldMetadataHeader::Software("", "", "", "")
		);
		FieldStore::store(field, metadata, file, StoreVersion::V1);

		auto loaded = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load(file));
		EXPECT_EQ(loaded->get_voxel_counts(), glm::uvec3(60));
		EXPECT_EQ(loaded->get_channel("c")->get_voxel_count(), 60u * 60u * 60u);
		EXPECT_FLOAT_EQ(loaded->get_channel("c")->get_layer<float>("flux")[60 * 60 * 60 - 1], 1.f);
		std::remove(file.c_str());
	}
}
