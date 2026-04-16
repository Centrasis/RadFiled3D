#include "RadFiled3D/Voxel.hpp"
#include "RadFiled3D/RadiationField.hpp"
#include "RadFiled3D/storage/RadiationFieldStore.hpp"
#include "RadFiled3D/storage/FieldAccessor.hpp"
#include "RadFiled3D/storage/Types.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <fstream>
#include <cstdio>

using namespace RadFiled3D;
using namespace RadFiled3D::Storage;

namespace {
	// Fixture class for tests that create .rf3 files — ensures cleanup via TearDown
	class SphericalVoxelStorage : public ::testing::Test {
	protected:
		void TearDown() override {
			std::vector<std::string> files = {
				"test_spherical.rf3", "test_spherical_join.rf3",
				"test_spherical_accessor.rf3", "test_spherical_metadata.rf3",
				"test_spherical_mixed.rf3"
			};
			for (auto& file : files) {
				if (std::ifstream(file).good())
					std::remove(file.c_str());
			}
		}
	};

	// === Voxel Unit Tests ===

	TEST(SphericalVoxelTest, Construction) {
		float buffer[72] = { 0.f };
		AngularResolvedVoxel voxel(12, 6, buffer);

		EXPECT_EQ(voxel.get_phi_segments(), 12);
		EXPECT_EQ(voxel.get_theta_segments(), 6);
		EXPECT_EQ(voxel.get_total_segments(), 72);
		EXPECT_EQ(voxel.get_bytes(), sizeof(float) * 72);
		EXPECT_EQ(voxel.get_type(), "spherical");
	}

	TEST(SphericalVoxelTest, DefaultConstruction) {
		AngularResolvedVoxel voxel;

		EXPECT_EQ(voxel.get_phi_segments(), 0);
		EXPECT_EQ(voxel.get_theta_segments(), 0);
		EXPECT_EQ(voxel.get_total_segments(), 0);
	}

	TEST(SphericalVoxelTest, AccessByIndex) {
		float buffer[72] = { 0.f };
		AngularResolvedVoxel voxel(12, 6, buffer);

		buffer[2 * 12 + 3] = 0.5f;
		EXPECT_FLOAT_EQ(voxel.get_value(3, 2), 0.5f);
	}

	TEST(SphericalVoxelTest, AddValue) {
		float buffer[72] = { 0.f };
		AngularResolvedVoxel voxel(12, 6, buffer);

		voxel.add_value(1.0f, 0.5f, 3.0f);
		EXPECT_FLOAT_EQ(voxel.get_value_by_coord(1.0f, 0.5f), 3.0f);

		float sum = 0.f;
		for (size_t i = 0; i < 72; i++)
			sum += buffer[i];
		EXPECT_FLOAT_EQ(sum, 3.0f);
	}

	TEST(SphericalVoxelTest, Clear) {
		float buffer[4] = { 1.f, 3.f, 2.f, 4.f };
		AngularResolvedVoxel voxel(2, 2, buffer);

		voxel.clear();
		for (size_t i = 0; i < 4; i++)
			EXPECT_FLOAT_EQ(buffer[i], 0.f);
	}

	TEST(SphericalVoxelTest, InplaceAdd) {
		float buf_a[4] = { 1.f, 2.f, 3.f, 4.f };
		float buf_b[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
		AngularResolvedVoxel a(2, 2, buf_a);
		AngularResolvedVoxel b(2, 2, buf_b);

		a += b;
		EXPECT_FLOAT_EQ(buf_a[0], 1.5f);
		EXPECT_FLOAT_EQ(buf_a[1], 2.5f);
		EXPECT_FLOAT_EQ(buf_a[2], 3.5f);
		EXPECT_FLOAT_EQ(buf_a[3], 4.5f);
	}

	TEST(SphericalVoxelTest, ScalarDivide) {
		float buffer[4] = { 2.f, 4.f, 6.f, 8.f };
		AngularResolvedVoxel voxel(2, 2, buffer);

		voxel /= 2.f;
		EXPECT_FLOAT_EQ(buffer[0], 1.f);
		EXPECT_FLOAT_EQ(buffer[1], 2.f);
		EXPECT_FLOAT_EQ(buffer[2], 3.f);
		EXPECT_FLOAT_EQ(buffer[3], 4.f);
	}

	TEST(SphericalVoxelTest, Header) {
		float buffer[72] = { 0.f };
		AngularResolvedVoxel voxel(12, 6, buffer);

		auto header = voxel.get_header();
		EXPECT_EQ(header.header_bytes, sizeof(AngularResolvedVoxel::SphericalDefinition));

		AngularResolvedVoxel::SphericalDefinition* def = (AngularResolvedVoxel::SphericalDefinition*)header.header;
		EXPECT_EQ(def->phi_segments, 12);
		EXPECT_EQ(def->theta_segments, 6);
	}

	TEST(SphericalVoxelTest, InitFromHeader) {
		AngularResolvedVoxel voxel;
		AngularResolvedVoxel::SphericalDefinition def(8, 4);

		voxel.init_from_header(&def);
		EXPECT_EQ(voxel.get_phi_segments(), 8);
		EXPECT_EQ(voxel.get_theta_segments(), 4);
		EXPECT_EQ(voxel.get_total_segments(), 32);
	}

	TEST(SphericalVoxelTest, OwningConstruction) {
		OwningAngularResolvedVoxel voxel(12, 6);

		EXPECT_EQ(voxel.get_phi_segments(), 12);
		EXPECT_EQ(voxel.get_total_segments(), 72);
		EXPECT_NE(voxel.get_raw(), nullptr);

		voxel.add_value(1.0f, 0.5f, 5.0f);
		float sum = 0.f;
		for (auto val : voxel.get_segments_data())
			sum += val;
		EXPECT_FLOAT_EQ(sum, 5.0f);
	}

	TEST(SphericalVoxelTest, OwningCopy) {
		OwningAngularResolvedVoxel a(2, 2);
		float* a_data = (float*)a.get_raw();
		a_data[0] = 1.f; a_data[1] = 2.f; a_data[2] = 3.f; a_data[3] = 4.f;

		OwningAngularResolvedVoxel b(a);
		float* b_data = (float*)b.get_raw();

		EXPECT_NE(a_data, b_data);
		EXPECT_FLOAT_EQ(b_data[0], 1.f);
		EXPECT_FLOAT_EQ(b_data[3], 4.f);
	}

	// === Storage Tests (use fixture for cleanup) ===

	TEST_F(SphericalVoxelStorage, StoreAndLoad) {
		auto field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.1f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_custom_layer<AngularResolvedVoxel<float>>("angular", AngularResolvedVoxel<float>(12, 6, nullptr), 0.f, "");
		channel->add_layer<float>("doserate", 0.f, "Gy/s");

		AngularResolvedVoxel<float>& sph = channel->get_voxel<AngularResolvedVoxel<float>>("angular", 0, 5, 0);
		for (size_t i = 0; i < 72; i++)
			sph.get_segments_data()[i] = static_cast<float>(i);

		for (size_t x = 0; x < 10; x++) {
			AngularResolvedVoxel<float>& diag = channel->get_voxel<AngularResolvedVoxel<float>>("angular", x, x, x);
			for (size_t i = 0; i < 72; i++)
				diag.get_segments_data()[i] = 99.f;
		}

		channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0) = 42.f;

		auto metadata = std::make_shared<V1::RadiationFieldMetadata>(
			Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				1000, "test_geom", "test_physics",
				Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(0.f, 0.f, -1.f), glm::vec3(0.f, 0.f, 1.f), 15000.f, "test_tube"
				)
			),
			Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software("test", "1.0", "", "")
		);

		FieldStore::store(field, metadata, "test_spherical.rf3");

		auto loaded_field = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test_spherical.rf3"));
		EXPECT_EQ(loaded_field->get_voxel_counts(), glm::uvec3(10));

		std::shared_ptr<VoxelGridBuffer> loaded_channel = std::static_pointer_cast<VoxelGridBuffer>(loaded_field->get_channel("test_channel"));

		EXPECT_FLOAT_EQ(loaded_channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), 42.f);

		AngularResolvedVoxel<float>& loaded_sph = loaded_channel->get_voxel<AngularResolvedVoxel<float>>("angular", 0, 5, 0);
		EXPECT_EQ(loaded_sph.get_phi_segments(), 12);
		EXPECT_EQ(loaded_sph.get_theta_segments(), 6);
		for (size_t i = 0; i < 72; i++)
			EXPECT_FLOAT_EQ(loaded_sph.get_segments_data()[i], static_cast<float>(i));

		for (size_t x = 0; x < 10; x++) {
			AngularResolvedVoxel<float>& loaded_diag = loaded_channel->get_voxel<AngularResolvedVoxel<float>>("angular", x, x, x);
			for (size_t i = 0; i < 72; i++)
				EXPECT_FLOAT_EQ(loaded_diag.get_segments_data()[i], 99.f);
		}
	}

	TEST(SphericalVoxelTest, FieldCopy) {
		auto field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.1f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_custom_layer<AngularResolvedVoxel<float>>("angular", AngularResolvedVoxel<float>(12, 6, nullptr), 0.f, "");
		channel->add_layer<float>("doserate", 5.f, "Gy/s");

		AngularResolvedVoxel<float>& sph = channel->get_voxel<AngularResolvedVoxel<float>>("angular", 5, 5, 5);
		for (size_t i = 0; i < 72; i++)
			sph.get_segments_data()[i] = static_cast<float>(i + 1);

		auto copy = std::dynamic_pointer_cast<CartesianRadiationField>(field->copy());
		std::shared_ptr<VoxelGridBuffer> copy_channel = std::static_pointer_cast<VoxelGridBuffer>(copy->get_channel("test_channel"));

		EXPECT_EQ(copy->get_voxel_counts(), field->get_voxel_counts());
		EXPECT_EQ(copy->get_voxel_dimensions(), field->get_voxel_dimensions());

		AngularResolvedVoxel<float>& copy_sph = copy_channel->get_voxel<AngularResolvedVoxel<float>>("angular", 5, 5, 5);
		EXPECT_EQ(copy_sph.get_phi_segments(), 12);
		EXPECT_EQ(copy_sph.get_theta_segments(), 6);
		for (size_t i = 0; i < 72; i++)
			EXPECT_FLOAT_EQ(copy_sph.get_segments_data()[i], static_cast<float>(i + 1));

		// Deep copy: modifying copy doesn't affect original
		copy_sph.get_segments_data()[0] = 999.f;
		EXPECT_FLOAT_EQ(sph.get_segments_data()[0], 1.f);

		EXPECT_FLOAT_EQ(copy_channel->get_voxel<ScalarVoxel<float>>("doserate", 5, 5, 5).get_data(), 5.f);
	}

	TEST(SphericalVoxelTest, VoxelBufferOperators) {
		auto field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.1f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("ch1"));

		channel->add_custom_layer<AngularResolvedVoxel<float>>("angular", AngularResolvedVoxel<float>(4, 2, nullptr), 0.f, "");
		channel->add_layer<float>("doserate", 10.f, "Gy/s");

		AngularResolvedVoxel<float>& sph = channel->get_voxel<AngularResolvedVoxel<float>>("angular", 0, 0, 0);
		for (size_t i = 0; i < 8; i++)
			sph.get_segments_data()[i] = static_cast<float>(i);

		VoxelBuffer* original = static_cast<VoxelBuffer*>(channel->copy());
		EXPECT_FLOAT_EQ(static_cast<VoxelGridBuffer*>(original)->get_voxel<AngularResolvedVoxel<float>>("angular", 0, 0, 0).get_segments_data()[3], 3.f);

		*channel += *original;
		EXPECT_FLOAT_EQ(channel->get_voxel<AngularResolvedVoxel<float>>("angular", 0, 0, 0).get_segments_data()[3], 6.f);
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 0, 0).get_data(), 20.f);

		*channel -= *original;
		EXPECT_FLOAT_EQ(channel->get_voxel<AngularResolvedVoxel<float>>("angular", 0, 0, 0).get_segments_data()[3], 3.f);
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 0, 0).get_data(), 10.f);

		*channel *= *original;
		EXPECT_FLOAT_EQ(channel->get_voxel<AngularResolvedVoxel<float>>("angular", 0, 0, 0).get_segments_data()[3], 9.f);
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 0, 0).get_data(), 100.f);

		*channel /= *channel;
		EXPECT_FLOAT_EQ(channel->get_voxel<AngularResolvedVoxel<float>>("angular", 0, 0, 0).get_segments_data()[0], 0.f);
		EXPECT_FLOAT_EQ(channel->get_voxel<AngularResolvedVoxel<float>>("angular", 0, 0, 0).get_segments_data()[3], 1.f);
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 0, 0).get_data(), 1.f);

		delete original;
	}

	TEST(SphericalVoxelTest, GridAccess3D) {
		auto field = std::make_shared<CartesianRadiationField>(glm::vec3(0.5f), glm::vec3(0.1f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("ch1"));

		channel->add_custom_layer<AngularResolvedVoxel<float>>("angular", AngularResolvedVoxel<float>(6, 3, nullptr), 0.f, "");

		channel->get_voxel<AngularResolvedVoxel<float>>("angular", 0, 0, 0).add_value(0.f, 0.f, 1.f);
		channel->get_voxel<AngularResolvedVoxel<float>>("angular", 4, 3, 2).add_value(1.f, 0.5f, 5.f);
		channel->get_voxel<AngularResolvedVoxel<float>>("angular", 2, 2, 2).add_value(3.14f, 1.57f, 10.f);

		size_t flat_432 = 2 * 5 * 5 + 3 * 5 + 4;
		size_t flat_222 = 2 * 5 * 5 + 2 * 5 + 2;

		float sum_0 = 0.f, sum_432 = 0.f, sum_222 = 0.f;
		for (auto val : channel->get_voxel_flat<AngularResolvedVoxel<float>>("angular", 0).get_segments_data()) sum_0 += val;
		for (auto val : channel->get_voxel_flat<AngularResolvedVoxel<float>>("angular", flat_432).get_segments_data()) sum_432 += val;
		for (auto val : channel->get_voxel_flat<AngularResolvedVoxel<float>>("angular", flat_222).get_segments_data()) sum_222 += val;

		EXPECT_FLOAT_EQ(sum_0, 1.f);
		EXPECT_FLOAT_EQ(sum_432, 5.f);
		EXPECT_FLOAT_EQ(sum_222, 10.f);

		float sum_empty = 0.f;
		for (auto val : channel->get_voxel<AngularResolvedVoxel<float>>("angular", 1, 1, 1).get_segments_data()) sum_empty += val;
		EXPECT_FLOAT_EQ(sum_empty, 0.f);
	}

	TEST_F(SphericalVoxelStorage, JoinFields) {
		auto create_field = [](float value) {
			auto field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.5f));
			std::shared_ptr<VoxelGridBuffer> ch = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("beam"));
			ch->add_custom_layer<AngularResolvedVoxel<float>>("angular", AngularResolvedVoxel<float>(4, 2, nullptr), 0.f, "");
			ch->add_layer<float>("flux", value, "counts");

			AngularResolvedVoxel<float>& sph = ch->get_voxel_flat<AngularResolvedVoxel<float>>("angular", 0);
			for (size_t i = 0; i < 8; i++)
				sph.get_segments_data()[i] = value;

			auto metadata = std::make_shared<V1::RadiationFieldMetadata>(
				Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
					1000, "geom", "physics",
					Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
						glm::vec3(0, 0, -1), glm::vec3(0, 0, 1), 15000.f, "tube"
					)
				),
				Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software("test", "1.0", "", "")
			);
			return std::make_pair(field, metadata);
		};

		auto [field1, meta1] = create_field(3.f);
		FieldStore::store(field1, meta1, "test_spherical_join.rf3");

		auto [field2, meta2] = create_field(7.f);
		FieldStore::join(field2, meta2, "test_spherical_join.rf3", FieldJoinMode::Add, FieldJoinCheckMode::NoChecks);

		auto loaded = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test_spherical_join.rf3"));
		std::shared_ptr<VoxelGridBuffer> loaded_ch = std::static_pointer_cast<VoxelGridBuffer>(loaded->get_channel("beam"));

		EXPECT_FLOAT_EQ(loaded_ch->get_voxel_flat<ScalarVoxel<float>>("flux", 0).get_data(), 10.f);

		AngularResolvedVoxel<float>& loaded_sph = loaded_ch->get_voxel_flat<AngularResolvedVoxel<float>>("angular", 0);
		EXPECT_EQ(loaded_sph.get_phi_segments(), 4);
		EXPECT_EQ(loaded_sph.get_theta_segments(), 2);
		for (size_t i = 0; i < 8; i++)
			EXPECT_FLOAT_EQ(loaded_sph.get_segments_data()[i], 10.f);
	}

	TEST_F(SphericalVoxelStorage, Accessor) {
		auto field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.5f));
		std::shared_ptr<VoxelGridBuffer> ch = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("beam"));
		ch->add_custom_layer<AngularResolvedVoxel<float>>("angular", AngularResolvedVoxel<float>(6, 3, nullptr), 0.f, "");
		ch->add_layer<float>("flux", 0.f, "counts");

		AngularResolvedVoxel<float>& sph = ch->get_voxel<AngularResolvedVoxel<float>>("angular", 1, 1, 1);
		for (size_t i = 0; i < 18; i++)
			sph.get_segments_data()[i] = static_cast<float>(i + 1);
		ch->get_voxel<ScalarVoxel<float>>("flux", 1, 1, 1) = 42.f;

		auto metadata = std::make_shared<V1::RadiationFieldMetadata>(
			Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				100, "geom", "physics",
				Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(0, 0, -1), glm::vec3(0, 0, 1), 15000.f, "tube"
				)
			),
			Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software("test", "1.0", "", "")
		);
		FieldStore::store(field, metadata, "test_spherical_accessor.rf3");

		// Full field load
		auto loaded_field = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test_spherical_accessor.rf3"));
		std::shared_ptr<VoxelGridBuffer> loaded_ch = std::static_pointer_cast<VoxelGridBuffer>(loaded_field->get_channel("beam"));

		AngularResolvedVoxel<float>& loaded_sph = loaded_ch->get_voxel<AngularResolvedVoxel<float>>("angular", 1, 1, 1);
		EXPECT_EQ(loaded_sph.get_phi_segments(), 6);
		EXPECT_EQ(loaded_sph.get_theta_segments(), 3);
		for (size_t i = 0; i < 18; i++)
			EXPECT_FLOAT_EQ(loaded_sph.get_segments_data()[i], static_cast<float>(i + 1));
		EXPECT_FLOAT_EQ(loaded_ch->get_voxel<ScalarVoxel<float>>("flux", 1, 1, 1).get_data(), 42.f);

		// Accessor with fresh stream (required pattern — see accessor_test.cpp)
		{
			std::ifstream stream1("test_spherical_accessor.rf3", std::ios::binary);
			auto accessor = FieldStore::construct_accessor(stream1);
			auto cartesian_accessor = std::dynamic_pointer_cast<Storage::V1::CartesianFieldAccessor>(accessor);
			ASSERT_NE(cartesian_accessor, nullptr);

			std::ifstream stream2("test_spherical_accessor.rf3", std::ios::binary);
			auto loaded_channel_acc = cartesian_accessor->accessChannel(stream2, "beam");
			ASSERT_NE(loaded_channel_acc, nullptr);
			EXPECT_TRUE(loaded_channel_acc->has_layer("angular"));
			AngularResolvedVoxel<float>& acc_sph = loaded_channel_acc->get_voxel_flat<AngularResolvedVoxel<float>>("angular", 7);
			EXPECT_EQ(acc_sph.get_phi_segments(), 6);
			for (size_t i = 0; i < 18; i++)
				EXPECT_FLOAT_EQ(acc_sph.get_segments_data()[i], static_cast<float>(i + 1));

			AngularResolvedVoxel<float>& acc_empty = loaded_channel_acc->get_voxel_flat<AngularResolvedVoxel<float>>("angular", 0);
			float sum = 0.f;
			for (auto val : acc_empty.get_segments_data()) sum += val;
			EXPECT_FLOAT_EQ(sum, 0.f);
		}
	}

	TEST_F(SphericalVoxelStorage, DynamicMetadata) {
		auto field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.5f));
		field->add_channel("beam");

		auto metadata = std::make_shared<V1::RadiationFieldMetadata>(
			Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				100, "geom", "physics",
				Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(0, 0, -1), glm::vec3(0, 0, 1), 15000.f, "tube"
				)
			),
			Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software("test", "1.0", "", "")
		);

		metadata->set_dynamic_custom_metadata<AngularResolvedVoxel<float>>("angular_reference", AngularResolvedVoxel<float>(8, 4, nullptr));
		AngularResolvedVoxel<float>& meta_sph = metadata->get_dynamic_metadata<AngularResolvedVoxel<float>>("angular_reference");
		EXPECT_EQ(meta_sph.get_phi_segments(), 8);
		EXPECT_EQ(meta_sph.get_theta_segments(), 4);
		for (size_t i = 0; i < 32; i++)
			meta_sph.get_segments_data()[i] = static_cast<float>(i);

		metadata->add_dynamic_metadata<float>("test_value", 3.14f);

		FieldStore::store(field, metadata, "test_spherical_metadata.rf3");

		auto loaded_metadata = std::dynamic_pointer_cast<V1::RadiationFieldMetadata>(FieldStore::load_metadata("test_spherical_metadata.rf3"));
		ASSERT_NE(loaded_metadata, nullptr);

		auto& loaded_float = loaded_metadata->get_dynamic_metadata<ScalarVoxel<float>>("test_value");
		EXPECT_FLOAT_EQ(loaded_float.get_data(), 3.14f);

		auto keys = loaded_metadata->get_dynamic_metadata_keys();
		bool found_angular = false;
		for (auto& key : keys)
			if (key == "angular_reference") found_angular = true;
		EXPECT_TRUE(found_angular);

		AngularResolvedVoxel<float>& loaded_sph = loaded_metadata->get_dynamic_metadata<AngularResolvedVoxel<float>>("angular_reference");
		EXPECT_EQ(loaded_sph.get_phi_segments(), 8);
		EXPECT_EQ(loaded_sph.get_theta_segments(), 4);
		for (size_t i = 0; i < 32; i++)
			EXPECT_FLOAT_EQ(loaded_sph.get_segments_data()[i], static_cast<float>(i));
	}

	// Mixed voxel types in same channel — production uses AngularResolvedVoxel + HistogramVoxel + float together
	TEST_F(SphericalVoxelStorage, MixedCustomVoxelTypes) {
		auto field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.5f));
		std::shared_ptr<VoxelGridBuffer> ch = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("beam"));

		ch->add_custom_layer<AngularResolvedVoxel<float>>("angular", AngularResolvedVoxel<float>(6, 3, nullptr), 0.f, "");
		ch->add_custom_layer<HistogramVoxel<float>>("spectrum", HistogramVoxel<float>(16, 1000.f, nullptr), 0.f, "eV");
		ch->add_layer<float>("flux", 0.f, "counts");
		ch->add_layer<glm::vec3>("direction", glm::vec3(0.f), "direction");

		// Set values in all layer types
		ch->get_voxel<AngularResolvedVoxel<float>>("angular", 0, 0, 0).add_value(1.f, 0.5f, 7.f);
		ch->get_voxel<HistogramVoxel<float>>("spectrum", 0, 0, 0).get_histogram()[5] = 42.f;
		ch->get_voxel<ScalarVoxel<float>>("flux", 0, 0, 0) = 100.f;
		ch->get_voxel<ScalarVoxel<glm::vec3>>("direction", 0, 0, 0) = glm::vec3(1.f, 0.f, 0.f);

		auto metadata = std::make_shared<V1::RadiationFieldMetadata>(
			Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				100, "geom", "physics",
				Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(0, 0, -1), glm::vec3(0, 0, 1), 15000.f, "tube"
				)
			),
			Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software("test", "1.0", "", "")
		);
		FieldStore::store(field, metadata, "test_spherical_mixed.rf3");

		// Load and verify all types
		auto loaded = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test_spherical_mixed.rf3"));
		std::shared_ptr<VoxelGridBuffer> loaded_ch = std::static_pointer_cast<VoxelGridBuffer>(loaded->get_channel("beam"));

		EXPECT_FLOAT_EQ(loaded_ch->get_voxel<ScalarVoxel<float>>("flux", 0, 0, 0).get_data(), 100.f);
		EXPECT_FLOAT_EQ(loaded_ch->get_voxel<HistogramVoxel<float>>("spectrum", 0, 0, 0).get_histogram()[5], 42.f);
		EXPECT_EQ(loaded_ch->get_voxel<HistogramVoxel<float>>("spectrum", 0, 0, 0).get_bins(), 16);

		AngularResolvedVoxel<float>& loaded_sph = loaded_ch->get_voxel<AngularResolvedVoxel<float>>("angular", 0, 0, 0);
		EXPECT_EQ(loaded_sph.get_phi_segments(), 6);
		EXPECT_EQ(loaded_sph.get_theta_segments(), 3);
		float sph_sum = 0.f;
		for (auto val : loaded_sph.get_segments_data()) sph_sum += val;
		EXPECT_FLOAT_EQ(sph_sum, 7.f);

		glm::vec3 loaded_dir = loaded_ch->get_voxel<ScalarVoxel<glm::vec3>>("direction", 0, 0, 0).get_data();
		EXPECT_FLOAT_EQ(loaded_dir.x, 1.f);
		EXPECT_FLOAT_EQ(loaded_dir.y, 0.f);
		EXPECT_FLOAT_EQ(loaded_dir.z, 0.f);
	}
}
