#include "RadFiled3D/Voxel.hpp"
#include "RadFiled3D/RadiationField.hpp"
#include "RadFiled3D/storage/RadiationFieldStore.hpp"
#include "RadFiled3D/storage/Types.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <fstream>
#include <cstdio>
#include <iostream>

using namespace RadFiled3D;
using namespace RadFiled3D::Storage;

TEST(SphericalVoxelTest, Construction) {
	float buffer[72] = { 0.f };
	SphericalVoxel voxel(12, 6, buffer);

	EXPECT_EQ(voxel.get_phi_segments(), 12);
	EXPECT_EQ(voxel.get_theta_segments(), 6);
	EXPECT_EQ(voxel.get_total_segments(), 72);
	EXPECT_EQ(voxel.get_bytes(), sizeof(float) * 72);
	EXPECT_EQ(voxel.get_type(), "spherical");
}

TEST(SphericalVoxelTest, DefaultConstruction) {
	SphericalVoxel voxel;

	EXPECT_EQ(voxel.get_phi_segments(), 0);
	EXPECT_EQ(voxel.get_theta_segments(), 0);
	EXPECT_EQ(voxel.get_total_segments(), 0);
}

TEST(SphericalVoxelTest, AccessByIndex) {
	float buffer[72] = { 0.f };
	SphericalVoxel voxel(12, 6, buffer);

	// Use add_value to set data, then read back with get_value
	buffer[2 * 12 + 3] = 0.5f;  // set raw buffer directly
	EXPECT_FLOAT_EQ(voxel.get_value(3, 2), 0.5f);  // read through voxel
}

TEST(SphericalVoxelTest, AddValue) {
	float buffer[72] = { 0.f };
	SphericalVoxel voxel(12, 6, buffer);

	voxel.add_value(1.0f, 0.5f, 3.0f);
	// Verify the exact segment got the value
	EXPECT_FLOAT_EQ(voxel.get_value_by_coord(1.0f, 0.5f), 3.0f);

	// Verify total sum is correct (only one segment was written)
	float sum = 0.f;
	for (size_t i = 0; i < 72; i++)
		sum += buffer[i];
	EXPECT_FLOAT_EQ(sum, 3.0f);
}


TEST(SphericalVoxelTest, Clear) {
	float buffer[4] = { 1.f, 3.f, 2.f, 4.f };
	SphericalVoxel voxel(2, 2, buffer);

	voxel.clear();
	for (size_t i = 0; i < 4; i++)
		EXPECT_FLOAT_EQ(buffer[i], 0.f);
}

TEST(SphericalVoxelTest, InplaceAdd) {
	float buf_a[4] = { 1.f, 2.f, 3.f, 4.f };
	float buf_b[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
	SphericalVoxel a(2, 2, buf_a);
	SphericalVoxel b(2, 2, buf_b);

	a += b;
	EXPECT_FLOAT_EQ(buf_a[0], 1.5f);
	EXPECT_FLOAT_EQ(buf_a[1], 2.5f);
	EXPECT_FLOAT_EQ(buf_a[2], 3.5f);
	EXPECT_FLOAT_EQ(buf_a[3], 4.5f);
}

TEST(SphericalVoxelTest, ScalarDivide) {
	float buffer[4] = { 2.f, 4.f, 6.f, 8.f };
	SphericalVoxel voxel(2, 2, buffer);

	voxel /= 2.f;
	EXPECT_FLOAT_EQ(buffer[0], 1.f);
	EXPECT_FLOAT_EQ(buffer[1], 2.f);
	EXPECT_FLOAT_EQ(buffer[2], 3.f);
	EXPECT_FLOAT_EQ(buffer[3], 4.f);
}

TEST(SphericalVoxelTest, Header) {
	float buffer[72] = { 0.f };
	SphericalVoxel voxel(12, 6, buffer);

	auto header = voxel.get_header();
	EXPECT_EQ(header.header_bytes, sizeof(SphericalVoxel::SphericalDefinition));

	SphericalVoxel::SphericalDefinition* def = (SphericalVoxel::SphericalDefinition*)header.header;
	EXPECT_EQ(def->phi_segments, 12);
	EXPECT_EQ(def->theta_segments, 6);
}

TEST(SphericalVoxelTest, InitFromHeader) {
	SphericalVoxel voxel;
	SphericalVoxel::SphericalDefinition def(8, 4);

	voxel.init_from_header(&def);
	EXPECT_EQ(voxel.get_phi_segments(), 8);
	EXPECT_EQ(voxel.get_theta_segments(), 4);
	EXPECT_EQ(voxel.get_total_segments(), 32);
}

TEST(SphericalVoxelTest, OwningConstruction) {
	OwningSphericalVoxel voxel(12, 6);

	EXPECT_EQ(voxel.get_phi_segments(), 12);
	EXPECT_EQ(voxel.get_total_segments(), 72);
	EXPECT_NE(voxel.get_raw(), nullptr);

	// Should be zero-initialized or at least accessible
	voxel.add_value(1.0f, 0.5f, 5.0f);
	float sum = 0.f;
	for (auto val : voxel.get_segments_data())
		sum += val;
	EXPECT_FLOAT_EQ(sum, 5.0f);
}

TEST(SphericalVoxelTest, OwningCopy) {
	OwningSphericalVoxel a(2, 2);
	float* a_data = (float*)a.get_raw();
	a_data[0] = 1.f; a_data[1] = 2.f; a_data[2] = 3.f; a_data[3] = 4.f;

	OwningSphericalVoxel b(a);
	float* b_data = (float*)b.get_raw();

	// Data should be equal but at different addresses
	EXPECT_NE(a_data, b_data);
	EXPECT_FLOAT_EQ(b_data[0], 1.f);
	EXPECT_FLOAT_EQ(b_data[3], 4.f);
}

TEST(SphericalVoxelTest, StoreAndLoad) {
	const std::string filename = "test_spherical.rf3";

	// Create a field with a spherical layer
	auto field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.1f));
	std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

	// Add a spherical layer with 12 phi x 6 theta segments
	channel->add_custom_layer<SphericalVoxel>("angular", SphericalVoxel(12, 6, nullptr), 0.f, "");
	// Also add a simple float layer to verify mixed types work
	channel->add_layer<float>("doserate", 0.f, "Gy/s");

	// Set some values in the spherical layer
	SphericalVoxel& sph = channel->get_voxel<SphericalVoxel>("angular", 0, 5, 0);
	for (size_t i = 0; i < 72; i++)
		sph.get_segments_data()[i] = static_cast<float>(i);

	// Set a value on the diagonal
	for (size_t x = 0; x < 10; x++) {
		SphericalVoxel& diag = channel->get_voxel<SphericalVoxel>("angular", x, x, x);
		for (size_t i = 0; i < 72; i++)
			diag.get_segments_data()[i] = 99.f;
	}

	// Set doserate too
	channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0) = 42.f;

	// --- Print before store ---
	std::cout << "\n=== BEFORE STORE ===" << std::endl;
	std::cout << "Field: " << field->get_voxel_counts().x << "x" << field->get_voxel_counts().y << "x" << field->get_voxel_counts().z << " voxels" << std::endl;
	std::cout << "Voxel dimensions: " << field->get_voxel_dimensions().x << "m" << std::endl;
	std::cout << "Channels: ";
	for (auto& ch : field->get_channels()) std::cout << ch.first << " ";
	std::cout << std::endl;
	std::cout << "Layers: angular (spherical " << sph.get_phi_segments() << "x" << sph.get_theta_segments() << "), doserate (float)" << std::endl;
	std::cout << "\nVoxel (0,5,0) angular - first 12 segments (phi row 0):" << std::endl;
	for (size_t i = 0; i < 12; i++)
		std::cout << "  [phi=" << i << ",theta=0] = " << sph.get_segments_data()[i] << std::endl;
	std::cout << "Voxel (0,5,0) doserate = " << channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data() << std::endl;
	std::cout << "\nDiagonal voxel (3,3,3) angular - first 6 segments:" << std::endl;
	SphericalVoxel& diag_print = channel->get_voxel<SphericalVoxel>("angular", 3, 3, 3);
	for (size_t i = 0; i < 6; i++)
		std::cout << "  [" << i << "] = " << diag_print.get_segments_data()[i] << std::endl;

	// Create metadata and store
	auto metadata = std::make_shared<V1::RadiationFieldMetadata>(
		Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
			1000, "test_geom", "test_physics",
			Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
				glm::vec3(0.f, 0.f, -1.f), glm::vec3(0.f, 0.f, 1.f), 15000.f, "test_tube"
			)
		),
		Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software("test", "1.0", "", "")
	);

	FieldStore::store(field, metadata, filename);

	// Load and verify
	auto loaded_field = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load(filename));
	EXPECT_EQ(loaded_field->get_voxel_counts(), glm::uvec3(10));

	std::shared_ptr<VoxelGridBuffer> loaded_channel = std::static_pointer_cast<VoxelGridBuffer>(loaded_field->get_channel("test_channel"));

	// --- Print after load ---
	std::cout << "\n=== AFTER LOAD ===" << std::endl;
	std::cout << "Field: " << loaded_field->get_voxel_counts().x << "x" << loaded_field->get_voxel_counts().y << "x" << loaded_field->get_voxel_counts().z << " voxels" << std::endl;
	std::cout << "Layers in channel: ";
	for (auto& l : loaded_channel->get_layers()) std::cout << l << " ";
	std::cout << std::endl;

	// Verify doserate
	float loaded_doserate = loaded_channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data();
	std::cout << "\nVoxel (0,5,0) doserate = " << loaded_doserate << std::endl;
	EXPECT_FLOAT_EQ(loaded_doserate, 42.f);

	// Verify spherical voxel at (0,5,0)
	SphericalVoxel& loaded_sph = loaded_channel->get_voxel<SphericalVoxel>("angular", 0, 5, 0);
	std::cout << "Voxel (0,5,0) angular: phi_segments=" << loaded_sph.get_phi_segments() << " theta_segments=" << loaded_sph.get_theta_segments() << std::endl;
	std::cout << "Voxel (0,5,0) angular - first 12 segments (phi row 0):" << std::endl;
	for (size_t i = 0; i < 12; i++)
		std::cout << "  [phi=" << i << ",theta=0] = " << loaded_sph.get_segments_data()[i] << std::endl;

	EXPECT_EQ(loaded_sph.get_phi_segments(), 12);
	EXPECT_EQ(loaded_sph.get_theta_segments(), 6);
	for (size_t i = 0; i < 72; i++)
		EXPECT_FLOAT_EQ(loaded_sph.get_segments_data()[i], static_cast<float>(i));

	// Verify diagonal voxels
	std::cout << "\nDiagonal voxel (3,3,3) angular - first 6 segments:" << std::endl;
	SphericalVoxel& loaded_diag_print = loaded_channel->get_voxel<SphericalVoxel>("angular", 3, 3, 3);
	for (size_t i = 0; i < 6; i++)
		std::cout << "  [" << i << "] = " << loaded_diag_print.get_segments_data()[i] << std::endl;

	for (size_t x = 0; x < 10; x++) {
		SphericalVoxel& loaded_diag = loaded_channel->get_voxel<SphericalVoxel>("angular", x, x, x);
		for (size_t i = 0; i < 72; i++)
			EXPECT_FLOAT_EQ(loaded_diag.get_segments_data()[i], 99.f);
	}

	std::cout << "\n=== STORE/LOAD MATCH ===" << std::endl;

	// Cleanup
	std::remove(filename.c_str());
}
