#include "RadFiled3D/VoxelGrid.hpp"
#include "RadFiled3D/RadiationField.hpp"
#include <iostream>
#include "RadFiled3D/storage/RadiationFieldStore.hpp"
#include "RadFiled3D/storage/Types.hpp"
#include <memory>
#include <vector>
#include <chrono>
#include <fstream>
#include "gtest/gtest.h"
#include <cstdio>
#include <thread>
#include <shared_mutex>

using namespace RadFiled3D;
using namespace RadFiled3D::Storage;

namespace {
	class Storage : public ::testing::Test {
	protected:
		void TearDown() override {
			std::vector<std::string> files = { "test01.rf3", "test02.rf3" };
			for (auto& file : files) {
				if (std::ifstream(file).good())
					std::remove(file.c_str());
			}
		}
	};

	TEST(FieldCreationTest, Dimensions) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		EXPECT_EQ(field->get_field_dimensions(), glm::vec3(2.5f));
		EXPECT_EQ(field->get_voxel_dimensions(), glm::vec3(0.05f));
	}

	TEST(FieldCreationTest, Voxels) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		EXPECT_EQ(field->get_voxel_counts(), glm::uvec3(50));
	}

	TEST(FieldCreationTest, Copy) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));

		auto ch1 = field->add_channel("test_channel");
		field->add_channel("test_channel2");

		ch1->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");
		ch1->add_layer<float>("doserate", 25.3f, "Gy/s");
		ch1->get_voxel_flat<ScalarVoxel<float>>("doserate", 2) = 0.5f;

		std::shared_ptr<CartesianRadiationField> field2 = std::dynamic_pointer_cast<CartesianRadiationField>(field->copy());
		EXPECT_EQ(field->get_field_dimensions(), field2->get_field_dimensions());
		EXPECT_EQ(field->get_voxel_dimensions(), field2->get_voxel_dimensions());
		EXPECT_EQ(field->get_voxel_counts(), field2->get_voxel_counts());
		EXPECT_EQ(field->get_channels().size(), field2->get_channels().size());
		EXPECT_EQ(field->get_channel("test_channel")->get_layers().size(), field2->get_channel("test_channel")->get_layers().size());
		EXPECT_EQ(field->get_channel("test_channel2")->get_layers().size(), field2->get_channel("test_channel2")->get_layers().size());

		EXPECT_EQ(field->get_channel("test_channel")->get_layer_unit("doserate"), field2->get_channel("test_channel")->get_layer_unit("doserate"));
		EXPECT_EQ(field->get_channel("test_channel")->get_layer_unit("spectra"), field2->get_channel("test_channel")->get_layer_unit("spectra"));

		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", 2).get_data(), field2->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", 2).get_data());
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", 0).get_data(), field2->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", 0).get_data());
		EXPECT_NE(field->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", 0).get_data(), field2->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", 2).get_data());
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", 0).get_bins(), field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", 0).get_bins());
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", 0).get_histogram_bin_width(), field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", 0).get_histogram_bin_width());
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", 0).get_histogram()[0], field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", 0).get_histogram()[0]);
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", 0).get_histogram()[1], field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", 0).get_histogram()[1]);
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", 0).get_histogram()[2], field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", 0).get_histogram()[2]);
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", 0).get_histogram()[3], field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", 0).get_histogram()[3]);

		// check if there are no pointers linked between copy and original
		field2 = std::shared_ptr<CartesianRadiationField>(NULL);
		EXPECT_EQ(field->get_channel("test_channel")->get_layers().size(), 2);
		EXPECT_EQ(field->get_channel("test_channel2")->get_layers().size(), 0);
		EXPECT_EQ(field->get_channel("test_channel")->get_layer_unit("doserate"), "Gy/s");
		EXPECT_EQ(field->get_channel("test_channel")->get_layer_unit("spectra"), "");
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", 2).get_data(), 0.5f);
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", 1).get_data(), 25.3f);
	}

	TEST(FieldChannels, ChannelCreation) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));
		EXPECT_NO_THROW(field->get_channel("test_channel"));
	}
	
	TEST(FieldChannels, LayerCreation) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));
		EXPECT_NO_THROW(channel->add_layer<float>("doserate", 0.f, "Gy/s"));
		EXPECT_NO_THROW(channel->add_layer<uint64_t>("test2", 0));
		EXPECT_NO_THROW(channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction"));
		EXPECT_NO_THROW(channel->add_layer<glm::vec3>("dirs2", glm::vec3(0.f), "normalized direction"));

		HistogramVoxel<float> hist(26, 0.1f, nullptr);
		channel->add_custom_layer<HistogramVoxel<float>, float>("hist", hist, 0.f);
		EXPECT_NO_THROW(channel->get_layer("doserate"));
		EXPECT_NO_THROW(channel->get_layer("test2"));
		EXPECT_NO_THROW(channel->get_layer("dirs"));
		EXPECT_NO_THROW(channel->get_layer("dirs2"));
		EXPECT_NO_THROW(channel->get_layer("hist"));
		
		EXPECT_EQ(channel->get_layer_unit("doserate"), "Gy/s");
		EXPECT_EQ(channel->get_layer_unit("test2"), "");
		EXPECT_EQ(channel->get_layer_unit("dirs"), "normalized direction");
		EXPECT_EQ(channel->get_layer_unit("dirs2"), "normalized direction");
		EXPECT_EQ(channel->get_layer_unit("hist"), "");

		EXPECT_EQ(channel->get_layers().size(), 5);
	}

	TEST(Voxels, VoxelCreation) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.1f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		const float magic_number = 123.45f;

		channel->add_layer<float>("doserate", magic_number, "Gy/s");
		for (size_t x = 0; x < 10; x++)
			for (size_t y = 0; y < 10; y++)
				for (size_t z = 0; z < 10; z++) {
					ScalarVoxel<float>& vxf = channel->get_voxel<ScalarVoxel<float>>("doserate", x, y, z);
					EXPECT_EQ(vxf.get_data(), magic_number);
				}

		float* data_buffer = channel->get_layer<float>("doserate");
		for (size_t i = 0; i < 1000; i++)
			EXPECT_EQ(data_buffer[i], magic_number);

		const glm::vec3 magic_vector = glm::vec3(1.f, 2.f, 3.f);

		channel->add_layer<glm::vec3>("dir", magic_vector, "");
		for (size_t x = 0; x < 10; x++)
			for (size_t y = 0; y < 10; y++)
				for (size_t z = 0; z < 10; z++) {
					ScalarVoxel<glm::vec3>& vxf = channel->get_voxel<ScalarVoxel<glm::vec3>>("dir", x, y, z);
					EXPECT_EQ(vxf.get_data(), magic_vector);
				}

		glm::vec3* data_vec_buffer = channel->get_layer<glm::vec3>("dir");
		for (size_t i = 0; i < 1000; i++)
			EXPECT_EQ(data_vec_buffer[i], magic_vector);
	}

	TEST(Voxels, VoxelAccess) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<float>("doserate", 0.f, "Gy/s");
		for (size_t i = 0; i < 10; i++)
			EXPECT_EQ(static_cast<ScalarVoxel<float>&>(channel->get_voxel("doserate", 0, i, 0)).get_data(), 0.f);
	}

	TEST(Voxels, VoxelModification) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<float>("doserate", 25.3f, "Gy/s");
		ScalarVoxel<float>& vxf = channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0);
		vxf = 13.5f;
		for (size_t i = 0; i < 10; i++)
			EXPECT_TRUE((i == 5 && static_cast<ScalarVoxel<float>&>(channel->get_voxel("doserate", 0, i, 0)).get_data() == 13.5f) || static_cast<ScalarVoxel<float>&>(channel->get_voxel("doserate", 0, i, 0)).get_data() == 25.3f);
	}

	TEST(Voxels, VoxelModificationVec3) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		ScalarVoxel<glm::vec3>& vxd = static_cast<ScalarVoxel<glm::vec3>&>(channel->get_voxel("dirs", 0, 5, 0));
		vxd = glm::vec3(1.f, 2.f, 3.f);
		vxd = static_cast<ScalarVoxel<glm::vec3>&>(channel->get_voxel("dirs", 0, 5, 0));
		EXPECT_EQ(vxd.get_data().x, 1.f);
		EXPECT_EQ(vxd.get_data().y, 2.f);
		EXPECT_EQ(vxd.get_data().z, 3.f);
	}

	TEST(Voxels, VoxelModificationHistograms) {
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

		metadata->add_dynamic_metadata<float>("test_float", 1.234f);
		metadata->add_dynamic_metadata<int>("test_int", 1234);
		metadata->set_dynamic_custom_metadata<HistogramVoxel<float>>("test_hist", HistogramVoxel<float>(10, 1.f, nullptr));
		metadata->get_dynamic_metadata<HistogramVoxel<float>>("test_hist").get_histogram()[0] = 1.f;

		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.1f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		const float magic_number = 0.134f;
		const float magic_diagonal_number = std::sqrt(3.f) * magic_number;

		channel->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), magic_number, "");
		HistogramVoxel<float>& hist1 = channel->get_voxel<HistogramVoxel<float>>("spectra", 0, 5, 0);
		for (size_t i = 0; i < 26; i++)
			hist1.get_histogram()[i] = static_cast<float>(i);

		for (size_t i = 0; i < 26; i++)
			EXPECT_EQ(hist1.get_histogram()[i], static_cast<float>(i));

		for (size_t x = 0; x < 10; x++)
		{
			HistogramVoxel<float>& hist = channel->get_voxel<HistogramVoxel<float>>("spectra", x, x, x);
			for (size_t i = 0; i < 26; i++)
				hist.get_histogram()[i] = magic_diagonal_number;
		}

		for (size_t x = 0; x < 10; x++)
			for (size_t y = 0; y < 10; y++)
				for (size_t z = 0; z < 10; z++) {
					HistogramVoxel<float>& hist = channel->get_voxel<HistogramVoxel<float>>("spectra", x, y, z);
					if (x == y && x == z) {
						for (size_t i = 0; i < 26; i++)
							EXPECT_EQ(hist.get_histogram()[i], magic_diagonal_number);
					}
					else {
						if (y == 5 && x == 0 && z == 0) {
							for (size_t i = 0; i < 26; i++)
								EXPECT_EQ(hist.get_histogram()[i], static_cast<float>(i));
						}
						else {
							for (size_t i = 0; i < 26; i++)
								EXPECT_EQ(hist.get_histogram()[i], magic_number);
						}
					}
				}

		FieldStore::store(field, metadata, "test_hist.rf3", StoreVersion::V1);
		std::shared_ptr<CartesianRadiationField> loaded_field = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test_hist.rf3"));
		std::shared_ptr<VoxelGridBuffer> loaded_channel = std::static_pointer_cast<VoxelGridBuffer>(loaded_field->get_channel("test_channel"));
		HistogramVoxel<float>& loaded_hist1 = loaded_channel->get_voxel<HistogramVoxel<float>>("spectra", 0, 5, 0);

		for (size_t x = 0; x < 10; x++)
			for (size_t y = 0; y < 10; y++)
				for (size_t z = 0; z < 10; z++) {
					HistogramVoxel<float>& hist = loaded_channel->get_voxel<HistogramVoxel<float>>("spectra", x, y, z);
					if (x == y && x == z) {
						for (size_t i = 0; i < 26; i++)
							EXPECT_EQ(hist.get_histogram()[i], magic_diagonal_number);
					}
					else {
						if (y == 5 && x == 0 && z == 0) {
							for (size_t i = 0; i < 26; i++)
								EXPECT_EQ(hist.get_histogram()[i], static_cast<float>(i));
						}
						else {
							for (size_t i = 0; i < 26; i++)
								EXPECT_EQ(hist.get_histogram()[i], magic_number);
						}
					}
				}
	}

	TEST(Voxels, VoxelBufferOperators) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.1f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		const float magic_number = 0.134f;

		channel->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), magic_number, "");
		HistogramVoxel<float>& hist1 = channel->get_voxel<HistogramVoxel<float>>("spectra", 0, 5, 0);
		for (size_t i = 0; i < 26; i++)
			hist1.get_histogram()[i] = static_cast<float>(i);

		channel->add_layer<float>("doserate", magic_number, "Gy/s");

		// check channel integrity, if all values are correctly set

		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(channel->get_voxel<HistogramVoxel<float>>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i));

		VoxelGridBuffer* original_channel = static_cast<VoxelGridBuffer*>(channel->copy());

		// check integrity of original channel
		EXPECT_FLOAT_EQ(original_channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(original_channel->get_voxel<HistogramVoxel<float>>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i));

		// check if the channel is still integer after the copy
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(channel->get_voxel<HistogramVoxel<float>>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i));

		*channel += *original_channel;

		// check if original channel is still the same after the operation
		EXPECT_FLOAT_EQ(original_channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(original_channel->get_voxel<HistogramVoxel<float>>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i));

		// check if the channel is correctly modified
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), 2.f * magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(channel->get_voxel<HistogramVoxel<float>>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i) * 2.f);

		*channel -= *original_channel;
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(channel->get_voxel<HistogramVoxel<float>>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i));

		*channel *= *original_channel;
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), magic_number * magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(channel->get_voxel<HistogramVoxel<float>>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i) * static_cast<float>(i));

		*channel /= *channel;
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), 1.f);
		EXPECT_TRUE(channel->get_voxel<HistogramVoxel<float>>("spectra", 0, 5, 0).get_histogram()[0] != channel->get_voxel<HistogramVoxel<float>>("spectra", 0, 5, 0).get_histogram()[0]); // check for nan
		for (size_t i = 1; i < 26; i++)
			EXPECT_FLOAT_EQ(channel->get_voxel<HistogramVoxel<float>>("spectra", 0, 5, 0).get_histogram()[i], 1.f);
	}

	TEST(Storage, Store) {
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

		EXPECT_NO_THROW(FieldStore::store(field, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test01.rf3", StoreVersion::V1));
	}

	TEST(Storage, Load) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_layer<float>("doserate", 25.3f, "Gy/s");

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
		metadata->add_dynamic_metadata<uint8_t>("test_byte", static_cast<uint8_t>(5));

		EXPECT_NO_THROW(FieldStore::store(field, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test02.rf3", StoreVersion::V1));

		std::shared_ptr<CartesianRadiationField> field2 = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test02.rf3"));
		auto metadata2 = std::static_pointer_cast<RadFiled3D::Storage::V1::RadiationFieldMetadata>(FieldStore::peek_metadata("test02.rf3"))->get_header();
		EXPECT_EQ(field2->get_field_dimensions(), field->get_field_dimensions());
		EXPECT_EQ(field2->get_voxel_dimensions(), field->get_voxel_dimensions());
		EXPECT_EQ(field2->get_voxel_counts(), field->get_voxel_counts());
		EXPECT_EQ(field2->get_channel("test_channel")->get_layer_unit("doserate"), "Gy/s");
		EXPECT_EQ(field2->get_channel("test_channel")->get_layer_unit("dirs"), "normalized direction");
		EXPECT_EQ(static_cast<VoxelGridBuffer*>(field2->get_channel("test_channel").get())->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), 25.3f);

		EXPECT_EQ(metadata2.simulation.primary_particle_count, 100);
		EXPECT_EQ(metadata2.simulation.tube.max_energy_eV, 100.f);
		EXPECT_EQ(metadata2.simulation.tube.radiation_direction, glm::vec3(1.f, 0.f, 0.f));
		EXPECT_EQ(metadata2.simulation.tube.radiation_origin, glm::vec3(0.f, 0.f, 0.f));
		EXPECT_EQ(strcmp(metadata2.simulation.tube.tube_id, "XRayTube"), 0);
		EXPECT_EQ(strcmp(metadata2.simulation.geometry, "geom"), 0);
		EXPECT_EQ(strcmp(metadata2.simulation.physics_list, "FTFP_BERT"), 0);

		auto metadata_full = FieldStore::load_metadata("test02.rf3");
		uint8_t test_byte = std::static_pointer_cast<RadFiled3D::Storage::V1::RadiationFieldMetadata>(metadata_full)->get_dynamic_metadata<ScalarVoxel<uint8_t>>("test_byte").get_data();
		EXPECT_EQ(test_byte, 5);
		
		auto channels = field2->get_channels();
		EXPECT_EQ(channels.size(), 1);
		for (auto& layer : channels) {
			auto layers = layer.second->get_layers();
			EXPECT_EQ(layers.size(), 2);
			EXPECT_EQ(layer.second->get_layer_unit("doserate"), "Gy/s");
			EXPECT_EQ(layer.second->get_layer_unit("dirs"), "normalized direction");
		}

		for (size_t vx = 0; vx < field2->get_voxel_counts().x * field2->get_voxel_counts().y * field2->get_voxel_counts().z; vx++) {
			float val1 = field->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", vx).get_data();
			float val2 = field2->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", vx).get_data();
			EXPECT_EQ(val1, val2);
		}
	}

	TEST(Storage, LoadHistogram) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");
		channel->add_layer<float>("doserate", 10.f, "Gy/s");

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

		EXPECT_NO_THROW(FieldStore::store(field, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test03.rf3", StoreVersion::V1));

		std::shared_ptr<CartesianRadiationField> field2 = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test03.rf3"));
		auto metadata2 = FieldStore::load_metadata("test03.rf3");
		EXPECT_EQ(field2->get_field_dimensions(), glm::vec3(2.5f));
		EXPECT_EQ(field2->get_voxel_dimensions(), glm::vec3(0.05f));
		EXPECT_EQ(field2->get_voxel_counts(), glm::uvec3(50));
		for (size_t i = 0; i < field2->get_voxel_counts().x * field2->get_voxel_counts().y * field2->get_voxel_counts().z; i++) {
			size_t bins = field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", i).get_bins();
			float width = field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel<float>>("spectra", i).get_histogram_bin_width();
			EXPECT_EQ(bins, 26);
			EXPECT_EQ(width, 10.f);
		}
	}

	// Round-trips a field whose scalar-float, glm::vec3 and histogram layers all carry distinct,
	// non-default per-voxel values, then reloads and checks every value survived. The existing Store/
	// Load/LoadHistogram tests only assert layer layout and default contents; this covers the
	// create -> modify -> store -> load path that the simulator depends on for its recorded values.
	// (Angular-resolved / spherical voxels are exercised by SphericalTest, so they are omitted here.)
	TEST(Storage, LoadModifiedLayers) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.1f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("mixed"));

		channel->add_layer<float>("doserate", 0.f, "Gy/s");
		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(16, 5.f, nullptr), 0.f, "keV");

		// Write distinct values into two interior voxels so per-voxel independence is covered too.
		channel->get_voxel<ScalarVoxel<float>>("doserate", 2, 3, 4) = 12.5f;
		channel->get_voxel<ScalarVoxel<glm::vec3>>("dirs", 2, 3, 4).get_data() = glm::vec3(0.1f, 0.2f, 0.3f);
		HistogramVoxel<float>& hist = channel->get_voxel<HistogramVoxel<float>>("spectra", 2, 3, 4);
		for (size_t i = 0; i < hist.get_bins(); i++)
			hist.get_histogram()[i] = static_cast<float>(i);

		channel->get_voxel<ScalarVoxel<float>>("doserate", 7, 1, 8) = -3.5f;
		channel->get_voxel<ScalarVoxel<glm::vec3>>("dirs", 7, 1, 8).get_data() = glm::vec3(1.f, -2.f, 4.f);

		std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				100, "geom", "FTFP_BERT",
				RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(1.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 0.f), 100.f, "XRayTube")),
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software("test", "1.0", "repo", "commit"));

		EXPECT_NO_THROW(FieldStore::store(field, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", StoreVersion::V1));

		std::shared_ptr<CartesianRadiationField> field2 = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test05.rf3"));
		ASSERT_NE(field2, nullptr);
		EXPECT_EQ(field2->get_voxel_counts(), glm::uvec3(10));

		std::shared_ptr<VoxelGridBuffer> channel2 = std::static_pointer_cast<VoxelGridBuffer>(field2->get_channel("mixed"));
		EXPECT_EQ(channel2->get_layers().size(), 3);
		EXPECT_EQ(channel2->get_layer_unit("doserate"), "Gy/s");
		EXPECT_EQ(channel2->get_layer_unit("dirs"), "normalized direction");

		// Scalar float + vec3 survived with their exact per-voxel values.
		EXPECT_EQ(channel2->get_voxel<ScalarVoxel<float>>("doserate", 2, 3, 4).get_data(), 12.5f);
		EXPECT_EQ(channel2->get_voxel<ScalarVoxel<glm::vec3>>("dirs", 2, 3, 4).get_data(), glm::vec3(0.1f, 0.2f, 0.3f));
		EXPECT_EQ(channel2->get_voxel<ScalarVoxel<float>>("doserate", 7, 1, 8).get_data(), -3.5f);
		EXPECT_EQ(channel2->get_voxel<ScalarVoxel<glm::vec3>>("dirs", 7, 1, 8).get_data(), glm::vec3(1.f, -2.f, 4.f));

		// Histogram layout + per-bin contents survived.
		HistogramVoxel<float>& hist2 = channel2->get_voxel<HistogramVoxel<float>>("spectra", 2, 3, 4);
		EXPECT_EQ(hist2.get_bins(), 16);
		EXPECT_EQ(hist2.get_histogram_bin_width(), 5.f);
		for (size_t i = 0; i < hist2.get_bins(); i++)
			EXPECT_EQ(hist2.get_histogram()[i], static_cast<float>(i));

		// An untouched voxel must round-trip as the layer default.
		EXPECT_EQ(channel2->get_voxel<ScalarVoxel<float>>("doserate", 0, 0, 0).get_data(), 0.f);

		std::remove("test05.rf3");
	}

	TEST(Storage, MetadataHists) {
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
		{
			size_t bins = 150;
			float bin_width = 4.6f;
			RadFiled3D::HistogramVoxel<float> hist(bins, bin_width, nullptr);
			metadata->set_dynamic_custom_metadata<RadFiled3D::HistogramVoxel<float>>("test_hist", hist);
			RadFiled3D::HistogramVoxel<float>& new_hist = *static_cast<RadFiled3D::HistogramVoxel<float>*>(metadata->get_dynamic_metadata().at("test_hist"));
			for (size_t i = 0; i < 10; i++)
				new_hist.get_histogram()[i] = 1.f * i;
			EXPECT_EQ(new_hist.get_bins(), bins);
			EXPECT_EQ(new_hist.get_histogram_bin_width(), bin_width);
		}

		RadFiled3D::HistogramVoxel<float>* voxel = static_cast<RadFiled3D::HistogramVoxel<float>*>(metadata->get_dynamic_metadata().at("test_hist"));

		for (size_t i = 0; i < 10; i++)
			EXPECT_FLOAT_EQ(voxel->get_histogram()[i], 1.f * i);
	}

	TEST(Storage, JoinFields) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");
		channel->add_layer<float>("doserate", 15.f, "Gy/s");

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

		EXPECT_NO_THROW(FieldStore::store(field, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test04.rf3", StoreVersion::V1));

		std::shared_ptr<CartesianRadiationField> field2 = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel2 = std::static_pointer_cast<VoxelGridBuffer>(field2->add_channel("test_channel"));
		channel2->add_layer<float>("doserate", 10.f, "Gy/s");
		channel2->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel2->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");

		ScalarVoxel<float>& vx1 = channel->get_voxel_flat<ScalarVoxel<float>>("doserate", 0);
		ScalarVoxel<float>& vx2 = channel2->get_voxel_flat<ScalarVoxel<float>>("doserate", 0);
		const float combined_doserate = vx1.get_data() + vx2.get_data();

		auto metadata2 = FieldStore::peek_metadata("test04.rf3");

		EXPECT_NO_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test04.rf3", FieldJoinMode::Add));

		metadata = std::dynamic_pointer_cast<RadFiled3D::Storage::V1::RadiationFieldMetadata>(FieldStore::load_metadata("test04.rf3"));
		// primary_particle_count is a uint64_t inside a #pragma pack(4) header, so it can sit at a
		// 4-byte-aligned offset. Read it into an aligned temporary before comparing: binding EXPECT_EQ's
		// const uint64_t& straight to the packed field is a misaligned reference (UBSan alignment).
		EXPECT_EQ(static_cast<uint64_t>(metadata->get_header().simulation.primary_particle_count), 200u);
		EXPECT_EQ(metadata->get_header().simulation.tube.max_energy_eV, 100.f);
		EXPECT_EQ(metadata->get_header().simulation.tube.radiation_direction, glm::vec3(1.f, 0.f, 0.f));
		EXPECT_EQ(metadata->get_header().simulation.tube.radiation_origin, glm::vec3(0.f, 0.f, 0.f));

		std::shared_ptr<CartesianRadiationField> field3 = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test04.rf3"));

		for (size_t i = 0; i < field2->get_voxel_counts().x * field3->get_voxel_counts().y * field3->get_voxel_counts().y; i++) {
			EXPECT_EQ(field3->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", i).get_data(), combined_doserate);
		}

		metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
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

		EXPECT_NO_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test04.rf3", FieldJoinMode::AddWeighted));

		vx1 = field3->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", 0);
		vx2 = channel2->get_voxel_flat<ScalarVoxel<float>>("doserate", 0);
		const float combined_doserate2 = vx1.get_data() * 2/3 + vx2.get_data() * 1/3;

		std::shared_ptr<CartesianRadiationField> field4 = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test04.rf3"));
		metadata = std::dynamic_pointer_cast<RadFiled3D::Storage::V1::RadiationFieldMetadata>(FieldStore::load_metadata("test04.rf3"));
		// packed uint64_t (see the note above): compare via an aligned temporary, not a reference bind.
		EXPECT_EQ(static_cast<uint64_t>(metadata->get_header().simulation.primary_particle_count), 300u);
		EXPECT_EQ(metadata->get_header().simulation.tube.max_energy_eV, 100.f);
		EXPECT_EQ(metadata->get_header().simulation.tube.radiation_direction, glm::vec3(1.f, 0.f, 0.f));
		EXPECT_EQ(metadata->get_header().simulation.tube.radiation_origin, glm::vec3(0.f, 0.f, 0.f));

		for (size_t i = 0; i < field2->get_voxel_counts().x * field4->get_voxel_counts().y * field4->get_voxel_counts().y; i++) {
			EXPECT_EQ(field4->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", i).get_data(), combined_doserate2);
		}
	}

	std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> make_join_test_metadata() {
		return std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
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
	}

	TEST(Storage, JoinByteLayersSaturate) {
		auto make_field = [](uint8_t mask_value, char offset_value) {
			auto field = std::make_shared<CartesianRadiationField>(glm::vec3(0.2f), glm::vec3(0.05f));
			auto channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("geometry"));
			channel->add_layer<uint8_t>("mask", mask_value, "");
			channel->add_layer<char>("offset", offset_value, "");
			return field;
		};
		std::remove("test08.rf3");
		auto metadata = std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(make_join_test_metadata());

		auto first = make_field(200, 100);
		first->get_channel("geometry")->get_voxel_flat<ScalarVoxel<uint8_t>>("mask", 1) = static_cast<uint8_t>(0);
		EXPECT_NO_THROW(FieldStore::store(first, metadata, "test08.rf3", StoreVersion::V1));
		EXPECT_NO_THROW(FieldStore::join(make_field(255, 100), metadata, "test08.rf3", FieldJoinMode::Add));

		auto joined = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test08.rf3"));
		auto geometry = joined->get_channel("geometry");
		EXPECT_EQ(geometry->get_voxel_flat<ScalarVoxel<uint8_t>>("mask", 0).get_data(), 255);
		EXPECT_EQ(geometry->get_voxel_flat<ScalarVoxel<uint8_t>>("mask", 1).get_data(), 255);
		EXPECT_EQ(geometry->get_voxel_flat<ScalarVoxel<char>>("offset", 0).get_data(), std::numeric_limits<char>::max());
		std::remove("test08.rf3");
	}

	TEST(Storage, JoinTakesOverLayersMissingInTarget) {
		std::remove("test09.rf3");
		auto metadata = std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(make_join_test_metadata());

		auto existing = std::make_shared<CartesianRadiationField>(glm::vec3(0.2f), glm::vec3(0.05f));
		std::static_pointer_cast<VoxelGridBuffer>(existing->add_channel("radiation"))->add_layer<float>("flux", 1.f, "");
		EXPECT_NO_THROW(FieldStore::store(existing, metadata, "test09.rf3", StoreVersion::V1));

		auto additional = std::make_shared<CartesianRadiationField>(glm::vec3(0.2f), glm::vec3(0.05f));
		std::static_pointer_cast<VoxelGridBuffer>(additional->add_channel("radiation"))->add_layer<float>("flux", 2.f, "");
		auto geometry = std::static_pointer_cast<VoxelGridBuffer>(additional->add_channel("geometry"));
		geometry->add_layer<uint8_t>("Patient", 0, "");
		const size_t marked_voxel = geometry->get_voxel_count() - 1;
		geometry->get_voxel_flat<ScalarVoxel<uint8_t>>("Patient", marked_voxel) = static_cast<uint8_t>(255);
		// voxel 0 differs from the rest, so a layer seeded from voxel 0 would show up as wrong values
		std::static_pointer_cast<VoxelGridBuffer>(additional->get_channel("radiation"))->add_layer<float>("extra", 0.f, "");
		additional->get_channel("radiation")->get_voxel_flat<ScalarVoxel<float>>("extra", 0) = 7.f;

		EXPECT_THROW(FieldStore::join(additional, metadata, "test09.rf3", FieldJoinMode::Add, FieldJoinCheckMode::FieldStructureOnly), RadiationFieldStoreException);
		EXPECT_NO_THROW(FieldStore::join(additional, metadata, "test09.rf3", FieldJoinMode::Add, FieldJoinCheckMode::FieldUnitsOnly));

		auto joined = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test09.rf3"));
		auto joined_geometry = joined->get_channel("geometry");
		for (size_t i = 0; i < joined_geometry->get_voxel_count(); i++)
			EXPECT_EQ(joined_geometry->get_voxel_flat<ScalarVoxel<uint8_t>>("Patient", i).get_data(), i == marked_voxel ? 255 : 0);
		auto radiation = joined->get_channel("radiation");
		EXPECT_EQ(radiation->get_voxel_flat<ScalarVoxel<float>>("flux", 3).get_data(), 3.f);
		EXPECT_EQ(radiation->get_voxel_flat<ScalarVoxel<float>>("extra", 0).get_data(), 7.f);
		EXPECT_EQ(radiation->get_voxel_flat<ScalarVoxel<float>>("extra", 1).get_data(), 0.f);
		std::remove("test09.rf3");
	}

	TEST(Storage, ReplaceKeepsOtherChannels) {
		std::remove("test10.rf3");
		std::remove("test10_expected.rf3");
		auto metadata = make_join_test_metadata();

		auto make_radiation = [](std::shared_ptr<CartesianRadiationField> field, float flux) {
			auto channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("scatter_field"));
			channel->add_layer<float>("flux", flux, "counts / primary_particles");
			channel->add_custom_layer<HistogramVoxel<float>>("spectrum", HistogramVoxel<float>(8, 10.f, nullptr), flux, "eV");
			std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("direct_beam"))->add_layer<float>("flux", 2.f * flux, "counts / primary_particles");
		};
		auto add_geometry = [](std::shared_ptr<CartesianRadiationField> field) {
			auto channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("geometry"));
			channel->add_layer<uint8_t>("Patient", 0, "occupancy");
			channel->add_layer<uint8_t>("Shield", 255, "occupancy");
			channel->get_voxel_flat<ScalarVoxel<uint8_t>>("Patient", 5) = static_cast<uint8_t>(255);
		};

		auto first = std::make_shared<CartesianRadiationField>(glm::vec3(0.2f), glm::vec3(0.05f));
		make_radiation(first, 1.f);
		add_geometry(first);
		FieldStore::replace(first, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test10.rf3");

		auto update = std::make_shared<CartesianRadiationField>(glm::vec3(0.2f), glm::vec3(0.05f));
		make_radiation(update, 5.f);
		auto updated_metadata = make_join_test_metadata();
		auto header = updated_metadata->get_header();
		header.simulation.primary_particle_count = 500;
		updated_metadata->set_header(header);
		EXPECT_NO_THROW(FieldStore::replace(update, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(updated_metadata), "test10.rf3"));

		auto expected = std::make_shared<CartesianRadiationField>(glm::vec3(0.2f), glm::vec3(0.05f));
		make_radiation(expected, 5.f);
		add_geometry(expected);
		FieldStore::store(expected, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(updated_metadata), "test10_expected.rf3");
		auto read_all = [](const std::string& file) {
			std::ifstream stream(file, std::ios::binary);
			return std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
		};
		EXPECT_EQ(read_all("test10.rf3"), read_all("test10_expected.rf3"));

		auto loaded = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test10.rf3"));
		EXPECT_EQ(loaded->get_channel("scatter_field")->get_voxel_flat<ScalarVoxel<float>>("flux", 3).get_data(), 5.f);
		EXPECT_EQ(loaded->get_channel("geometry")->get_voxel_flat<ScalarVoxel<uint8_t>>("Patient", 5).get_data(), 255);
		EXPECT_EQ(loaded->get_channel("geometry")->get_voxel_flat<ScalarVoxel<uint8_t>>("Patient", 4).get_data(), 0);
		EXPECT_EQ(static_cast<uint64_t>(std::dynamic_pointer_cast<RadFiled3D::Storage::V1::RadiationFieldMetadata>(FieldStore::load_metadata("test10.rf3"))->get_header().simulation.primary_particle_count), 500u);

		auto other_grid = std::make_shared<CartesianRadiationField>(glm::vec3(0.2f), glm::vec3(0.1f));
		make_radiation(other_grid, 9.f);
		EXPECT_THROW(FieldStore::replace(other_grid, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test10.rf3"), RadiationFieldStoreException);
		EXPECT_EQ(read_all("test10.rf3"), read_all("test10_expected.rf3"));

		std::remove("test10.rf3");
		std::remove("test10_expected.rf3");
	}

	TEST(SaturatingArithmetic, ClampsInsteadOfWrapping) {
		EXPECT_EQ(SaturatingArithmetic::add<uint8_t>(200, 100), 255);
		EXPECT_EQ(SaturatingArithmetic::subtract<uint8_t>(5, 10), 0);
		EXPECT_EQ(SaturatingArithmetic::multiply<uint8_t>(16, 16), 255);
		EXPECT_EQ(SaturatingArithmetic::divide<uint8_t>(7, 0), 255);
		EXPECT_EQ(SaturatingArithmetic::mean<uint8_t>(255, 255), 255);
		EXPECT_EQ(SaturatingArithmetic::blend<uint8_t>(255, 255, 0.3f), 255);
		EXPECT_EQ(SaturatingArithmetic::add<int8_t>(-100, -100), -128);
		EXPECT_EQ(SaturatingArithmetic::subtract<int8_t>(100, -100), 127);
		EXPECT_EQ(SaturatingArithmetic::multiply<int8_t>(-100, 2), -128);
		EXPECT_EQ(SaturatingArithmetic::multiply<int8_t>(-100, -2), 127);
		EXPECT_EQ(SaturatingArithmetic::divide<int8_t>(-128, -1), 127);
		EXPECT_EQ(SaturatingArithmetic::mean<int8_t>(3, -4), 0);
		EXPECT_EQ(SaturatingArithmetic::add<int>(std::numeric_limits<int>::max(), 1), std::numeric_limits<int>::max());
		EXPECT_EQ(SaturatingArithmetic::add<uint64_t>(std::numeric_limits<uint64_t>::max(), 1), std::numeric_limits<uint64_t>::max());
		EXPECT_EQ(SaturatingArithmetic::multiply<int64_t>(std::numeric_limits<int64_t>::min(), -1), std::numeric_limits<int64_t>::max());
		EXPECT_EQ(SaturatingArithmetic::add<uint8_t>(3, 4), 7);
		EXPECT_EQ(SaturatingArithmetic::multiply<int>(-6, 7), -42);
	}

	TEST(Storage, JoinFieldsChecks) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");
		channel->add_layer<float>("doserate", 15.f, "Gy/s");

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

		EXPECT_NO_THROW(FieldStore::store(field, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", StoreVersion::V1));

		std::shared_ptr<CartesianRadiationField> field2 = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel2 = std::static_pointer_cast<VoxelGridBuffer>(field2->add_channel("test_channel"));
		channel2->add_layer<float>("doserate", 10.f, "Gy/s");
		channel2->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel2->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");

		auto metadata2 = FieldStore::peek_metadata("test05.rf3");

		metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				101,
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

		EXPECT_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::Strict), RadiationFieldStoreException);
		EXPECT_NO_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::MetadataSimulationSimilar));

		metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				101,
				"other geom",
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
		EXPECT_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::MetadataSimulationSimilar), RadiationFieldStoreException);
		EXPECT_NO_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::MetadataSoftwareEqual));

		metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				101,
				"other geom",
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
				"1.2",
				"repo",
				"commit"
			)
		);
		EXPECT_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::MetadataSoftwareEqual), RadiationFieldStoreException);
		EXPECT_NO_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::MetadataSoftwareSimilar));

		metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				101,
				"other geom",
				"FTFP_BERT",
				RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(1.f, 0.f, 0.f),
					glm::vec3(0.f, 0.f, 0.f),
					100.f,
					"XRayTube"
				)
			),
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
				"test2",
				"1.2",
				"repo",
				"commit3"
			)
		);
		EXPECT_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::MetadataSoftwareSimilar), RadiationFieldStoreException);
		EXPECT_NO_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::FieldStructureOnly));

		field2->add_channel("other channel");
		EXPECT_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::FieldStructureOnly), RadiationFieldStoreException);
		EXPECT_NO_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::FieldUnitsOnly));

		std::shared_ptr<CartesianRadiationField> field3 = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		field3->add_channel("other channel than ever");
		std::shared_ptr<VoxelGridBuffer> channel3 = std::static_pointer_cast<VoxelGridBuffer>(field3->add_channel("test_channel"));
		channel3->add_layer<float>("doserate", 10.f, "Gy");
		channel3->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel3->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");
		EXPECT_THROW(FieldStore::join(field3, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::FieldUnitsOnly), RadiationFieldStoreException);
		EXPECT_NO_THROW(FieldStore::join(field3, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::NoChecks));
	}

	TEST(Storage, LoadSingleLayer) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");
		channel->add_layer<float>("doserate", 15.f, "Gy/s");

		std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				1,
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

		EXPECT_NO_THROW(FieldStore::store(field, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test06.rf3", StoreVersion::V1));

		std::ifstream file("test06.rf3");
		auto type = FieldStore::peek_field_type(file);
		EXPECT_EQ(type, FieldType::Cartesian);
		auto layer = FieldStore::load_single_layer(file, "test_channel", "doserate");
		EXPECT_EQ(layer->get_unit(), "Gy/s");
		EXPECT_EQ(layer->get_voxel_count(), 125000);
		for (size_t i = 0; i < layer->get_voxel_count(); i++) {
			float val = layer->get_voxel_flat<ScalarVoxel<float>>(i).get_data();
			EXPECT_FLOAT_EQ(val, 15.f);
		}
	}

	// TODO: Fix synchronization test. But currently I do not use this feature.
	/*TEST(Storage, JoinFieldsSynchronization) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");
		channel->add_layer<float>("doserate", 15.f, "Gy/s");

		std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				1,
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

		FieldStore::enable_file_lock_synchronization(true);
		EXPECT_NO_THROW(FieldStore::store(field, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test07.rf3", StoreVersion::V1));
		EXPECT_NO_THROW(FieldStore::join(field, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test07.rf3", FieldJoinMode::Add, FieldJoinCheckMode::Strict));

		metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				0,
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
		EXPECT_NO_THROW(FieldStore::store(field, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test07.rf3", StoreVersion::V1));
		// launch 10 threads that will try to join the field at the same time
		std::vector<std::thread> threads;
		metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				1,
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

		EXPECT_EQ(metadata->get_header().simulation.primary_particle_count, 1);

		for (size_t i = 0; i < 10; i++) {
			threads.push_back(std::thread([&]() {
				size_t a = 0;
				while (a < 10) {
					std::shared_ptr<CartesianRadiationField> field2 = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
					std::shared_ptr<VoxelGridBuffer> channel2 = std::static_pointer_cast<VoxelGridBuffer>(field2->add_channel("test_channel"));
					channel2->add_layer<float>("doserate", 10.f, "Gy/s");
					channel2->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
					channel2->add_custom_layer<HistogramVoxel<float>>("spectra", HistogramVoxel<float>(26, 10.f, nullptr), .123f, "");
					EXPECT_NO_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test07.rf3", FieldJoinMode::Add, FieldJoinCheckMode::MetadataSimulationSimilar));
					a++;
				}
			}));
		}

		for (auto& thread : threads) {
			thread.join();
		}

		metadata = std::dynamic_pointer_cast<RadFiled3D::Storage::V1::RadiationFieldMetadata>(FieldStore::load_metadata("test07.rf3"));
		EXPECT_EQ(metadata->get_header().simulation.primary_particle_count, 100);
	}*/
}

TEST(Storage, MissingFileReportsItself) {
	const std::string missing = "this_file_does_not_exist.rf3";

	// The path-taking importers used to `throw new ...`, i.e. throw a pointer, which
	// no `catch (const RadiationFieldStoreException&)` could ever match.
	EXPECT_THROW(FieldStore::load(missing), RadiationFieldStoreException);
	EXPECT_THROW(FieldStore::load_metadata(missing), RadiationFieldStoreException);
	EXPECT_THROW(FieldStore::peek_metadata(missing), RadiationFieldStoreException);
	EXPECT_THROW(FieldStore::get_store_version(missing), RadiationFieldStoreException);
	EXPECT_THROW(FieldStore::construct_accessor(missing), RadiationFieldStoreException);

	// ... and the message must name the file instead of blaming its (empty) contents.
	try {
		FieldStore::load(missing);
		FAIL() << "expected a RadiationFieldStoreException";
	}
	catch (const RadiationFieldStoreException& e) {
		const std::string what = e.what();
		EXPECT_NE(what.find(missing), std::string::npos) << what;
		EXPECT_NE(what.find("does not exist"), std::string::npos) << what;
		EXPECT_EQ(what.find("version string"), std::string::npos) << what;
	}
}

TEST(Storage, MissingDynamicMetadataKeyReportsItself) {
	auto metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>();
	metadata->add_dynamic_metadata<float>("known_key", 1.f);

	try {
		metadata->get_dynamic_metadata<ScalarVoxel<float>>("absent_key");
		FAIL() << "expected a RadiationFieldStoreException";
	}
	catch (const RadiationFieldStoreException& e) {
		const std::string what = e.what();
		EXPECT_NE(what.find("absent_key"), std::string::npos) << what;
		EXPECT_NE(what.find("known_key"), std::string::npos) << what;  // lists what IS there
	}

	EXPECT_NO_THROW(metadata->get_dynamic_metadata<ScalarVoxel<float>>("known_key"));
}

TEST(Storage, NonRadFiled3DFileIsRejected) {
	// A file that exists but is not a field file must say exactly that, rather than
	// failing later on its contents ("Found version string: '<binary>' was invalid!").
	const std::string foreign = "not_a_field_file.bin";
	{
		std::ofstream out(foreign, std::ios::binary);
		const unsigned char junk[] = { 0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a, 0xb0, 0x00, 0x1f, 0x7f, 0x42 };
		out.write((const char*)junk, sizeof(junk));
	}

	try {
		FieldStore::load(foreign);
		FAIL() << "expected a RadiationFieldStoreException";
	}
	catch (const RadiationFieldStoreException& e) {
		const std::string what = e.what();
		EXPECT_NE(what.find("is not a RadFiled3D (.rf3) file"), std::string::npos) << what;
		EXPECT_NE(what.find(foreign), std::string::npos) << what;
		// the raw bytes must be escaped, never pasted verbatim into the message
		EXPECT_NE(what.find("\\x89"), std::string::npos) << what;
		for (char c : what)
			EXPECT_TRUE(static_cast<unsigned char>(c) < 0x80) << "message must stay printable ASCII: " << what;
	}

	EXPECT_THROW(FieldStore::load_metadata(foreign), RadiationFieldStoreException);
	EXPECT_THROW(FieldStore::get_store_version(foreign), RadiationFieldStoreException);
	EXPECT_THROW(FieldStore::construct_accessor(foreign), RadiationFieldStoreException);

	// a file too short to even hold the header
	const std::string truncated = "truncated_field_file.rf3";
	{
		std::ofstream out(truncated, std::ios::binary);
		out << "1.1";
	}
	try {
		FieldStore::load(truncated);
		FAIL() << "expected a RadiationFieldStoreException";
	}
	catch (const RadiationFieldStoreException& e) {
		const std::string what = e.what();
		EXPECT_NE(what.find("is not a RadFiled3D (.rf3) file"), std::string::npos) << what;
		EXPECT_NE(what.find("too short"), std::string::npos) << what;
	}

	std::remove(foreign.c_str());
	std::remove(truncated.c_str());
}
