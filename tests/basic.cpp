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
#include <fstream>
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

		ch1->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");
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
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", 0).get_bins(), field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", 0).get_bins());
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", 0).get_histogram_bin_width(), field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", 0).get_histogram_bin_width());
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", 0).get_histogram()[0], field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", 0).get_histogram()[0]);
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", 0).get_histogram()[1], field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", 0).get_histogram()[1]);
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", 0).get_histogram()[2], field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", 0).get_histogram()[2]);
		EXPECT_EQ(field->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", 0).get_histogram()[3], field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", 0).get_histogram()[3]);

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

		HistogramVoxel hist(26, 0.1f, nullptr);
		channel->add_custom_layer<HistogramVoxel, float>("hist", hist, 0.f);
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
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.1f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		const float magic_number = 0.134f;

		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), magic_number, "");
		HistogramVoxel& hist1 = channel->get_voxel<HistogramVoxel>("spectra", 0, 5, 0);
		for (size_t i = 0; i < 26; i++)
			hist1.get_histogram()[i] = static_cast<float>(i);

		for (size_t i = 0; i < 26; i++)
			EXPECT_EQ(hist1.get_histogram()[i], static_cast<float>(i));

		for (size_t x = 0; x < 10; x++)
			for (size_t y = 0; y < 10; y++)
				for (size_t z = 0; z < 10; z++) {
					HistogramVoxel& hist = channel->get_voxel<HistogramVoxel>("spectra", x, y, z);
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

	TEST(Voxels, VoxelBufferOperators) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.1f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		const float magic_number = 0.134f;

		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), magic_number, "");
		HistogramVoxel& hist1 = channel->get_voxel<HistogramVoxel>("spectra", 0, 5, 0);
		for (size_t i = 0; i < 26; i++)
			hist1.get_histogram()[i] = static_cast<float>(i);

		channel->add_layer<float>("doserate", magic_number, "Gy/s");

		// check channel integrity, if all values are correctly set

		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(channel->get_voxel<HistogramVoxel>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i));

		VoxelGridBuffer* original_channel = static_cast<VoxelGridBuffer*>(channel->copy());

		// check integrity of original channel
		EXPECT_FLOAT_EQ(original_channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(original_channel->get_voxel<HistogramVoxel>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i));

		// check if the channel is still integer after the copy
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(channel->get_voxel<HistogramVoxel>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i));

		*channel += *original_channel;

		// check if original channel is still the same after the operation
		EXPECT_FLOAT_EQ(original_channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(original_channel->get_voxel<HistogramVoxel>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i));

		// check if the channel is correctly modified
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), 2.f * magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(channel->get_voxel<HistogramVoxel>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i) * 2.f);

		*channel -= *original_channel;
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(channel->get_voxel<HistogramVoxel>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i));

		*channel *= *original_channel;
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), magic_number * magic_number);
		for (size_t i = 0; i < 26; i++)
			EXPECT_FLOAT_EQ(channel->get_voxel<HistogramVoxel>("spectra", 0, 5, 0).get_histogram()[i], static_cast<float>(i) * static_cast<float>(i));

		*channel /= *channel;
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0).get_data(), 1.f);
		EXPECT_FLOAT_EQ(channel->get_voxel<HistogramVoxel>("spectra", 0, 5, 0).get_histogram()[0], 0.f);
		for (size_t i = 1; i < 26; i++)
			EXPECT_FLOAT_EQ(channel->get_voxel<HistogramVoxel>("spectra", 0, 5, 0).get_histogram()[i], 1.f);
	}

	TEST(Storage, Store) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_layer<float>("doserate", 25.3f, "Gy/s");
		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");

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
		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");
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
			size_t bins = field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", i).get_bins();
			float width = field2->get_channel("test_channel")->get_voxel_flat<HistogramVoxel>("spectra", i).get_histogram_bin_width();
			EXPECT_EQ(bins, 26);
			EXPECT_EQ(width, 10.f);
		}
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
			RadFiled3D::HistogramVoxel hist(bins, bin_width, nullptr);
			metadata->set_dynamic_custom_metadata<RadFiled3D::HistogramVoxel>("test_hist", hist);
			RadFiled3D::HistogramVoxel& new_hist = *static_cast<RadFiled3D::HistogramVoxel*>(metadata->get_dynamic_metadata().at("test_hist"));
			for (size_t i = 0; i < 10; i++)
				new_hist.get_histogram()[i] = 1.f * i;
			EXPECT_EQ(new_hist.get_bins(), bins);
			EXPECT_EQ(new_hist.get_histogram_bin_width(), bin_width);
		}

		RadFiled3D::HistogramVoxel* voxel = static_cast<RadFiled3D::HistogramVoxel*>(metadata->get_dynamic_metadata().at("test_hist"));

		for (size_t i = 0; i < 10; i++)
			EXPECT_FLOAT_EQ(voxel->get_histogram()[i], 1.f * i);
	}

	TEST(Storage, JoinFields) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");
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
		channel2->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");

		ScalarVoxel<float>& vx1 = channel->get_voxel_flat<ScalarVoxel<float>>("doserate", 0);
		ScalarVoxel<float>& vx2 = channel2->get_voxel_flat<ScalarVoxel<float>>("doserate", 0);
		const float combined_doserate = vx1.get_data() + vx2.get_data();

		auto metadata2 = FieldStore::peek_metadata("test04.rf3");

		EXPECT_NO_THROW(FieldStore::join(field2, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test04.rf3", FieldJoinMode::Add));

		metadata = std::dynamic_pointer_cast<RadFiled3D::Storage::V1::RadiationFieldMetadata>(FieldStore::load_metadata("test04.rf3"));
		EXPECT_EQ(metadata->get_header().simulation.primary_particle_count, 200);
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
		EXPECT_EQ(metadata->get_header().simulation.primary_particle_count, 300);
		EXPECT_EQ(metadata->get_header().simulation.tube.max_energy_eV, 100.f);
		EXPECT_EQ(metadata->get_header().simulation.tube.radiation_direction, glm::vec3(1.f, 0.f, 0.f));
		EXPECT_EQ(metadata->get_header().simulation.tube.radiation_origin, glm::vec3(0.f, 0.f, 0.f));

		for (size_t i = 0; i < field2->get_voxel_counts().x * field4->get_voxel_counts().y * field4->get_voxel_counts().y; i++) {
			EXPECT_EQ(field4->get_channel("test_channel")->get_voxel_flat<ScalarVoxel<float>>("doserate", i).get_data(), combined_doserate2);
		}
	}

	TEST(Storage, JoinFieldsChecks) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");
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
		channel2->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");

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
		channel3->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");
		EXPECT_THROW(FieldStore::join(field3, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::FieldUnitsOnly), RadiationFieldStoreException);
		EXPECT_NO_THROW(FieldStore::join(field3, std::static_pointer_cast<RadFiled3D::Storage::RadiationFieldMetadata>(metadata), "test05.rf3", FieldJoinMode::Add, RadFiled3D::Storage::FieldJoinCheckMode::NoChecks));
	}

	TEST(Storage, LoadSingleLayer) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");
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
		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");
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

		FieldStore::enable_file_lock_syncronization(true);
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
					channel2->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");
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
