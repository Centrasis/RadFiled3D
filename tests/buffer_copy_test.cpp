#include "radfiled3d/voxel.hpp"
#include "radfiled3d/voxel_buffer.hpp"
#include "radfiled3d/polar_segments.hpp"
#include "radfiled3d/radiation_field.hpp"
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>

using namespace radfiled3d;

namespace {
	const std::vector<std::string> layer_names = { "scalar", "hist", "angular", "vmf" };

	void add_all_layer_types(VoxelBuffer& buffer, float bin_width = 0.5f, const glm::uvec2& segments = glm::uvec2(4, 2)) {
		buffer.add_layer<float>("scalar", 0.f, "Gy");
		buffer.add_custom_layer<HistogramVoxel<float>>("hist", HistogramVoxel<float>(7, bin_width, nullptr), 0.f, "counts");
		buffer.add_custom_layer<AngularResolvedVoxel<float>>("angular", AngularResolvedVoxel<float>(segments, nullptr), 0.f, "counts");
		buffer.add_custom_layer<VMFMixtureVoxel<float>>("vmf", VMFMixtureVoxel<float>(3, nullptr), 0.f, "");
	}

	size_t elements_per_voxel(const VoxelBuffer& buffer, const std::string& layer) {
		return buffer.get_voxel_flat(layer, 0).get_bytes() / sizeof(float);
	}

	float& element(const VoxelBuffer& buffer, const std::string& layer, size_t voxel, size_t idx) {
		return ((float*)buffer.get_voxel_flat(layer, voxel).get_raw())[idx];
	}

	void fill_distinct(VoxelBuffer& buffer) {
		for (size_t l = 0; l < layer_names.size(); l++) {
			const size_t n = elements_per_voxel(buffer, layer_names[l]);
			for (size_t v = 0; v < buffer.get_voxel_count(); v++)
				for (size_t e = 0; e < n; e++)
					element(buffer, layer_names[l], v, e) = static_cast<float>(l * 100000 + v * 100 + e) + 0.25f;
		}
	}

	void expect_same_content(const VoxelBuffer& a, const VoxelBuffer& b) {
		ASSERT_EQ(a.get_voxel_count(), b.get_voxel_count());
		EXPECT_EQ(b.get_voxel_flat<HistogramVoxel<float>>("hist", 0).get_bins(), 7u);
		EXPECT_FLOAT_EQ(b.get_voxel_flat<HistogramVoxel<float>>("hist", 0).get_histogram_bin_width(), 0.5f);
		EXPECT_EQ(b.get_voxel_flat<AngularResolvedVoxel<float>>("angular", 0).get_segments(), glm::uvec2(4, 2));
		EXPECT_EQ(b.get_voxel_flat<VMFMixtureVoxel<float>>("vmf", 0).get_lobes(), 3u);
		for (const auto& layer : layer_names) {
			EXPECT_EQ(a.get_type(layer), b.get_type(layer));
			EXPECT_EQ(a.get_layer_unit(layer), b.get_layer_unit(layer));
			const size_t n = elements_per_voxel(a, layer);
			ASSERT_EQ(n, elements_per_voxel(b, layer));
			for (size_t v = 0; v < a.get_voxel_count(); v++)
				for (size_t e = 0; e < n; e++)
					ASSERT_FLOAT_EQ(element(a, layer, v, e), element(b, layer, v, e)) << layer << " voxel " << v << " element " << e;
		}
	}

	void expect_deep_copy(const VoxelBuffer& original, const VoxelBuffer& copy) {
		expect_same_content(original, copy);
		EXPECT_TRUE(original == copy);

		for (const auto& layer : layer_names) {
			const size_t bytes = copy.get_voxel_flat(layer, 0).get_bytes();
			const char* copy_data = copy.get_layer<char>(layer);
			for (size_t v = 0; v < copy.get_voxel_count(); v++)
				ASSERT_EQ(copy.get_voxel_flat(layer, v).get_raw(), (const void*)(copy_data + v * bytes)) << layer << " voxel " << v;
			EXPECT_NE((const void*)copy_data, (const void*)original.get_layer<char>(layer));
		}

		std::unique_ptr<VoxelBuffer> reference(original.copy());
		for (const auto& layer : layer_names) {
			const size_t n = elements_per_voxel(copy, layer);
			for (size_t v = 0; v < copy.get_voxel_count(); v++)
				for (size_t e = 0; e < n; e++)
					element(copy, layer, v, e) = -1.f;
		}
		expect_same_content(original, *reference);
		EXPECT_FALSE(original == copy);
	}

	TEST(BufferCopyTest, VoxelBufferCopyAllLayerTypes) {
		VoxelBuffer buffer(27);
		add_all_layer_types(buffer);
		fill_distinct(buffer);

		std::unique_ptr<VoxelBuffer> copy(buffer.copy());
		expect_deep_copy(buffer, *copy);
	}

	TEST(BufferCopyTest, PolarSegmentsBufferCopyAllLayerTypes) {
		auto field = std::make_shared<PolarRadiationField>(glm::uvec2(4, 3));
		auto channel = field->add_channel("beam");
		add_all_layer_types(*channel);
		fill_distinct(*channel);

		std::unique_ptr<VoxelBuffer> copy(channel->copy());
		ASSERT_NE(dynamic_cast<PolarSegmentsBuffer*>(copy.get()), nullptr);
		EXPECT_EQ(static_cast<PolarSegmentsBuffer*>(copy.get())->get_segments_count(), glm::uvec2(4, 3));
		expect_deep_copy(*channel, *copy);

		auto field_copy = std::dynamic_pointer_cast<PolarRadiationField>(field->copy());
		ASSERT_NE(field_copy, nullptr);
		expect_deep_copy(*channel, *field_copy->get_channel("beam"));
	}

	TEST(BufferCopyTest, CartesianFieldCopyAllLayerTypes) {
		auto field = std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.25f));
		auto channel = field->add_channel("beam");
		add_all_layer_types(*channel);
		fill_distinct(*channel);

		auto field_copy = std::dynamic_pointer_cast<CartesianRadiationField>(field->copy());
		expect_deep_copy(*channel, *field_copy->get_channel("beam"));
	}

	TEST(BufferEqualityTest, EqualBuffersCompareEqual) {
		VoxelBuffer a(8);
		VoxelBuffer b(8);
		add_all_layer_types(a);
		add_all_layer_types(b);
		EXPECT_TRUE(a == b);
		fill_distinct(a);
		EXPECT_FALSE(a == b);
		fill_distinct(b);
		EXPECT_TRUE(a == b);
		EXPECT_TRUE(b == a);
	}

	TEST(BufferEqualityTest, LastElementOfLastVoxelDiffers) {
		VoxelBuffer a(8);
		add_all_layer_types(a);
		fill_distinct(a);

		for (const auto& layer : layer_names) {
			std::unique_ptr<VoxelBuffer> b(a.copy());
			ASSERT_TRUE(a == *b);
			element(*b, layer, b->get_voxel_count() - 1, elements_per_voxel(*b, layer) - 1) += 1.f;
			EXPECT_FALSE(a == *b) << layer;
			EXPECT_FALSE(*b == a) << layer;
		}
	}

	TEST(BufferEqualityTest, HeaderOnlyDifferences) {
		VoxelBuffer a(8);
		add_all_layer_types(a, 0.5f);
		fill_distinct(a);

		VoxelBuffer bin_width(8);
		add_all_layer_types(bin_width, 0.6f);
		fill_distinct(bin_width);
		EXPECT_FALSE(a == bin_width);

		// Same number of segments (and bytes), different segmentation
		VoxelBuffer segmentation(8);
		add_all_layer_types(segmentation, 0.5f, glm::uvec2(2, 4));
		fill_distinct(segmentation);
		EXPECT_FALSE(a == segmentation);

		VoxelBuffer unit(8);
		unit.add_layer<float>("scalar", 0.f, "Sv");
		unit.add_custom_layer<HistogramVoxel<float>>("hist", HistogramVoxel<float>(7, 0.5f, nullptr), 0.f, "counts");
		unit.add_custom_layer<AngularResolvedVoxel<float>>("angular", AngularResolvedVoxel<float>(glm::uvec2(4, 2), nullptr), 0.f, "counts");
		unit.add_custom_layer<VMFMixtureVoxel<float>>("vmf", VMFMixtureVoxel<float>(3, nullptr), 0.f, "");
		fill_distinct(unit);
		EXPECT_FALSE(a == unit);

		// 15 floats per voxel either way: 3 lobes vs. a 15-bin histogram
		VoxelBuffer type_a(8);
		VoxelBuffer type_b(8);
		type_a.add_custom_layer<VMFMixtureVoxel<float>>("x", VMFMixtureVoxel<float>(3, nullptr), 0.f, "");
		type_b.add_custom_layer<HistogramVoxel<float>>("x", HistogramVoxel<float>(15, 1.f, nullptr), 0.f, "");
		EXPECT_FALSE(type_a == type_b);

		VoxelBuffer other_name(8);
		other_name.add_custom_layer<VMFMixtureVoxel<float>>("y", VMFMixtureVoxel<float>(3, nullptr), 0.f, "");
		EXPECT_FALSE(type_a == other_name);

		VoxelBuffer other_count(9);
		other_count.add_custom_layer<VMFMixtureVoxel<float>>("x", VMFMixtureVoxel<float>(3, nullptr), 0.f, "");
		EXPECT_FALSE(type_a == other_count);
	}

	TEST(BufferEqualityTest, VoxelEqualityComparesAllElements) {
		float a_data[15] = { 0.f };
		float b_data[15] = { 0.f };
		HistogramVoxel<float> ha(15, 1.f, a_data);
		HistogramVoxel<float> hb(15, 1.f, b_data);
		EXPECT_TRUE(ha == hb);
		b_data[14] = 1.f;
		EXPECT_FALSE(ha == hb);
		EXPECT_FALSE(ha == HistogramVoxel<float>(15, 2.f, a_data));

		AngularResolvedVoxel<float> aa(glm::uvec2(5, 3), a_data);
		EXPECT_FALSE(aa == AngularResolvedVoxel<float>(glm::uvec2(5, 3), b_data));
		EXPECT_FALSE(aa == AngularResolvedVoxel<float>(glm::uvec2(3, 5), a_data));
		EXPECT_TRUE(aa == AngularResolvedVoxel<float>(glm::uvec2(5, 3), a_data));

		VMFMixtureVoxel<float> va(3, a_data);
		EXPECT_FALSE(va == VMFMixtureVoxel<float>(3, b_data));
		b_data[14] = 0.f;
		EXPECT_TRUE(va == VMFMixtureVoxel<float>(3, b_data));
	}
}
