#include "radfiled3d/voxel.hpp"
#include "radfiled3d/radiation_field.hpp"
#include "radfiled3d/storage/radiation_field_store.hpp"
#include "radfiled3d/storage/field_accessor.hpp"
#include "radfiled3d/storage/types.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <fstream>
#include <cstdio>
#include <numbers>

using namespace radfiled3d;
using namespace radfiled3d::storage;

namespace {
	class VMFMixtureStorage : public ::testing::Test {
	protected:
		void TearDown() override {
			std::vector<std::string> files = {
				"test_vmf_store.rf3", "test_vmf_join.rf3", "test_vmf_join_ratio.rf3",
				"test_vmf_join_new_layer.rf3", "test_vmf_replace.rf3"
			};
			for (auto& file : files) {
				if (std::ifstream(file).good())
					std::remove(file.c_str());
			}
		}
	};

	std::shared_ptr<v1::RadiationFieldMetadata> make_metadata(size_t primaries = 1000) {
		return std::make_shared<v1::RadiationFieldMetadata>(
			storage::filed_types::v1::RadiationFieldMetadataHeader::Simulation(
				primaries, "geom", "physics",
				storage::filed_types::v1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(0.f, 0.f, -1.f), glm::vec3(0.f, 0.f, 1.f), 15000.f, "tube"
				)
			),
			storage::filed_types::v1::RadiationFieldMetadataHeader::Software("test", "1.0", "", "")
		);
	}

	// 4 x 4 x 4 voxels, so an asymmetric index like (1, 2, 3) detects transposed axes
	std::shared_ptr<CartesianRadiationField> make_field() {
		return std::make_shared<CartesianRadiationField>(glm::vec3(1.f), glm::vec3(0.25f));
	}

	std::shared_ptr<VoxelGridBuffer> add_vmf_channel(std::shared_ptr<CartesianRadiationField> field, const std::string& name, uint32_t lobes) {
		auto channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel(name));
		channel->add_custom_layer<VMFMixtureVoxel<float>>("vmf", VMFMixtureVoxel<float>(lobes, nullptr), 0.f, "");
		channel->add_layer<float>("flux", 0.f, "counts");
		return channel;
	}

	double integrate_over_sphere(const VMFMixtureVoxel<float>& voxel, size_t n) {
		const double golden_angle = std::numbers::pi * (3.0 - std::sqrt(5.0));
		double sum = 0.0;
		for (size_t i = 0; i < n; i++) {
			const double z = 1.0 - (2.0 * i + 1.0) / static_cast<double>(n);
			const double r = std::sqrt(1.0 - z * z);
			const double phi = golden_angle * static_cast<double>(i);
			sum += static_cast<double>(voxel.density(glm::vec3(r * std::cos(phi), r * std::sin(phi), z)));
		}
		return sum * 4.0 * std::numbers::pi / static_cast<double>(n);
	}

	void expect_vec_near(const glm::vec3& a, const glm::vec3& b, float tol) {
		EXPECT_NEAR(a.x, b.x, tol);
		EXPECT_NEAR(a.y, b.y, tol);
		EXPECT_NEAR(a.z, b.z, tol);
	}

	void fill_test_lobes(VMFMixtureVoxel<float>& voxel, float offset) {
		voxel.set_lobe(0, 0.5f, glm::normalize(glm::vec3(1.f, 0.f, 0.f)), 10.f + offset);
		voxel.set_lobe(1, 0.3f, glm::normalize(glm::vec3(0.f, 1.f, 1.f)), 2.5f + offset);
		voxel.set_lobe(2, 0.2f, glm::normalize(glm::vec3(0.f, 0.f, -1.f)), 0.f + offset);
	}

	void expect_test_lobes(const VMFMixtureVoxel<float>& voxel, float offset) {
		ASSERT_EQ(voxel.get_lobes(), 3u);
		EXPECT_FLOAT_EQ(voxel.get_weight(0), 0.5f);
		EXPECT_FLOAT_EQ(voxel.get_weight(1), 0.3f);
		EXPECT_FLOAT_EQ(voxel.get_weight(2), 0.2f);
		expect_vec_near(voxel.get_mean(0), glm::vec3(1.f, 0.f, 0.f), 1e-6f);
		expect_vec_near(voxel.get_mean(1), glm::normalize(glm::vec3(0.f, 1.f, 1.f)), 1e-6f);
		expect_vec_near(voxel.get_mean(2), glm::vec3(0.f, 0.f, -1.f), 1e-6f);
		EXPECT_FLOAT_EQ(voxel.get_kappa(0), 10.f + offset);
		EXPECT_FLOAT_EQ(voxel.get_kappa(1), 2.5f + offset);
		EXPECT_FLOAT_EQ(voxel.get_kappa(2), 0.f + offset);
	}

	// === Voxel Unit Tests ===

	TEST(VMFMixtureVoxelTest, Construction) {
		float buffer[15] = { 0.f };
		VMFMixtureVoxel<float> voxel(3, buffer);
		EXPECT_EQ(voxel.get_lobes(), 3u);
		EXPECT_EQ(voxel.get_bytes(), sizeof(float) * 15);
		EXPECT_EQ(voxel.get_lobes_data().size(), 15u);
		EXPECT_EQ(voxel.get_raw(), buffer);
		EXPECT_EQ(voxel.get_type(), "vmf_mixture");
		EXPECT_EQ(typing::Helper::get_dtype(voxel.get_type()), typing::DType::VMFMixture);
		EXPECT_EQ(typing::Helper::get_bytes_of_dtype(typing::DType::VMFMixture), sizeof(float));

		VMFMixtureVoxel<float> empty;
		EXPECT_EQ(empty.get_lobes(), 0u);
		EXPECT_EQ(empty.get_bytes(), 0u);
		EXPECT_EQ(empty.get_raw(), nullptr);
	}

	TEST(VMFMixtureVoxelTest, LobeAccessAndLayout) {
		float buffer[15] = { 0.f };
		VMFMixtureVoxel<float> voxel(3, buffer);
		fill_test_lobes(voxel, 0.f);
		expect_test_lobes(voxel, 0.f);

		EXPECT_FLOAT_EQ(buffer[5], 0.3f);
		EXPECT_FLOAT_EQ(buffer[6], 0.f);
		EXPECT_FLOAT_EQ(buffer[7], std::sqrt(0.5f));
		EXPECT_FLOAT_EQ(buffer[8], std::sqrt(0.5f));
		EXPECT_FLOAT_EQ(buffer[9], 2.5f);
		EXPECT_FLOAT_EQ(voxel.get_lobes_data()[4], 10.f);

		voxel.clear();
		for (float v : buffer)
			EXPECT_FLOAT_EQ(v, 0.f);
	}

	TEST(VMFMixtureVoxelTest, HeaderRoundTrip) {
		float buffer[20] = { 0.f };
		VMFMixtureVoxel<float> voxel(4, buffer);
		auto header = voxel.get_header();
		EXPECT_EQ(header.header_bytes, sizeof(VMFMixtureVoxel<float>::VMFDefinition));
		EXPECT_EQ(((VMFMixtureVoxel<float>::VMFDefinition*)header.header)->lobes, 4u);

		VMFMixtureVoxel<float> restored;
		restored.init_from_header(header.header);
		EXPECT_EQ(restored.get_lobes(), 4u);
		EXPECT_EQ(restored.get_bytes(), sizeof(float) * 20);

		OwningVMFMixtureVoxel<float> owning;
		owning.init_from_header(header.header);
		EXPECT_EQ(owning.get_lobes(), 4u);
		for (float v : owning.get_lobes_data())
			EXPECT_FLOAT_EQ(v, 0.f);
	}

	TEST(VMFMixtureVoxelTest, OwningCopy) {
		OwningVMFMixtureVoxel<float> a(3);
		fill_test_lobes(a, 0.f);

		OwningVMFMixtureVoxel<float> b(a);
		EXPECT_NE(a.get_raw(), b.get_raw());
		expect_test_lobes(b, 0.f);

		OwningVMFMixtureVoxel<float> c(1);
		c = a;
		EXPECT_NE(a.get_raw(), c.get_raw());
		expect_test_lobes(c, 0.f);

		b.set_lobe(0, 0.9f, glm::vec3(0.f, 1.f, 0.f), 1.f);
		EXPECT_FLOAT_EQ(a.get_weight(0), 0.5f);

		OwningVMFMixtureVoxel<float> d(3, (float*)a.get_raw());
		expect_test_lobes(d, 0.f);
	}

	TEST(VMFMixtureVoxelTest, DensityIntegratesToOne) {
		const glm::vec3 mean = glm::normalize(glm::vec3(0.3f, -0.5f, 0.8f));
		for (float kappa : { 0.f, 1e-6f, 1.f, 50.f, 500.f }) {
			OwningVMFMixtureVoxel<float> voxel(1);
			voxel.set_lobe(0, 1.f, mean, kappa);
			EXPECT_NEAR(integrate_over_sphere(voxel, 2000000), 1.0, 1e-3) << "kappa = " << kappa;
		}
	}

	TEST(VMFMixtureVoxelTest, DensityValues) {
		OwningVMFMixtureVoxel<float> voxel(2);
		voxel.set_lobe(0, 1.f, glm::vec3(0.f, 0.f, 1.f), 0.f);
		EXPECT_NEAR(voxel.density(glm::vec3(1.f, 0.f, 0.f)), 1.0 / (4.0 * std::numbers::pi), 1e-7);

		const double kappa = 20.0;
		voxel.set_lobe(0, 1.f, glm::vec3(0.f, 0.f, 1.f), static_cast<float>(kappa));
		const double peak = kappa / (2.0 * std::numbers::pi * (1.0 - std::exp(-2.0 * kappa)));
		EXPECT_NEAR(voxel.density(glm::vec3(0.f, 0.f, 3.f)), peak, peak * 1e-5);
		EXPECT_NEAR(voxel.density(glm::vec3(0.f, 0.f, -1.f)), peak * std::exp(-2.0 * kappa), 1e-12);

		voxel.set_lobe(0, 0.25f, glm::vec3(0.f, 0.f, 1.f), 5.f);
		voxel.set_lobe(1, 0.75f, glm::vec3(1.f, 0.f, 0.f), 80.f);
		EXPECT_NEAR(integrate_over_sphere(voxel, 2000000), 1.0, 1e-3);

		voxel.clear();
		EXPECT_FLOAT_EQ(voxel.density(glm::vec3(0.f, 0.f, 1.f)), 0.f);
	}

	TEST(VMFMixtureVoxelTest, MergeIdenticalLobesKeepsKappa) {
		for (float kappa : { 0.5f, 5.f, 50.f, 500.f }) {
			OwningVMFMixtureVoxel<float> a(2);
			OwningVMFMixtureVoxel<float> b(2);
			const glm::vec3 m0 = glm::normalize(glm::vec3(1.f, 2.f, 3.f));
			const glm::vec3 m1 = glm::normalize(glm::vec3(-1.f, 0.f, 0.5f));
			a.set_lobe(0, 0.6f, m0, kappa);
			a.set_lobe(1, 0.4f, m1, 2.f * kappa);
			b.set_lobe(0, 0.6f, m0, kappa);
			b.set_lobe(1, 0.4f, m1, 2.f * kappa);

			OwningVMFMixtureVoxel<float> out(2);
			VMFMixtureVoxel<float>::merge(a, 0.3, b, 0.7, out);

			float weight_sum = 0.f;
			for (size_t k = 0; k < 2; k++)
				weight_sum += out.get_weight(k);
			EXPECT_NEAR(weight_sum, 1.f, 1e-6f);

			const size_t k0 = (glm::dot(out.get_mean(0), m0) > 0.99f) ? 0 : 1;
			const size_t k1 = 1 - k0;
			EXPECT_NEAR(out.get_weight(k0), 0.6f, 1e-6f);
			EXPECT_NEAR(out.get_weight(k1), 0.4f, 1e-6f);
			expect_vec_near(out.get_mean(k0), m0, 1e-5f);
			expect_vec_near(out.get_mean(k1), m1, 1e-5f);
			EXPECT_NEAR(out.get_kappa(k0), kappa, kappa * 1e-3f) << "kappa = " << kappa;
			EXPECT_NEAR(out.get_kappa(k1), 2.f * kappa, 2.f * kappa * 1e-3f) << "kappa = " << kappa;
		}
	}

	TEST(VMFMixtureVoxelTest, MergeReducesClosestPair) {
		OwningVMFMixtureVoxel<float> a(2);
		OwningVMFMixtureVoxel<float> b(2);
		const glm::vec3 x(1.f, 0.f, 0.f);
		const glm::vec3 near_x = glm::normalize(glm::vec3(1.f, 0.1f, 0.f));
		a.set_lobe(0, 0.5f, x, 20.f);
		a.set_lobe(1, 0.5f, glm::vec3(0.f, 0.f, 1.f), 20.f);
		b.set_lobe(0, 0.5f, near_x, 20.f);
		b.set_lobe(1, 0.5f, glm::vec3(0.f, -1.f, 0.f), 20.f);

		OwningVMFMixtureVoxel<float> out(3);
		VMFMixtureVoxel<float>::merge(a, 1.0, b, 1.0, out);

		float weight_sum = 0.f;
		int merged = -1;
		for (size_t k = 0; k < 3; k++) {
			weight_sum += out.get_weight(k);
			if (std::abs(out.get_weight(k) - 0.5f) < 1e-6f)
				merged = static_cast<int>(k);
		}
		EXPECT_NEAR(weight_sum, 1.f, 1e-6f);
		ASSERT_GE(merged, 0);
		expect_vec_near(out.get_mean(merged), glm::normalize(x + near_x), 1e-5f);
		// The merged lobe is broader than its components, as the angle between them adds spread
		EXPECT_LT(out.get_kappa(merged), 20.f);
		EXPECT_GT(out.get_kappa(merged), 10.f);

		// Reducing to a single lobe of weights summing to 1
		OwningVMFMixtureVoxel<float> single(1);
		VMFMixtureVoxel<float>::merge(a, 1.0, b, 1.0, single);
		EXPECT_NEAR(single.get_weight(0), 1.f, 1e-6f);
		EXPECT_NEAR(glm::length(single.get_mean(0)), 1.f, 1e-5f);
		EXPECT_GE(single.get_kappa(0), 0.f);
	}

	TEST(VMFMixtureVoxelTest, MergeEmptyInputs) {
		OwningVMFMixtureVoxel<float> empty(2);
		OwningVMFMixtureVoxel<float> b(2);
		b.set_lobe(0, 2.f, glm::vec3(0.f, 1.f, 0.f), 7.f);
		b.set_lobe(1, 2.f, glm::vec3(0.f, 0.f, 1.f), 3.f);

		OwningVMFMixtureVoxel<float> out(2);
		VMFMixtureVoxel<float>::merge(empty, 0.5, b, 0.5, out);
		EXPECT_NEAR(out.get_weight(0), 0.5f, 1e-6f);
		EXPECT_NEAR(out.get_weight(1), 0.5f, 1e-6f);
		expect_vec_near(out.get_mean(0), glm::vec3(0.f, 1.f, 0.f), 1e-6f);
		EXPECT_FLOAT_EQ(out.get_kappa(0), 7.f);
		EXPECT_FLOAT_EQ(out.get_kappa(1), 3.f);

		VMFMixtureVoxel<float>::merge(b, 1.0, empty, 1.0, out);
		EXPECT_NEAR(out.get_weight(0), 0.5f, 1e-6f);
		EXPECT_FLOAT_EQ(out.get_kappa(1), 3.f);

		out.set_lobe(0, 1.f, glm::vec3(1.f, 0.f, 0.f), 1.f);
		VMFMixtureVoxel<float>::merge(empty, 1.0, empty, 1.0, out);
		for (float v : out.get_lobes_data())
			EXPECT_FLOAT_EQ(v, 0.f);
	}

	TEST(VMFMixtureVoxelTest, MergeRejectsZeroLobeOutput) {
		OwningVMFMixtureVoxel<float> a(2);
		OwningVMFMixtureVoxel<float> b(2);
		a.set_lobe(0, 1.f, glm::vec3(1.f, 0.f, 0.f), 5.f);
		b.set_lobe(0, 1.f, glm::vec3(0.f, 1.f, 0.f), 5.f);

		// out's lobe count is the target the inputs are reduced onto, so 0 lobes means the
		// merged mixture has nowhere to go. This used to return silently.
		OwningVMFMixtureVoxel<float> no_lobes(0);
		EXPECT_THROW(VMFMixtureVoxel<float>::merge(a, 0.5, b, 0.5, no_lobes), std::invalid_argument);

		// a single lobe is a valid target: the two inputs are reduced onto it
		OwningVMFMixtureVoxel<float> one(1);
		EXPECT_NO_THROW(VMFMixtureVoxel<float>::merge(a, 0.5, b, 0.5, one));
		EXPECT_NEAR(one.get_weight(0), 1.f, 1e-6f);
	}

	TEST(VMFMixtureVoxelTest, MergeInPlace) {
		OwningVMFMixtureVoxel<float> a(2);
		OwningVMFMixtureVoxel<float> b(2);
		a.set_lobe(0, 1.f, glm::vec3(1.f, 0.f, 0.f), 5.f);
		b.set_lobe(0, 1.f, glm::vec3(0.f, 1.f, 0.f), 5.f);

		VMFMixtureVoxel<float>::merge(a, 0.25, b, 0.75, a);
		const size_t kx = (a.get_mean(0).x > 0.5f) ? 0 : 1;
		EXPECT_NEAR(a.get_weight(kx), 0.25f, 1e-6f);
		EXPECT_NEAR(a.get_weight(1 - kx), 0.75f, 1e-6f);
		expect_vec_near(a.get_mean(1 - kx), glm::vec3(0.f, 1.f, 0.f), 1e-6f);
	}

	// === Field / Storage Tests ===

	TEST(VMFMixtureFieldTest, LayerAccess) {
		auto field = make_field();
		auto channel = add_vmf_channel(field, "beam", 3);
		EXPECT_EQ(channel->get_type("vmf"), "vmf_mixture");
		EXPECT_EQ(channel->get_voxel_flat<VMFMixtureVoxel<float>>("vmf", 0).get_lobes(), 3u);

		fill_test_lobes(channel->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3), 0.f);
		const size_t flat = channel->get_voxel_idx(1, 2, 3);
		EXPECT_EQ(flat, 3u * 16u + 2u * 4u + 1u);
		const float* raw = channel->get_layer<float>("vmf");
		EXPECT_FLOAT_EQ(raw[flat * 15 + 0], 0.5f);
		EXPECT_FLOAT_EQ(raw[flat * 15 + 9], 2.5f);
		EXPECT_FLOAT_EQ(raw[flat * 15 + 10], 0.2f);
		EXPECT_FLOAT_EQ(channel->get_voxel_flat<VMFMixtureVoxel<float>>("vmf", flat - 1).get_weight(0), 0.f);
	}

	TEST(VMFMixtureFieldTest, FieldCopy) {
		auto field = make_field();
		auto channel = add_vmf_channel(field, "beam", 3);
		VMFMixtureVoxel<float>& vx = channel->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3);
		fill_test_lobes(vx, 0.f);

		auto copy = std::dynamic_pointer_cast<CartesianRadiationField>(field->copy());
		auto copy_channel = std::static_pointer_cast<VoxelGridBuffer>(copy->get_channel("beam"));
		VMFMixtureVoxel<float>& copy_vx = copy_channel->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3);
		EXPECT_NE(copy_vx.get_raw(), vx.get_raw());
		expect_test_lobes(copy_vx, 0.f);

		copy_vx.set_lobe(0, 0.9f, glm::vec3(0.f, 1.f, 0.f), 1.f);
		EXPECT_FLOAT_EQ(vx.get_weight(0), 0.5f);
	}

	TEST(VMFMixtureFieldTest, ArithmeticThrows) {
		auto field = make_field();
		auto channel = add_vmf_channel(field, "beam", 2);
		channel->get_voxel<ScalarVoxel<float>>("flux", 1, 2, 3) = 4.f;
		channel->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).set_lobe(0, 1.f, glm::vec3(0.f, 0.f, 1.f), 3.f);

		VoxelBuffer* other = channel->copy();
		EXPECT_THROW(*channel += *other, std::runtime_error);
		EXPECT_THROW(*channel -= *other, std::runtime_error);
		EXPECT_THROW(*channel *= *other, std::runtime_error);
		EXPECT_THROW(*channel /= *other, std::runtime_error);
		EXPECT_THROW(*channel += 1.f, std::runtime_error);
		EXPECT_THROW(*channel -= 1.f, std::runtime_error);
		EXPECT_THROW(*channel *= 2.f, std::runtime_error);
		EXPECT_THROW(*channel /= 2.f, std::runtime_error);
		delete other;

		try {
			*channel *= 2.f;
		}
		catch (const std::runtime_error& e) {
			EXPECT_NE(std::string(e.what()).find("is not defined for vMF mixture layers"), std::string::npos);
		}

		// Nothing was modified by the failed operations
		EXPECT_FLOAT_EQ(channel->get_voxel<ScalarVoxel<float>>("flux", 1, 2, 3).get_data(), 4.f);
		EXPECT_FLOAT_EQ(channel->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).get_weight(0), 1.f);
		EXPECT_FLOAT_EQ(channel->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).get_kappa(0), 3.f);
	}

	TEST_F(VMFMixtureStorage, StoreAndLoad) {
		auto field = make_field();
		auto channel = add_vmf_channel(field, "beam", 3);
		fill_test_lobes(channel->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3), 0.f);
		fill_test_lobes(channel->get_voxel<VMFMixtureVoxel<float>>("vmf", 3, 0, 0), 1.f);
		channel->get_voxel<ScalarVoxel<float>>("flux", 1, 2, 3) = 42.f;
		const size_t flat = channel->get_voxel_idx(1, 2, 3);

		FieldStore::store(field, make_metadata(), "test_vmf_store.rf3");

		auto loaded = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test_vmf_store.rf3"));
		auto loaded_ch = std::static_pointer_cast<VoxelGridBuffer>(loaded->get_channel("beam"));
		EXPECT_EQ(loaded_ch->get_type("vmf"), "vmf_mixture");
		expect_test_lobes(loaded_ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3), 0.f);
		expect_test_lobes(loaded_ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 3, 0, 0), 1.f);
		EXPECT_EQ(loaded_ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 0, 0, 0).get_lobes(), 3u);
		for (float v : loaded_ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 0, 0, 0).get_lobes_data())
			EXPECT_FLOAT_EQ(v, 0.f);
		EXPECT_FLOAT_EQ(loaded_ch->get_voxel<ScalarVoxel<float>>("flux", 1, 2, 3).get_data(), 42.f);
		EXPECT_TRUE(*loaded_ch == *channel);

		{
			std::ifstream stream("test_vmf_store.rf3", std::ios::binary);
			auto layer = FieldStore::load_single_layer(stream, "beam", "vmf");
			ASSERT_NE(layer, nullptr);
			EXPECT_EQ(layer->get_voxel_count(), 64u);
			expect_test_lobes(layer->get_voxel_flat<VMFMixtureVoxel<float>>(flat), 0.f);
		}

		{
			std::ifstream stream1("test_vmf_store.rf3", std::ios::binary);
			auto accessor = std::dynamic_pointer_cast<storage::v1::CartesianFieldAccessor>(FieldStore::construct_accessor(stream1));
			ASSERT_NE(accessor, nullptr);

			std::ifstream stream2("test_vmf_store.rf3", std::ios::binary);
			auto voxel = accessor->accessVoxelFlat<float, VMFMixtureVoxel<float>>(stream2, "beam", "vmf", flat);
			ASSERT_NE(voxel, nullptr);
			EXPECT_EQ(voxel->get_type(), "vmf_mixture");
			expect_test_lobes(*voxel, 0.f);

			auto voxel_3d = accessor->accessVoxel<float, VMFMixtureVoxel<float>>(stream2, "beam", "vmf", glm::uvec3(3, 0, 0));
			expect_test_lobes(*voxel_3d, 1.f);

			auto voxels = accessor->accessVoxelsFlat<float, VMFMixtureVoxel<float>>(stream2, "beam", "vmf", { flat, 0 });
			ASSERT_EQ(voxels.size(), 2u);
			expect_test_lobes(*voxels[0], 0.f);
			EXPECT_FLOAT_EQ(voxels[1]->get_weight(0), 0.f);

			auto acc_channel = accessor->accessChannel(stream2, "beam");
			expect_test_lobes(acc_channel->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3), 0.f);

			auto acc_layer = accessor->accessLayer(stream2, "beam", "vmf");
			expect_test_lobes(acc_layer->get_voxel<VMFMixtureVoxel<float>>(1, 2, 3), 0.f);
		}
	}

	TEST_F(VMFMixtureStorage, JoinIdenticalLobes) {
		auto create = [](float flux) {
			auto field = make_field();
			auto ch = add_vmf_channel(field, "beam", 2);
			ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).set_lobe(0, 0.7f, glm::vec3(0.f, 0.f, 1.f), 40.f);
			ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).set_lobe(1, 0.3f, glm::vec3(1.f, 0.f, 0.f), 4.f);
			ch->get_voxel<ScalarVoxel<float>>("flux", 1, 2, 3) = flux;
			return field;
		};

		FieldStore::store(create(3.f), make_metadata(1000), "test_vmf_join.rf3");
		FieldStore::join(create(7.f), make_metadata(1000), "test_vmf_join.rf3", FieldJoinMode::Add, FieldJoinCheckMode::NoChecks);

		auto loaded = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test_vmf_join.rf3"));
		auto ch = std::static_pointer_cast<VoxelGridBuffer>(loaded->get_channel("beam"));
		EXPECT_FLOAT_EQ(ch->get_voxel<ScalarVoxel<float>>("flux", 1, 2, 3).get_data(), 10.f);

		const VMFMixtureVoxel<float>& vx = ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3);
		ASSERT_EQ(vx.get_lobes(), 2u);
		EXPECT_NEAR(vx.get_weight(0) + vx.get_weight(1), 1.f, 1e-6f);
		const size_t kz = (vx.get_mean(0).z > 0.5f) ? 0 : 1;
		EXPECT_NEAR(vx.get_weight(kz), 0.7f, 1e-6f);
		expect_vec_near(vx.get_mean(kz), glm::vec3(0.f, 0.f, 1.f), 1e-6f);
		expect_vec_near(vx.get_mean(1 - kz), glm::vec3(1.f, 0.f, 0.f), 1e-6f);
		EXPECT_NEAR(vx.get_kappa(kz), 40.f, 40.f * 1e-3f);
		EXPECT_NEAR(vx.get_kappa(1 - kz), 4.f, 4.f * 1e-3f);

		// Empty voxels stay empty
		for (float v : ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 0, 0, 0).get_lobes_data())
			EXPECT_FLOAT_EQ(v, 0.f);

		// Joining the same file repeatedly must not drift kappa
		for (int i = 0; i < 5; i++)
			FieldStore::join(create(1.f), make_metadata(1000), "test_vmf_join.rf3", FieldJoinMode::Add, FieldJoinCheckMode::NoChecks);
		loaded = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test_vmf_join.rf3"));
		ch = std::static_pointer_cast<VoxelGridBuffer>(loaded->get_channel("beam"));
		const VMFMixtureVoxel<float>& vx2 = ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3);
		const size_t kz2 = (vx2.get_mean(0).z > 0.5f) ? 0 : 1;
		EXPECT_NEAR(vx2.get_kappa(kz2), 40.f, 40.f * 1e-3f);
		EXPECT_NEAR(vx2.get_kappa(1 - kz2), 4.f, 4.f * 1e-3f);
	}

	TEST_F(VMFMixtureStorage, JoinWeightsByRatio) {
		auto create = [](const glm::vec3& dir) {
			auto field = make_field();
			auto ch = add_vmf_channel(field, "beam", 2);
			ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).set_lobe(0, 1.f, dir, 25.f);
			return field;
		};

		// ratio = 3000 / (1000 + 3000) = 0.75 for the additional source
		FieldStore::store(create(glm::vec3(1.f, 0.f, 0.f)), make_metadata(1000), "test_vmf_join_ratio.rf3");
		FieldStore::join(create(glm::vec3(0.f, 1.f, 0.f)), make_metadata(3000), "test_vmf_join_ratio.rf3", FieldJoinMode::Add, FieldJoinCheckMode::NoChecks);

		auto loaded = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test_vmf_join_ratio.rf3"));
		auto ch = std::static_pointer_cast<VoxelGridBuffer>(loaded->get_channel("beam"));
		const VMFMixtureVoxel<float>& vx = ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3);
		const size_t kx = (vx.get_mean(0).x > 0.5f) ? 0 : 1;
		EXPECT_NEAR(vx.get_weight(kx), 0.25f, 1e-6f);
		EXPECT_NEAR(vx.get_weight(1 - kx), 0.75f, 1e-6f);
		expect_vec_near(vx.get_mean(1 - kx), glm::vec3(0.f, 1.f, 0.f), 1e-6f);
		EXPECT_FLOAT_EQ(vx.get_kappa(kx), 25.f);

		// Direct store-level joins
		auto target = create(glm::vec3(1.f, 0.f, 0.f));
		const storage::v1::FieldStore store;
		store.join(target, create(glm::vec3(0.f, 1.f, 0.f)), FieldJoinMode::Mean, FieldJoinCheckMode::NoChecks);
		auto target_ch = std::static_pointer_cast<VoxelGridBuffer>(target->get_channel("beam"));
		EXPECT_NEAR(target_ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).get_weight(0), 0.5f, 1e-6f);
		EXPECT_NEAR(target_ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).get_weight(1), 0.5f, 1e-6f);

		auto add_target = create(glm::vec3(1.f, 0.f, 0.f));
		store.join(add_target, create(glm::vec3(0.f, 1.f, 0.f)), FieldJoinMode::Add, FieldJoinCheckMode::NoChecks);
		auto add_ch = std::static_pointer_cast<VoxelGridBuffer>(add_target->get_channel("beam"));
		EXPECT_FLOAT_EQ(add_ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).get_weight(0), 1.f);
		EXPECT_FLOAT_EQ(add_ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).get_weight(1), 0.f);

		store.join(add_target, create(glm::vec3(0.f, 1.f, 0.f)), FieldJoinMode::AddWeighted, FieldJoinCheckMode::NoChecks, 0.5f);
		EXPECT_NEAR(add_ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).get_weight(1), 0.5f, 1e-6f);
		expect_vec_near(add_ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).get_mean(1), glm::vec3(0.f, 1.f, 0.f), 1e-6f);

		auto identity_target = create(glm::vec3(1.f, 0.f, 0.f));
		store.join(identity_target, create(glm::vec3(0.f, 1.f, 0.f)), FieldJoinMode::Identity, FieldJoinCheckMode::NoChecks);
		auto identity_ch = std::static_pointer_cast<VoxelGridBuffer>(identity_target->get_channel("beam"));
		EXPECT_FLOAT_EQ(identity_ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).get_weight(0), 1.f);
		EXPECT_FLOAT_EQ(identity_ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3).get_weight(1), 0.f);

		EXPECT_THROW(store.join(create(glm::vec3(1.f, 0.f, 0.f)), create(glm::vec3(0.f, 1.f, 0.f)), FieldJoinMode::Subtract, FieldJoinCheckMode::NoChecks), RadiationFieldStoreException);
		EXPECT_THROW(store.join(create(glm::vec3(1.f, 0.f, 0.f)), create(glm::vec3(0.f, 1.f, 0.f)), FieldJoinMode::Multiply, FieldJoinCheckMode::NoChecks), RadiationFieldStoreException);
	}

	TEST_F(VMFMixtureStorage, JoinTakesOverNewLayer) {
		auto target = make_field();
		std::static_pointer_cast<VoxelGridBuffer>(target->add_channel("beam"))->add_layer<float>("flux", 1.f, "counts");
		FieldStore::store(target, make_metadata(), "test_vmf_join_new_layer.rf3");

		auto source = make_field();
		auto ch = add_vmf_channel(source, "beam", 3);
		fill_test_lobes(ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3), 0.f);
		FieldStore::join(source, make_metadata(), "test_vmf_join_new_layer.rf3", FieldJoinMode::Add, FieldJoinCheckMode::NoChecks);

		auto loaded = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test_vmf_join_new_layer.rf3"));
		auto loaded_ch = std::static_pointer_cast<VoxelGridBuffer>(loaded->get_channel("beam"));
		ASSERT_TRUE(loaded_ch->has_layer("vmf"));
		expect_test_lobes(loaded_ch->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3), 0.f);
	}

	TEST_F(VMFMixtureStorage, Replace) {
		auto field = make_field();
		auto beam = add_vmf_channel(field, "beam", 3);
		auto kept = add_vmf_channel(field, "kept", 3);
		fill_test_lobes(beam->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3), 0.f);
		fill_test_lobes(kept->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3), 2.f);
		FieldStore::store(field, make_metadata(), "test_vmf_replace.rf3");

		auto update = make_field();
		auto new_beam = add_vmf_channel(update, "beam", 3);
		fill_test_lobes(new_beam->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3), 5.f);
		FieldStore::replace(update, make_metadata(), "test_vmf_replace.rf3");

		auto loaded = std::static_pointer_cast<CartesianRadiationField>(FieldStore::load("test_vmf_replace.rf3"));
		expect_test_lobes(std::static_pointer_cast<VoxelGridBuffer>(loaded->get_channel("beam"))->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3), 5.f);
		expect_test_lobes(std::static_pointer_cast<VoxelGridBuffer>(loaded->get_channel("kept"))->get_voxel<VMFMixtureVoxel<float>>("vmf", 1, 2, 3), 2.f);
	}
}
