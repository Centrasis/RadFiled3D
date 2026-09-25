#include "RadFiled3D/VoxelGrid.hpp"
#include "RadFiled3D/RadiationField.hpp"
#include "RadFiled3D/GridTracer.hpp"
#include <iostream>
#include "RadFiled3D/storage/RadiationFieldStore.hpp"
#include <memory>
#include <vector>
#include <set>
#include <chrono>
#include <fstream>
#include "gtest/gtest.h"
#include <cstdio>

using namespace RadFiled3D;

namespace {
	TEST(Sampling, TraceInside) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.1));
		auto buffer = field.add_channel("test");

		SamplingGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(0.f), glm::vec3(1.f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 9);

		for (int i = 0; i < 9; i++)
			EXPECT_EQ(result[i], ((VoxelGridBuffer*)buffer.get())->get_voxel_idx_by_coord(0.1f * (i + 1), 0.1f * (i + 1), 0.1f * (i + 1)));


		result = tracer.trace(glm::vec3(0.f), glm::vec3(0.f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 0);

		result = tracer.trace(glm::vec3(0.f), glm::vec3(0.15f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 1);

		result = tracer.trace(glm::vec3(0.f), glm::vec3(0.22f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 2);

		result = tracer.trace(glm::vec3(0.f, 0.5f, 0.5f), glm::vec3(1.f, 0.5f, 0.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 9);

		result = tracer.trace(glm::vec3(0.f, 0.5f, 0.5f), glm::vec3(0.5f, 0.5f, 0.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 5);
	}

	TEST(Sampling, TraceOutside) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.1));
		auto buffer = field.add_channel("test");

		SamplingGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(-2.f), glm::vec3(-1.f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 0);

		result = tracer.trace(glm::vec3(2.f), glm::vec3(3.f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 0);
	}

	TEST(Sampling, TraceAnywhere) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.1));
		auto buffer = field.add_channel("test");

		SamplingGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(-0.5f), glm::vec3(0.5f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 5);

		result = tracer.trace(glm::vec3(0.5f), glm::vec3(-0.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 5);

		result = tracer.trace(glm::vec3(0.5f), glm::vec3(2.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 4);
		size_t max_idx = 0.f;
		for (size_t idx : result)
			if (idx > max_idx)
				max_idx = idx;
		EXPECT_EQ(max_idx, field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z - 1);
	}

	TEST(Sampling, TraceEdgeCase) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.02f));
		auto buffer = field.add_channel("test");
		auto half_dim = (field.get_field_dimensions() / 2.f) * 1000.f;
		SamplingGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace((glm::vec3(4.20631f, 126.352f, 71.0123f) + half_dim) / 1000.f, (glm::vec3(-244.532f, -111.553f, 500.f) + half_dim) / 1000.f);
		auto unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());

		size_t max_idx = 0.f;
		for (size_t idx : result)
			if (idx > max_idx)
				max_idx = idx;
		EXPECT_LE(max_idx, field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z - 1);
	}

	TEST(Sampling, TraceBigField) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.001));
		auto buffer = field.add_channel("test");

		SamplingGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(0.f), glm::vec3(1.f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		// the channel has the field's 1000³ voxels (it had 999³ before VoxelGrid rounded like CartesianRadiationField)
		EXPECT_EQ(result.size(), 999);
	}

	TEST(Bresenham, TraceInside) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.1));
		auto buffer = field.add_channel("test");

		BresenhamGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(0.f), glm::vec3(1.f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 9);

		for (int i = 0; i < 9; i++)
			EXPECT_EQ(result[i], ((VoxelGridBuffer*)buffer.get())->get_voxel_idx_by_coord(0.1f * (i + 1), 0.1f * (i + 1), 0.1f * (i + 1)));
			

		result = tracer.trace(glm::vec3(0.f), glm::vec3(0.f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 0);

		result = tracer.trace(glm::vec3(0.f), glm::vec3(0.1f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 1);

		result = tracer.trace(glm::vec3(0.f), glm::vec3(0.2f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 2);

		result = tracer.trace(glm::vec3(0.f, 0.5f, 0.5f), glm::vec3(1.f, 0.5f, 0.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 9);

		result = tracer.trace(glm::vec3(0.f, 0.5f, 0.5f), glm::vec3(0.5f, 0.5f, 0.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 5);
	}

	TEST(Bresenham, TraceOutside) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.1));
		auto buffer = field.add_channel("test");

		BresenhamGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(-2.f), glm::vec3(-1.f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 0);

		result = tracer.trace(glm::vec3(2.f), glm::vec3(3.f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 0);
	}

	TEST(Bresenham, TraceAnywhere) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.1));
		auto buffer = field.add_channel("test");

		BresenhamGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(-0.5f), glm::vec3(0.5f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 5);

		result = tracer.trace(glm::vec3(0.5f), glm::vec3(-0.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 5);

		result = tracer.trace(glm::vec3(0.5f), glm::vec3(2.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 4);

		size_t max_idx = 0.f;
		for (size_t idx : result)
			if (idx > max_idx)
				max_idx = idx;
		EXPECT_EQ(max_idx, field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z - 1);
	}

	TEST(Bresenham, TraceBigField) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.001));
		auto buffer = field.add_channel("test");

		BresenhamGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(0.f), glm::vec3(1.f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		// the channel has the field's 1000³ voxels (it had 999³ before VoxelGrid rounded like CartesianRadiationField)
		EXPECT_EQ(result.size(), 999);
	}

	TEST(LineTracing, TraceInside) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.1));
		auto buffer = field.add_channel("test");
		LinetracingGridTracer tracer(*(VoxelGridBuffer*)buffer.get());
		auto result = tracer.trace(glm::vec3(0.f), glm::vec3(1.f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 42);

		result = tracer.trace(glm::vec3(0.f), glm::vec3(0.f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 0);

		result = tracer.trace(glm::vec3(0.f), glm::vec3(0.15f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 4);

		result = tracer.trace(glm::vec3(0.05f, 0.05f, 0.05f), glm::vec3(0.195f, 0.195f, 0.195f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 4);

		result = tracer.trace(glm::vec3(0.f, 0.05f, 0.f), glm::vec3(0.f, 0.18f, 0.f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 1);

		result = tracer.trace(glm::vec3(0.f, 0.5f, 0.5f), glm::vec3(1.f, 0.5f, 0.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 9);

		result = tracer.trace(glm::vec3(0.f, 0.5f, 0.5f), glm::vec3(0.5f, 0.5f, 0.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 5);
	}

	TEST(LineTracing, TraceOutside) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.1));
		auto buffer = field.add_channel("test");
		LinetracingGridTracer tracer(*(VoxelGridBuffer*)buffer.get());
		auto result = tracer.trace(glm::vec3(-2.f), glm::vec3(-1.f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 0);
		result = tracer.trace(glm::vec3(2.f), glm::vec3(3.f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 0);
	}

	TEST(LineTracing, TraceAnywhere) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.1));
		auto buffer = field.add_channel("test");
		LinetracingGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(-0.5f), glm::vec3(0.5f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 17);

		result = tracer.trace(glm::vec3(0.5f), glm::vec3(-0.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 17);

		result = tracer.trace(glm::vec3(0.5f), glm::vec3(2.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 19);

		size_t max_idx = 0.f;
		for (size_t idx : result)
			if (idx > max_idx)
				max_idx = idx;
		EXPECT_EQ(max_idx, field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z - 1);
	}

	TEST(LineTracing, TraceBigField) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.001));
		auto buffer = field.add_channel("test");
		LinetracingGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(0.f), glm::vec3(1.f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		// the channel has the field's 1000³ voxels (it had 999³ before VoxelGrid rounded like CartesianRadiationField)
		// the exact diagonal runs through voxel corners, so neighbours touching it at the corners count too
		EXPECT_EQ(result.size(), 3873);
	}
}