#include <RadFiled3D/storage/RadiationFieldStore.hpp>
#include <RadFiled3D/RadiationField.hpp>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <memory>


using namespace RadFiled3D;
using namespace RadFiled3D::Storage;

int main() {
	// Create a radiation field with dimensions 2.5x2.5x2.5 and voxel dimensions 0.05x0.05x0.05.
	// Be aware, that most user-level methods of the RadiationData namespace are working on shared pointers to avoid memory leaks.
	// Only on Voxel-Level there are references used to avoid overhead. So it is stringly encuraged to avoid copying voxels to local variables. Keep the references and release them as soon as possible.
	std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));

	// Add a channel to the radiation field called "a channel" (could be used to separate different observations like the scattering of a specific object or the origin x-ray beam energy contribution)
	field->add_channel("a channel");
	std::shared_ptr<VoxelGridBuffer> channel = field->get_channel("a channel"); // NOTE: the get_channel method returns the VoxelBuffer of the type of the RadiationField. In this case it returns a VoxelGridBuffer.
	
	// Add a layer to the channel called "doserate" with initial value 0.f and unit "Gy/s"
	channel->add_layer<float>("doserate", 0.f, "Gy/s");
	// Add a layer to the channel called "directions" with initial value glm::vec3(0.f) and unit "normalized PCA direction"
	channel->add_layer<glm::vec3>("direction", glm::vec3(0.f), "normalized PCA direction");
	// Add a layer to the channel called "spectrum" which is not a simple scalar value voxel, but a more complex voxel with an custom constructor
	channel->add_custom_layer("spectrum", HistogramVoxel<float>(26, 10.f, nullptr), 0.f, "keV");

	// Access the "doserate" layer and set the value of the voxel at index (0, 5, 0) to 123.f
	channel->get_voxel<ScalarVoxel<float>>("doserate", 0, 5, 0) = 123.f;

	// It is also possible, to access the voxel by its coordinate in 3D space, assuming, that the voxel grids origin is at (0, 0, 0)
	channel->get_voxel_by_coord<ScalarVoxel<float>>("doserate", 1.f, 0.5, 1.8f) = 0.25f;

	// Access the "direction" layer and set the value of the voxel at index (0, 5, 0) to glm::vec3(1.f, 2.f, 3.f).
	// NOTE: All inplace arithmetic operations are supported by the voxel types, when provided with an instance of their value type.
	channel->get_voxel<ScalarVoxel<glm::vec3>>("direction", 0, 5, 0) += glm::vec3(1.f, 2.f, 3.f);

	// Access the "spectrum" layer and get the histogram data
	auto hist = channel->get_voxel<HistogramVoxel<float>>("spectrum", 0, 5, 0);
	// Request a view to the histogram data that is easy to work with and implements the std::collection interface
	auto hist_data = hist.get_histogram();
	std::cout << "Hist bins: " << hist_data.size() << std::endl;
	std::cout << "Or hist bins from object: " << hist.get_bins() << std::endl;

	// The histogram data is a std::span<float> that can be modified directly
	hist_data[2] = 0.5f;
	std::cout << hist.get_histogram()[2] << std::endl;
	std::cout << hist.get_histogram()[1] << std::endl;

	// Create the metadata object: The metadata object contains information about the simulation and the software used to create the radiation field
	std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
		RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
			0,
			"SomeGeometryFile",
			"FTFP_BERT",
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
				glm::vec3(0.f, 1.f, 0.f),
				glm::vec3(0.f),
				0.f,
				"SomeTubeID"
			)
		),
		RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
			"Example01",
			"DEV",
			"",
			"HEAD"
		)
	);

	// Store the radiation field and the metadata to a file called "test_field.rf3"
	FieldStore::store(field, metadata, "test_field.rf3", StoreVersion::V1);

	// Load the radiation field and the metadata from the file "test_field.rf3"
	auto loaded_field = FieldStore::load("test_field.rf3");

	// Using spherical coordinates work just accordingly to the code above by just using the right RadiationField
	std::shared_ptr<PolarRadiationField> s_field = std::make_shared<PolarRadiationField>(glm::uvec2(32, 32));
	s_field->add_channel("a channel");
	std::shared_ptr<PolarSegmentsBuffer> s_channel = s_field->get_channel("a channel"); // NOTE: Now we got a PolarSegmentsBuffer buffer to work with

	s_channel->add_layer<float>("doserate", 25.3f, "Gy/s");
	ScalarVoxel<float>& s_voxel = s_channel->get_segment_by_coord<ScalarVoxel<float>>("doserate", 0.5f * glm::pi<float>(), 0.5f * glm::pi<float>()) = 123.f;
	// ... and so on...
	return 0;
}