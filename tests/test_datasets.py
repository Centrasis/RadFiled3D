try:
    import torch
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False


def test_radfield3d_voxelwise_dataset():
    if TORCH_INSTALLED:
        from RadFiled3D.RadFiled3D import CartesianRadiationField, vec3, DType, FieldShape
        from RadFiled3D.utils import FieldStore, StoreVersion
        from RadFiled3D.metadata.v1 import Metadata
        from RadFiled3D.pytorch.datasets.radfield3d import RadField3DVoxelwiseDataset, TrainingInputData
        import os
        import random
        import numpy as np

        spectrum = np.zeros((150, 2), dtype=np.float32)
        spectrum[:, 0] = np.arange(150, dtype=np.float32) * 10.0
        spectrum[:, 1] = 1.0 / 150.0

        METADATA = Metadata.default()
        METADATA.simulation.tube.max_energy_eV = 1500.0
        METADATA.simulation.tube.spectrum = spectrum
        METADATA.simulation.tube.field_shape = FieldShape.CONE
        METADATA.simulation.tube.opening_angle_deg = 30.0

        field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
        field.add_channel("scatter_field")
        field.add_channel("xray_beam")
        field.get_channel("scatter_field").add_layer("hits", "unit1", DType.FLOAT32)
        field.get_channel("scatter_field").add_layer("error", "unit1", DType.FLOAT32)
        field.get_channel("scatter_field").add_histogram_layer("spectrum", 32, 0.1, "unit1")
        field.get_channel("xray_beam").add_layer("hits", "unit1", DType.FLOAT32)
        field.get_channel("xray_beam").add_layer("error", "unit1", DType.FLOAT32)
        field.get_channel("xray_beam").add_histogram_layer("spectrum", 32, 0.1, "unit1")

        os.makedirs("test_dataset", exist_ok=True)

        FieldStore.store(field, METADATA, "test_dataset/test01.rf3", StoreVersion.V1)
        FieldStore.store(field, METADATA, "test_dataset/test02.rf3", StoreVersion.V1)
        FieldStore.store(field, METADATA, "test_dataset/test03.rf3", StoreVersion.V1)

        dataset = RadField3DVoxelwiseDataset(file_paths=["test_dataset/test01.rf3", "test_dataset/test02.rf3", "test_dataset/test03.rf3"])
        ds_len = 3 * field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z
        assert len(dataset) == ds_len, f"Dataset length does not match expected voxel count: {len(dataset)} != {ds_len}"

        test_in: TrainingInputData = dataset.__getitems__([random.randint(0, len(dataset)) for _ in range(100)])
        assert test_in.ground_truth.scatter_field.error.shape[0] == 100, "Ground truth error shape does not match expected batch size."
        assert test_in.ground_truth.scatter_field.fluence.shape[0] == 100, "Ground truth fluence shape does not match expected batch size."
        assert test_in.ground_truth.scatter_field.spectrum.shape[0] == 100, "Ground truth spectrum shape does not match expected batch size."
        assert test_in.ground_truth.xray_beam.error.shape[0] == 100, "X-ray beam error shape does not match expected batch size."
        assert test_in.ground_truth.xray_beam.fluence.shape[0] == 100, "X-ray beam fluence shape does not match expected batch size."
        assert test_in.ground_truth.xray_beam.spectrum.shape[0] == 100, "X-ray beam spectrum shape does not match expected batch size."
        assert test_in.input.direction.shape[0] == 100, "Input direction shape does not match expected batch size."
        assert test_in.input.position.shape[0] == 100, "Input position shape does not match expected batch size."
        assert test_in.input.spectrum.shape[0] == 100, "Input tube spectrum shape does not match expected batch size."


def test_radfield3d_dataset():
    if TORCH_INSTALLED:
        from RadFiled3D.RadFiled3D import CartesianRadiationField, vec3, DType, FieldShape
        from RadFiled3D.utils import FieldStore, StoreVersion
        from RadFiled3D.metadata.v1 import Metadata
        from RadFiled3D.pytorch.datasets.radfield3d import RadField3DDataset, TrainingInputData
        from RadFiled3D.pytorch.radiationfieldloader import DataLoaderBuilder
        import os
        import numpy as np

        spectrum = np.zeros((150, 2), dtype=np.float32)
        spectrum[:, 0] = np.arange(150, dtype=np.float32) * 10.0
        spectrum[:, 1] = 1.0 / 150.0

        METADATA = Metadata.default()
        METADATA.simulation.tube.max_energy_eV = 1500.0
        METADATA.simulation.tube.spectrum = spectrum
        METADATA.simulation.tube.field_shape = FieldShape.CONE
        METADATA.simulation.tube.opening_angle_deg = 30.0

        field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
        field.add_channel("scatter_field")
        field.add_channel("xray_beam")
        field.get_channel("scatter_field").add_layer("hits", "unit1", DType.FLOAT32)
        field.get_channel("scatter_field").add_layer("error", "unit1", DType.FLOAT32)
        field.get_channel("scatter_field").add_histogram_layer("spectrum", 32, 0.1, "unit1")
        field.get_channel("xray_beam").add_layer("hits", "unit1", DType.FLOAT32)
        field.get_channel("xray_beam").add_layer("error", "unit1", DType.FLOAT32)
        field.get_channel("xray_beam").add_histogram_layer("spectrum", 32, 0.1, "unit1")

        os.makedirs("test_dataset", exist_ok=True)

        for i in range(10):
            FieldStore.store(field, METADATA, f"test_dataset/test0{i}.rf3", StoreVersion.V1)

        dataset = RadField3DDataset(file_paths=[f"test_dataset/test0{i}.rf3" for i in range(10)])
        ds_len = 10
        assert len(dataset) == ds_len, f"Dataset length does not match expected voxel count: {len(dataset)} != {ds_len}"

        builder = DataLoaderBuilder(
            dataset_class=RadField3DDataset,
            dataset_path="test_dataset",
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0
        )
        ds = builder.build_train_dataset()
        assert len(ds) == ds_len, f"Dataset length does not match expected voxel count: {len(ds)} != {ds_len}"

        elem = ds.__getitem__(0)
        assert isinstance(elem, TrainingInputData), "Dataset element is not of type TrainingInputData."
        assert len(elem.input.beam_shape_parameters.shape) == 1, "Beam shape parameters length does not match expected value."
        assert elem.input.beam_shape_parameters[0] == 30.0, "Beam shape parameter does not match expected value."
