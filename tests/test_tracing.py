from RadFiled3D.RadFiled3D import vec3, GridTracer, GridTracerFactory, GridTracerAlgorithm, CartesianRadiationField


def test_sampling_algorithm():
    field = CartesianRadiationField(vec3(1.0, 1.0, 1.0), vec3(0.01, 0.01, 0.01))
    field.add_channel("test")

    tracer = GridTracerFactory.construct(field, GridTracerAlgorithm.SAMPLING)
    assert tracer is not None

    indices = tracer.trace(vec3(0.5, 0.5, 0.0), vec3(0.5, 0.5, 1.0))
    assert len(indices) == 99, f"Expected 99 (instead of 100) indices when tracing a straight line from bottom to top, but got {len(indices)}"

    indices = tracer.trace(vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5))
    assert len(indices) == 0, f"Expected no indices when tracing from a point to itself, but got {len(indices)}"

    indices = tracer.trace(vec3(2.0, 2.0, 2.0), vec3(3.5, 3.5, 3.0))
    assert len(indices) == 0, f"Expected no indices when tracing outside the grid (positive values), but got {len(indices)}"

    indices = tracer.trace(vec3(-2.0, -2.0, -2.0), vec3(-3.5, -3.5, -3.0))
    assert len(indices) == 0, f"Expected no indices when tracing outside the grid (negative values), but got {len(indices)}"

    indices = tracer.trace(vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 1.0))
    assert len(indices) == 49, f"Expected 49 (instead of 50) indices when tracing a straight line from the middle to the top, but got {len(indices)}"

    indices = tracer.trace(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0))
    assert len(indices) == 99, f"Expected 99 (instead of 100 as sampling is only counting incident hits) indices when tracing a straight line from the bottom to the top, but got {len(indices)}"


def test_bresenham_algorithm():
    field = CartesianRadiationField(vec3(1.0, 1.0, 1.0), vec3(0.01, 0.01, 0.01))
    field.add_channel("test")

    tracer = GridTracerFactory.construct(field, GridTracerAlgorithm.BRESENHAM)
    assert tracer is not None

    indices = tracer.trace(vec3(0.5, 0.5, 0.0), vec3(0.5, 0.5, 1.0))
    assert len(indices) == 99, f"Expected 99 indices when tracing a straight line from bottom to top, but got {len(indices)}"

    indices = tracer.trace(vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5))
    assert len(indices) == 0, f"Expected no indices when tracing from a point to itself, but got {len(indices)}"

    indices = tracer.trace(vec3(2.0, 2.0, 2.0), vec3(3.5, 3.5, 3.0))
    assert len(indices) == 0, f"Expected no indices when tracing outside the grid (positive values), but got {len(indices)}"

    indices = tracer.trace(vec3(-2.0, -2.0, -2.0), vec3(-3.5, -3.5, -3.0))
    assert len(indices) == 0, f"Expected no indices when tracing outside the grid (negative values), but got {len(indices)}"

    indices = tracer.trace(vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 1.0))
    assert len(indices) == 49, f"Expected 50 indices when tracing a straight line from the middle to the top, but got {len(indices)}"

    indices = tracer.trace(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0))
    assert len(indices) == 99, f"Expected 99 indices when tracing a straight line from the bottom to the top, but got {len(indices)}"
