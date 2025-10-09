import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "pyCMakeSetupTool"))

from pyCMakeSetupTool import setup, CMakeBuilder
import platform


major_minor_version = '.'.join(platform.python_version().split('.')[:2])

setup(
    name="RadFiled3D",
    version="1.0.0",
    dependencies=[
        "numpy>=2.0",
        "rich>=13.9"
    ],
    author="Felix Lehner",
    author_email="felix.lehner@ptb.de",
    license=open(os.path.join(os.path.dirname(__file__), "LICENSE")).read(),
    min_python="3.11",
    description=open(os.path.join(os.path.dirname(__file__), "README.md")).read(),
    cmake_builder=CMakeBuilder(
        build_target="RadFiled3D",
        stubs_dir=os.path.join(os.path.dirname(__file__), "python/stubs"),
        module_folder=os.path.join(os.path.dirname(__file__), "RadFiled3D"),
        project_dir=os.path.dirname(__file__),
        dependencies_file=None,
        partial_native_python_module_folder=os.path.join(os.path.dirname(__file__), "python/RadFiled3D"),
        cmake_parameters_configure=[
            ("BUILD_TESTS", "OFF"),
            ("BUILD_EXAMPLES", "OFF"),
            ("PYTHON_VERSION", major_minor_version),
            ("Python_ROOT_DIR", sys.exec_prefix)
        ]
    )
)
