from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["marigold_image_processing"] = ["MarigoldImageProcessor"]
    _import_structure["pipeline_marigold_depth"] = ["MarigoldDepthOutput", "MarigoldDepthPipeline"]
    _import_structure["pipeline_marigold_intrinsics"] = ["MarigoldIntrinsicsOutput", "MarigoldIntrinsicsPipeline"]
    _import_structure["pipeline_marigold_normals"] = ["MarigoldNormalsOutput", "MarigoldNormalsPipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .marigold_image_processing import MarigoldImageProcessor
        from .pipeline_marigold_depth import MarigoldDepthOutput, MarigoldDepthPipeline
        from .pipeline_marigold_intrinsics import MarigoldIntrinsicsOutput, MarigoldIntrinsicsPipeline
        from .pipeline_marigold_normals import MarigoldNormalsOutput, MarigoldNormalsPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
