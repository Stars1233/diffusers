# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

from ..utils import is_transformers_available, logging
from .auto import DiffusersAutoQuantizer
from .base import DiffusersQuantizer


logger = logging.get_logger(__name__)


class PipelineQuantizationConfig:
    def __init__(
        self,
        quant_backend: str = None,
        quant_kwargs: dict = None,
        modules_to_quantize: list = None,
        mapping: dict = None,
    ):
        self.quant_backend = quant_backend
        # Initialize kwargs to be {} to set to the defaults.
        self.quant_kwargs = quant_kwargs or {}
        self.modules_to_quantize = modules_to_quantize
        self.mapping = mapping

        self.post_init()

    def post_init(self):
        mapping = self.mapping
        self.is_granular = True if mapping is not None else False

        self._validate_init_args()

    def _validate_init_args(self):
        if not self.is_granular and not self.quant_backend:
            raise ValueError("Must provide a `quant_backend` when not providing a `mapping`.")

        if not self.quant_kwargs and not self.mapping:
            raise ValueError("Both `quant_kwargs` and `mapping` cannot be None.")

        if self.quant_backend is not None:
            self._validate_init_kwargs_in_backends()

        if self.mapping is not None:
            self._validate_mapping_args()

    def _validate_init_kwargs_in_backends(self):
        quant_backend = self.quant_backend

        self._check_backend_availability(quant_backend)

        if is_transformers_available():
            from transformers.quantizers.auto import (
                AUTO_QUANTIZATION_CONFIG_MAPPING as QUANT_CONFIG_MAPPING_TRANSFORMERS,
            )
        else:
            QUANT_CONFIG_MAPPING_TRANSFORMERS = None

        from ..quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING as QUANT_CONFIG_MAPPING_DIFFUSERS

        if QUANT_CONFIG_MAPPING_TRANSFORMERS is not None:
            init_kwargs_transformers = inspect.signature(QUANT_CONFIG_MAPPING_TRANSFORMERS[quant_backend].__init__)
            init_kwargs_transformers = {name for name in init_kwargs_transformers.parameters if name != "self"}
        else:
            init_kwargs_transformers = None

        init_kwargs_diffusers = inspect.signature(QUANT_CONFIG_MAPPING_DIFFUSERS[quant_backend].__init__)
        init_kwargs_diffusers = {name for name in init_kwargs_diffusers.parameters if name != "self"}

        if init_kwargs_transformers != init_kwargs_diffusers:
            raise ValueError(
                "The signatures of the __init__ methods of the quantization config classes in `diffusers` and `transformers` don't match. "
                f"Please provide a `mapping` instead in the {self.__class__.__name__} class."
            )

    def _validate_mapping_args(self):
        mapping = self.mapping
        for module_name, config in mapping.items():
            quant_backend = config.get("quant_backend")
            self._check_backend_availability(quant_backend)

    def _check_backend_availability(self, quant_backend: str):
        if is_transformers_available():
            from transformers.quantizers.auto import (
                AUTO_QUANTIZATION_CONFIG_MAPPING as QUANT_CONFIG_MAPPING_TRANSFORMERS,
            )
        else:
            QUANT_CONFIG_MAPPING_TRANSFORMERS = None

        from ..quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING as QUANT_CONFIG_MAPPING_DIFFUSERS

        available_backends_transformers = (
            list(QUANT_CONFIG_MAPPING_TRANSFORMERS.keys()) if QUANT_CONFIG_MAPPING_TRANSFORMERS else None
        )
        available_backends_diffusers = list(QUANT_CONFIG_MAPPING_DIFFUSERS.keys())

        if (
            QUANT_CONFIG_MAPPING_TRANSFORMERS and quant_backend not in QUANT_CONFIG_MAPPING_TRANSFORMERS
        ) or quant_backend not in QUANT_CONFIG_MAPPING_DIFFUSERS:
            error_message = f"Provided quant_backend={quant_backend} was not found."
            if available_backends_transformers:
                error_message += f"\nAvailable ones (transformers): {available_backends_transformers}."
            error_message += f"\nAvailable ones (diffusers): {available_backends_diffusers}."
            raise ValueError(error_message)

    def _resolve_quant_config(self, is_diffusers: bool = True, module_name: str = None):
        if is_diffusers:
            from ..quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING
        else:
            if is_transformers_available():
                from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING
            else:
                raise ValueError("Transformers library is not available.")

        mapping = self.mapping
        modules_to_quantize = self.modules_to_quantize

        # Granular case
        if self.is_granular and module_name in mapping:
            logger.debug(f"Initializing quantization config class for {module_name}.")
            config = mapping[module_name]
            quant_backend = config.get("quant_backend")
            quant_config_cls = AUTO_QUANTIZATION_CONFIG_MAPPING[quant_backend]
            quant_kwargs = config.get("quant_kwargs", {})
            return quant_config_cls(**quant_kwargs)

        # Global config case
        else:
            should_quantize = False
            # Only quantize the modules requested for.
            if modules_to_quantize and module_name in modules_to_quantize:
                should_quantize = True
            # No specification for `modules_to_quantize` == all modules should be quantized.
            elif not self.is_granular and not modules_to_quantize:
                should_quantize = True

            if should_quantize:
                logger.debug(f"Initializing quantization config class for {module_name}.")
                quant_config_cls = AUTO_QUANTIZATION_CONFIG_MAPPING[self.quant_backend]
                quant_kwargs = self.quant_kwargs
                return quant_config_cls(**quant_kwargs)

        # Fallback: no applicable configuration found.
        return None
