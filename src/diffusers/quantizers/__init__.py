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

from ..utils import is_transformers_available
from .auto import DiffusersAutoQuantizer
from .base import DiffusersQuantizer


class PipelineQuantizationConfig:
    def __init__(self, quant_backend: str, quant_kwargs: dict, exclude_modules: list, mapping: dict = None):
        if mapping is not None:
            self.mapping = mapping
            self.is_granular = True
        else:
            self.quant_backend = quant_backend
            self.quant_kwargs = quant_kwargs or {}
            self.exclude_modules = exclude_modules or []
            self.is_granular = False

        self.post_init()

    def post_init(self):
        if self.is_granular and self.mapping is None:
            raise ValueError(
                "In the granular case, a `mapping` defining the quantization configs"
                " for the desired modules have to be defined."
            )

    def _resolve_quant_config(self, is_diffusers=True, module_name=None):
        if is_diffusers:
            from ..quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING
        else:
            if is_transformers_available():
                from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING

        # Granular case.
        if self.is_granular:
            config = self.mapping.get(module_name)
            quant_backend = config.get("quant_backend")
            if quant_backend not in AUTO_QUANTIZATION_CONFIG_MAPPING:
                raise ValueError(
                    f"Module '{module_name}': Provided quant_backend={quant_backend} was not found. "
                    f"Available ones are: {list(AUTO_QUANTIZATION_CONFIG_MAPPING.keys())}."
                )
            quant_config_cls = AUTO_QUANTIZATION_CONFIG_MAPPING[quant_backend]
            quant_kwargs = config.get("quant_kwargs")

            return quant_config_cls(**quant_kwargs)

        # Global config case.
        elif module_name not in self.exclude_modules:
            quant_backend = self.quant_backend
            if quant_backend not in AUTO_QUANTIZATION_CONFIG_MAPPING:
                raise ValueError(
                    f"Provided quant_backend={quant_backend} was not found. "
                    f"Available ones are: {list(AUTO_QUANTIZATION_CONFIG_MAPPING.keys())}."
                )
            quant_config_cls = AUTO_QUANTIZATION_CONFIG_MAPPING[quant_backend]
            quant_kwargs = self.quant_kwargs
            return quant_config_cls(**quant_kwargs)

        else:
            return None
