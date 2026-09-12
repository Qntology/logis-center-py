from transformers.models.auto import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    AutoImageProcessor,
)

from .configuration_ax_ve import AXVEConfig
from .configuration_ax_ve import AXVEVisionConfig

from .modeling_ax_ve import AXVEVisionModel
from .processing_ax_ve import AXVEProcessor
from .image_processing_ax_ve import AXVEImageProcessor

AutoConfig.register(AXVEConfig.model_type, AXVEConfig)
AutoConfig.register(AXVEVisionConfig.model_type, AXVEVisionConfig)
AutoModel.register(AXVEVisionConfig, AXVEVisionModel)
AutoProcessor.register(AXVEConfig, AXVEProcessor)
AutoImageProcessor.register(AXVEConfig, AXVEImageProcessor)

print("A.X VE vision encoder registered")
