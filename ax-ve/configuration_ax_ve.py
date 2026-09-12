from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)


class AXVEVisionConfig(PretrainedConfig):
    model_type = "ax_ve"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1152,
        image_size=384,
        intermediate_size=4304,
        num_attention_heads=16,
        num_hidden_layers=27,
        num_channels=3,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        use_native_resolution=True,
        num_position_embeddings=576,
        spatial_merge_size=2, 
        residual_visual_indexes = [6, 12, 18],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.use_native_resolution = use_native_resolution
        self.image_size = image_size
        self.num_position_embeddings = num_position_embeddings
        self.residual_visual_indexes = residual_visual_indexes
        self.spatial_merge_size = spatial_merge_size 


class AXVEConfig(PretrainedConfig):
    model_type = "ax_ve_model"
    sub_configs = {
        "vision_config": AXVEVisionConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_config=None,
        image_token_index=163723,
        pad_token_id=163692,
        tie_word_embeddings=False,
        **kwargs,
    ):

        self.image_token_index = image_token_index
        self.pad_token_id = pad_token_id

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif isinstance(vision_config, PretrainedConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            raise ValueError(f"Invalid vision_config: {vision_config}")

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

__all__ = ["AXVEConfig", "AXVEVisionConfig"]
