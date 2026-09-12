from transformers import PretrainedConfig

DEFAULT_SIGLIP2_REF = "google/siglip2-base-patch16-naflex"


class HayaiConfig(PretrainedConfig):
    model_type = "hayai"

    def __init__(
        self,
        vocab_size=0,
        d_model=512,
        d_vision=768,
        d_ffn=2048,
        n_layers=12,
        siglip2_ref=DEFAULT_SIGLIP2_REF,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_vision = d_vision
        self.d_ffn = d_ffn
        self.n_layers = n_layers
        self.siglip2_ref = siglip2_ref or DEFAULT_SIGLIP2_REF
        super().__init__(**kwargs)
