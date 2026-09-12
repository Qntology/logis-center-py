import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import PreTrainedModel, Siglip2VisionModel, Siglip2VisionConfig
from functools import lru_cache

try:
    from .configuration_hayai import HayaiConfig, DEFAULT_SIGLIP2_REF
except ImportError:
    from configuration_hayai import HayaiConfig, DEFAULT_SIGLIP2_REF


_SIGLIP2_FALLBACK_KWARGS = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_channels": 3,
    "patch_size": 16,
    "hidden_act": "gelu_pytorch_tanh",
    "layer_norm_eps": 1e-6,
    "attention_dropout": 0.0,
    "num_patches": 256,
}


def _build_fallback_siglip2_config() -> Siglip2VisionConfig:
    kwargs = dict(_SIGLIP2_FALLBACK_KWARGS)
    try:
        return Siglip2VisionConfig(**kwargs)
    except TypeError:
        kwargs.pop("num_patches", None)
        return Siglip2VisionConfig(**kwargs)


def resolve_siglip2_vision_config(ref) -> Siglip2VisionConfig:
    candidates = []
    if ref:
        candidates.append(str(ref))

    here = Path(__file__).resolve().parent
    local_guess = here.parent / "models" / "siglip2-base-patch16-naflex"
    if str(local_guess) not in candidates:
        candidates.append(str(local_guess))

    if DEFAULT_SIGLIP2_REF not in candidates:
        candidates.append(DEFAULT_SIGLIP2_REF)

    last_err = None
    for cand in candidates:
        p = Path(cand)
        if p.is_dir() and not (p / "config.json").exists():
            continue
        try:
            return Siglip2VisionConfig.from_pretrained(cand)
        except Exception as e:
            last_err = e
            continue

    print(
        "[HayaiModel] SigLIP2 비전 설정을 외부에서 불러오지 못해 내장 기본값을 사용합니다. "
        f"(last error: {last_err})"
    )
    return _build_fallback_siglip2_config()


@lru_cache(maxsize=None)
def _get_2d_visual_freqs(h: int, w: int, d_axis: int, theta: float, device_str: str):
    """Cached 2D RoPE frequencies for a given spatial shape."""
    device = torch.device(device_str)
    freqs = 1.0 / (theta ** (torch.arange(0, d_axis, 2, device=device).float() / d_axis))
    grid_y = torch.arange(h, device=device, dtype=torch.float32)
    grid_x = torch.arange(w, device=device, dtype=torch.float32)
    freqs_y = torch.outer(grid_y, freqs)
    freqs_x = torch.outer(grid_x, freqs)
    grid_y_ext = freqs_y.unsqueeze(1).expand(h, w, -1)
    grid_x_ext = freqs_x.unsqueeze(0).expand(h, w, -1)
    vis_freqs = torch.cat([grid_y_ext, grid_x_ext], dim=-1).flatten(0, 1)  # (h*w, d_axis)
    cos = torch.cos(vis_freqs)
    sin = torch.sin(vis_freqs)
    return cos, sin


@lru_cache(maxsize=None)
def _get_1d_text_freqs(n_text: int, d_axis: int, theta: float, device_str: str):
    """Cached 1D RoPE frequencies for text tokens."""
    device = torch.device(device_str)
    freqs = 1.0 / (theta ** (torch.arange(0, d_axis, 2, device=device).float() / d_axis))
    t_text = torch.arange(n_text, device=device, dtype=torch.float32)
    text_freqs_1d = torch.outer(t_text, freqs)
    text_freqs = torch.cat([text_freqs_1d, text_freqs_1d], dim=-1)  # (n_text, d_axis)
    cos_text = torch.cos(text_freqs)
    sin_text = torch.sin(text_freqs)
    return cos_text, sin_text


def compute_batch_2d_mrope_freqs(spatial_shapes: torch.Tensor, m_vision: int, n_text: int,
                                 d_head: int = 64, theta: float = 10000.0, device="cuda"):
    """Compute batched 2D RoPE cos/sin for vision tokens and 1D RoPE for text tokens."""
    b = spatial_shapes.shape[0]
    d_axis = d_head // 2
    total_len = m_vision + n_text

    cos_batch = torch.ones((b, total_len, d_axis), device=device)
    sin_batch = torch.zeros((b, total_len, d_axis), device=device)

    # Vectorized text frequencies using global LRU cache
    cos_text, sin_text = _get_1d_text_freqs(n_text, d_axis, theta, str(device))
    cos_batch[:, m_vision:, :] = cos_text.unsqueeze(0).expand(b, -1, -1)
    sin_batch[:, m_vision:, :] = sin_text.unsqueeze(0).expand(b, -1, -1)

    # Visual frequencies per sample
    for i in range(b):
        h = spatial_shapes[i, 0].item()
        w = spatial_shapes[i, 1].item()
        actual_vis = min(h * w, m_vision)
        if actual_vis <= 0:
            continue
        cos_vis, sin_vis = _get_2d_visual_freqs(h, w, d_axis, theta, str(device))
        cos_batch[i, :actual_vis] = cos_vis[:actual_vis]
        sin_batch[i, :actual_vis] = sin_vis[:actual_vis]

    return cos_batch, sin_batch


def apply_rotary_emb_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Optimized RoPE application using complex number multiplication."""
    orig_dtype = x.dtype
    x = x.float()
    
    # Ensure contiguity for view_as_complex
    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
    freqs_complex = torch.view_as_complex(torch.stack([cos, sin], dim=-1).float().contiguous())
    
    # Broadcast and multiply
    out_complex = x_complex * freqs_complex.unsqueeze(2)
    out = torch.view_as_real(out_complex).flatten(-2)
    return out.to(orig_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.has_fused_rms_norm = hasattr(F, "rms_norm")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_fused_rms_norm:
            return F.rms_norm(x, (self.weight.numel(),), self.weight, self.eps)
            
        orig_dtype = x.dtype
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(variance + self.eps)
        return (normed.to(orig_dtype)) * self.weight


class MLPProjector(nn.Module):
    def __init__(self, d_vision: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_vision, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, h_q: int = 8, h_kv: int = 2, d_head: int = 64):
        super().__init__()
        self.h_q = h_q
        self.h_kv = h_kv
        self.d_head = d_head
        self.num_queries_per_kv = h_q // h_kv
        self.w_q = nn.Linear(d_model, h_q * d_head, bias=False)
        self.w_k = nn.Linear(d_model, h_kv * d_head, bias=False)
        self.w_v = nn.Linear(d_model, h_kv * d_head, bias=False)
        self.w_o = nn.Linear(h_q * d_head, d_model, bias=False)
        self.q_norm = RMSNorm(d_head)
        self.k_norm = RMSNorm(d_head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                cos_sin: tuple = None, kv_cache: dict = None,
                layer_idx: int = None, cache_seqlens: int = 0) -> torch.Tensor:
        b, s, _ = x.shape

        q = self.w_q(x).view(b, s, self.h_q, self.d_head)
        k = self.w_k(x).view(b, s, self.h_kv, self.d_head)
        v = self.w_v(x).view(b, s, self.h_kv, self.d_head)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if cos_sin is not None:
            cos, sin = cos_sin
            q = apply_rotary_emb_2d(q, cos, sin)
            k = apply_rotary_emb_2d(k, cos, sin)

        # STATIC KV CACHE implementation
        if kv_cache is not None:
            k_cache, v_cache = kv_cache[layer_idx]
            # Write new keys/values into the preallocated cache
            k_t_write = k.transpose(1, 2)
            v_t_write = v.transpose(1, 2)
            
            k_cache[:, :, cache_seqlens:cache_seqlens + s] = k_t_write
            v_cache[:, :, cache_seqlens:cache_seqlens + s] = v_t_write
            
            total_len = cache_seqlens + s
            # Slice and make contiguous for FlashAttention
            k = k_cache[:, :, :total_len].contiguous()
            v = v_cache[:, :, :total_len].contiguous()
        
        q_t = q.transpose(1, 2)              # (b, h_q, s, d)
        k_t = k                              # (b, h_kv, total_len, d)
        v_t = v                              # (b, h_kv, total_len, d)

        sdpa_kwargs = {
            "attn_mask": mask,
            "dropout_p": 0.0,
            "is_causal": False,
        }

        # Try to use native GQA support (PyTorch 2.5+) to avoid expensive repeat_interleave
        if self.h_kv != self.h_q:
            try:
                context = F.scaled_dot_product_attention(
                    q_t, k_t, v_t, enable_gqa=True, **sdpa_kwargs
                )
            except TypeError:
                # Fallback for older PyTorch versions
                k_t = k_t.repeat_interleave(self.num_queries_per_kv, dim=1)
                v_t = v_t.repeat_interleave(self.num_queries_per_kv, dim=1)
                context = F.scaled_dot_product_attention(q_t, k_t, v_t, **sdpa_kwargs)
        else:
            context = F.scaled_dot_product_attention(q_t, k_t, v_t, **sdpa_kwargs)

        context = context.transpose(1, 2).contiguous().view(b, s, -1)
        return self.w_o(context)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, h_q: int, h_kv: int, d_ffn: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, h_q, h_kv)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ffn)
        self.attn_res_scale = nn.Parameter(torch.ones(d_model))
        self.ffn_res_scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                cos_sin: tuple = None, kv_cache: dict = None,
                layer_idx: int = None, cache_seqlens: int = 0) -> torch.Tensor:
        x = x + self.attn_res_scale * self.attn(
            self.attn_norm(x), mask=mask, cos_sin=cos_sin,
            kv_cache=kv_cache, layer_idx=layer_idx, cache_seqlens=cache_seqlens
        )
        x = x + self.ffn_res_scale * self.ffn(self.ffn_norm(x))
        return x


class VisualCausalOCRDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, d_vision: int = 768,
                 d_ffn: int = 2048, n_layers: int = 12):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.projector = MLPProjector(d_vision, d_model)
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, h_q=8, h_kv=2, d_ffn=d_ffn)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        self._mask_cache = {}

    def generate_block_causal_mask(self, m_vision: int, n_text: int, device) -> torch.Tensor:
        key = (m_vision, n_text, str(device))
        if key not in self._mask_cache:
            total_len = m_vision + n_text
            # Use float additive mask (-1e9) instead of boolean for FlashAttention performance
            mask = torch.zeros((total_len, total_len), dtype=torch.float32, device=device)
            mask[:m_vision, m_vision:] = -1e9
            
            text_causal = torch.triu(torch.ones((n_text, n_text), dtype=torch.float32, device=device), diagonal=1) * -1e9
            mask[m_vision:, m_vision:] = text_causal
            
            mask = mask.unsqueeze(0).unsqueeze(0)
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(self, visual_features: torch.Tensor, spatial_shapes: torch.Tensor,
                text_token_ids: torch.Tensor) -> torch.Tensor:
        b, m_vision, _ = visual_features.shape
        _, n_text = text_token_ids.shape

        vision_embeddings = self.projector(visual_features)
        text_embeddings = self.token_embeddings(text_token_ids)
        x = torch.cat([vision_embeddings, text_embeddings], dim=1)

        mask = self.generate_block_causal_mask(m_vision, n_text, x.device)
        cos_batch, sin_batch = compute_batch_2d_mrope_freqs(
            spatial_shapes, m_vision, n_text, d_head=64, device=x.device
        )
        cos_sin = (cos_batch, sin_batch)

        for layer in self.layers:
            x = layer(x, mask=mask, cos_sin=cos_sin, cache_seqlens=0)

        text_outputs = x[:, m_vision:]
        text_outputs = self.final_norm(text_outputs)
        logits = self.output_head(text_outputs)
        return logits


class HayaiModel(PreTrainedModel):
    config_class = HayaiConfig

    def __init__(self, config: HayaiConfig):
        super().__init__(config)
        siglip2_ref = getattr(config, "siglip2_ref", DEFAULT_SIGLIP2_REF)
        vision_config = resolve_siglip2_vision_config(siglip2_ref)
        self.vision_encoder = Siglip2VisionModel(vision_config)
        self.decoder = VisualCausalOCRDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_vision=config.d_vision,
            d_ffn=config.d_ffn,
            n_layers=config.n_layers
        )
        self.post_init()
        self.tie_weights()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        has_wrapper = any(
            k.startswith("vision_encoder.vision_model.") for k in self.state_dict().keys()
        )
        ckpt_has_wrapper = any(
            k.startswith("vision_encoder.vision_model.") for k in state_dict.keys()
        )
        if has_wrapper and not ckpt_has_wrapper:
            state_dict = {
                (k.replace("vision_encoder.", "vision_encoder.vision_model.", 1)
                 if k.startswith("vision_encoder.") and not k.startswith("vision_encoder.vision_model.")
                 else k): v
                for k, v in state_dict.items()
            }
        elif not has_wrapper and ckpt_has_wrapper:
            state_dict = {
                k.replace("vision_encoder.vision_model.", "vision_encoder.", 1): v
                for k, v in state_dict.items()
            }
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def get_input_embeddings(self):
        return self.decoder.token_embeddings

    def set_input_embeddings(self, value):
        self.decoder.token_embeddings = value

    def get_output_embeddings(self):
        return self.decoder.output_head

    def set_output_embeddings(self, new_embeddings):
        self.decoder.output_head = new_embeddings

    def forward(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor,
                spatial_shapes: torch.Tensor, text_token_ids: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_encoder(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes
        )
        visual_features = vision_outputs.last_hidden_state
        logits = self.decoder(visual_features, spatial_shapes, text_token_ids)
        return logits

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor,
                 spatial_shapes: torch.Tensor, tokenizer,
                 max_new_tokens: int = 256, num_beams: int = 1,
                 repetition_penalty: float = 1.00, length_penalty: float = 1.0,
                 early_stopping: bool = True) -> list:
        """Autoregressive generation supporting Greedy Search and Beam Search with Static KV Cache."""
        device = pixel_values.device
        b = pixel_values.size(0)

        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        amp_dtype = torch.float16 if device.type == "cuda" else torch.float32

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            # 1. Vision Encoder forward
            vision_outputs = self.vision_encoder(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes
            )
            visual_features = vision_outputs.last_hidden_state
            m_vision = visual_features.size(1)

            # 2. Prepare prefill inputs (Vision + BOS)
            bos_tokens = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
            vision_embeddings = self.decoder.projector(visual_features)
            bos_embeddings = self.decoder.token_embeddings(bos_tokens)
            x = torch.cat([vision_embeddings, bos_embeddings], dim=1)
            
            # Use model computation dtype for cache allocation
            amp_dtype = x.dtype 

            # Precompute text RoPE for positions 0..max_new_tokens
            d_axis = self.decoder.layers[0].attn.d_head // 2
            freqs = 1.0 / (10000.0 ** (torch.arange(0, d_axis, 2, device=device).float() / d_axis))
            t_text_all = torch.arange(max_new_tokens + 1, device=device, dtype=torch.float32)
            text_freqs_1d_all = torch.outer(t_text_all, freqs)
            text_freqs_all = torch.cat([text_freqs_1d_all, text_freqs_1d_all], dim=-1)
            cos_text_all = torch.cos(text_freqs_all)
            sin_text_all = torch.sin(text_freqs_all)

            mask = self.decoder.generate_block_causal_mask(m_vision, 1, device)
            cos_batch, sin_batch = compute_batch_2d_mrope_freqs(
                spatial_shapes, m_vision, 1, d_head=64, device=device
            )

            # STATIC KV CACHE preallocation
            max_seq_len = m_vision + max_new_tokens + 1
            kv_cache = {}
            for i, layer in enumerate(self.decoder.layers):
                k_cache = torch.zeros((b, layer.attn.h_kv, max_seq_len, layer.attn.d_head), 
                                      dtype=amp_dtype, device=device)
                v_cache = torch.zeros((b, layer.attn.h_kv, max_seq_len, layer.attn.d_head), 
                                      dtype=amp_dtype, device=device)
                kv_cache[i] = (k_cache, v_cache)
                
            cache_seqlens = 0
            
            # Prefill pass
            for i, layer in enumerate(self.decoder.layers):
                x = layer(x, mask=mask, cos_sin=(cos_batch, sin_batch),
                          kv_cache=kv_cache, layer_idx=i, cache_seqlens=cache_seqlens)
            cache_seqlens += x.size(1)

            x_bos = x[:, -1:]
            x_bos = self.decoder.final_norm(x_bos)
            logits = self.decoder.output_head(x_bos)
            next_token_logits = logits[:, -1, :].clone()

            # -------------------------------------------------------------
            # Path A: Fast Greedy Decoding (num_beams == 1)
            # -------------------------------------------------------------
            if num_beams == 1:
                if repetition_penalty != 1.0:
                    penalty = torch.full_like(next_token_logits, 1.0)
                    penalty[:, bos_id] = repetition_penalty
                    next_token_logits = torch.where(
                        next_token_logits < 0,
                        next_token_logits * penalty,
                        next_token_logits / penalty,
                    )

                next_tokens = torch.argmax(next_token_logits, dim=-1)
                generated_tokens = torch.full(
                    (b, max_new_tokens + 1), pad_id, dtype=torch.long, device=device
                )
                generated_tokens[:, 0] = bos_id
                generated_tokens[:, 1] = next_tokens
                unfinished = (next_tokens != eos_id) & (next_tokens != pad_id)

                for step in range(1, max_new_tokens):
                    if not unfinished.any():
                        break

                    cur_input_tokens = next_tokens.unsqueeze(1)
                    x_step = self.decoder.token_embeddings(cur_input_tokens)

                    cos_step = cos_text_all[step].unsqueeze(0).expand(b, 1, -1)
                    sin_step = sin_text_all[step].unsqueeze(0).expand(b, 1, -1)

                    for i, layer in enumerate(self.decoder.layers):
                        x_step = layer(x_step, mask=None, cos_sin=(cos_step, sin_step),
                                       kv_cache=kv_cache, layer_idx=i, cache_seqlens=cache_seqlens)
                    cache_seqlens += 1

                    x_step = self.decoder.final_norm(x_step)
                    logits_step = self.decoder.output_head(x_step)[:, -1, :].clone()

                    if repetition_penalty != 1.0:
                        penalty = torch.ones_like(logits_step)
                        penalty = penalty.scatter(1, generated_tokens[:, :step+1], repetition_penalty)
                        logits_step = torch.where(
                            logits_step < 0,
                            logits_step * penalty,
                            logits_step / penalty,
                        )

                    next_tokens = torch.argmax(logits_step, dim=-1)
                    next_tokens = next_tokens * unfinished + pad_id * (~unfinished)
                    generated_tokens[:, step+1] = next_tokens
                    unfinished = unfinished & (next_tokens != eos_id) & (next_tokens != pad_id)

                results = []
                for seq in generated_tokens:
                    clean_tokens = [t for t in seq.tolist()[1:] if t not in (eos_id, pad_id)]
                    results.append(tokenizer.decode(clean_tokens, skip_special_tokens=True))
                return results

            # -------------------------------------------------------------
            # Path B: Beam Search (num_beams > 1)
            # -------------------------------------------------------------
            # Expand Static KV cache batch dimension to hold `b * num_beams` entries in-place
            for layer_idx in list(kv_cache.keys()):
                k_cache, v_cache = kv_cache[layer_idx]
                kv_cache[layer_idx] = (
                    k_cache.repeat_interleave(num_beams, dim=0),
                    v_cache.repeat_interleave(num_beams, dim=0)
                )

            # Apply repetition penalty to step 0
            if repetition_penalty != 1.0:
                penalty = torch.full_like(next_token_logits, 1.0)
                penalty[:, bos_id] = repetition_penalty
                next_token_logits = torch.where(
                    next_token_logits < 0,
                    next_token_logits * penalty,
                    next_token_logits / penalty,
                )

            log_probs = F.log_softmax(next_token_logits, dim=-1)

            beam_scores = torch.zeros((b, num_beams), dtype=torch.float32, device=device)
            beam_scores[:, 1:] = -1e9

            running_sequences = torch.full((b, num_beams, max_new_tokens + 1), pad_id, dtype=torch.long, device=device)
            running_sequences[:, :, 0] = bos_id

            done_hypotheses = [[] for _ in range(b)]
            vocab_size = self.decoder.vocab_size

            topk_scores, topk_tokens = torch.topk(log_probs, num_beams, dim=-1)
            beam_scores = topk_scores
            running_sequences[:, :, 1] = topk_tokens

            cur_input_tokens = topk_tokens.view(b * num_beams, 1)

            for batch_idx in range(b):
                for beam_idx in range(num_beams):
                    tok = topk_tokens[batch_idx, beam_idx].item()
                    if tok == eos_id:
                        score = (beam_scores[batch_idx, beam_idx].item()) / (1.0 ** length_penalty)
                        done_hypotheses[batch_idx].append((score, [tok]))
                        beam_scores[batch_idx, beam_idx] = -1e9

            for step in range(1, max_new_tokens):
                if early_stopping and all(len(d) >= num_beams for d in done_hypotheses):
                    break

                x_step = self.decoder.token_embeddings(cur_input_tokens)
                cos_step = cos_text_all[step].unsqueeze(0).expand(b * num_beams, 1, -1)
                sin_step = sin_text_all[step].unsqueeze(0).expand(b * num_beams, 1, -1)

                for i, layer in enumerate(self.decoder.layers):
                    x_step = layer(x_step, mask=None, cos_sin=(cos_step, sin_step),
                                   kv_cache=kv_cache, layer_idx=i, cache_seqlens=cache_seqlens)
                cache_seqlens += 1

                x_step = self.decoder.final_norm(x_step)
                logits_step = self.decoder.output_head(x_step)[:, -1, :].clone()

                if repetition_penalty != 1.0:
                    flat_seqs = running_sequences.view(b * num_beams, -1)[:, :step+1]
                    penalty = torch.ones_like(logits_step)
                    penalty = penalty.scatter(1, flat_seqs, repetition_penalty)
                    logits_step = torch.where(
                        logits_step < 0,
                        logits_step * penalty,
                        logits_step / penalty,
                    )

                next_log_probs = F.log_softmax(logits_step, dim=-1)
                next_log_probs = next_log_probs.view(b, num_beams, vocab_size)

                next_scores = beam_scores.unsqueeze(-1) + next_log_probs

                new_beam_tokens = torch.zeros((b, num_beams), dtype=torch.long, device=device)
                new_beam_scores = torch.zeros((b, num_beams), dtype=torch.float32, device=device)
                selected_parent_beams = []

                for batch_idx in range(b):
                    flat_scores = next_scores[batch_idx].view(-1)
                    cand_scores, cand_indices = torch.topk(flat_scores, 2 * num_beams, largest=True, sorted=True)

                    active_beam_cnt = 0
                    for cand_score, cand_idx in zip(cand_scores, cand_indices):
                        parent_beam = (cand_idx // vocab_size).item()
                        tok_id = (cand_idx % vocab_size).item()

                        if tok_id == eos_id:
                            cur_len = step + 1
                            norm_score = cand_score.item() / (cur_len ** length_penalty)
                            full_seq = running_sequences[batch_idx, parent_beam, 1:step+1].tolist() + [tok_id]
                            done_hypotheses[batch_idx].append((norm_score, full_seq))
                        else:
                            new_beam_scores[batch_idx, active_beam_cnt] = cand_score
                            new_beam_tokens[batch_idx, active_beam_cnt] = tok_id
                            running_sequences[batch_idx, active_beam_cnt, :step+1] = \
                                running_sequences[batch_idx, parent_beam, :step+1]
                            running_sequences[batch_idx, active_beam_cnt, step+1] = tok_id

                            selected_parent_beams.append(batch_idx * num_beams + parent_beam)
                            active_beam_cnt += 1

                            if active_beam_cnt == num_beams:
                                break

                    while active_beam_cnt < num_beams:
                        new_beam_scores[batch_idx, active_beam_cnt] = -1e9
                        new_beam_tokens[batch_idx, active_beam_cnt] = pad_id
                        selected_parent_beams.append(batch_idx * num_beams)
                        active_beam_cnt += 1

                beam_scores = new_beam_scores
                cur_input_tokens = new_beam_tokens.view(b * num_beams, 1)

                # IN-PLACE Reorder Static KV Cache across parent beams
                reorder_idx = torch.tensor(selected_parent_beams, dtype=torch.long, device=device)
                for layer_idx in kv_cache:
                    k_cache, v_cache = kv_cache[layer_idx]
                    k_cache.copy_(k_cache.index_select(0, reorder_idx))
                    v_cache.copy_(v_cache.index_select(0, reorder_idx))

        # 3. Final Hypothesis Selection & Decode
        results = []
        for batch_idx in range(b):
            if len(done_hypotheses[batch_idx]) > 0:
                best_score, best_seq = max(done_hypotheses[batch_idx], key=lambda x: x[0])
            else:
                best_beam_idx = torch.argmax(beam_scores[batch_idx]).item()
                best_seq = running_sequences[batch_idx, best_beam_idx, 1:].tolist()

            clean_tokens = [t for t in best_seq if t not in (eos_id, pad_id)]
            results.append(tokenizer.decode(clean_tokens, skip_special_tokens=True))

        return results

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model._fixup_vision_encoder_weights(pretrained_model_name_or_path, **kwargs)
        return model

    def _collect_local_state_dict(self, base_dir: Path):
        raw_sd = {}

        shards = sorted(base_dir.glob("*.safetensors"))
        if shards:
            from safetensors.torch import load_file
            for shard in shards:
                try:
                    raw_sd.update(load_file(str(shard)))
                except Exception:
                    continue
            if raw_sd:
                return raw_sd

        for name in ("pytorch_model.bin", "model.bin"):
            p = base_dir / name
            if p.exists():
                try:
                    raw_sd.update(torch.load(str(p), map_location="cpu"))
                except Exception:
                    pass
        return raw_sd

    def _fixup_vision_encoder_weights(self, pretrained_model_name_or_path, **kwargs):
        raw_sd = {}
        base = Path(str(pretrained_model_name_or_path))

        if base.is_dir():
            raw_sd = self._collect_local_state_dict(base)
        else:
            from transformers.utils import cached_file

            revision = kwargs.get("revision", None)
            token = kwargs.get("token", kwargs.get("use_auth_token", None))

            weights_path = cached_file(
                pretrained_model_name_or_path, "model.safetensors",
                revision=revision, token=token,
                _raise_exceptions_for_missing_entries=False,
            )
            if weights_path is not None:
                from safetensors.torch import load_file
                raw_sd = load_file(weights_path)
            else:
                weights_path = cached_file(
                    pretrained_model_name_or_path, "pytorch_model.bin",
                    revision=revision, token=token,
                    _raise_exceptions_for_missing_entries=False,
                )
                if weights_path is None:
                    return
                raw_sd = torch.load(weights_path, map_location="cpu")

        if not raw_sd:
            return

        raw_vision_keys = {
            k[len("vision_encoder."):]: v
            for k, v in raw_sd.items()
            if k.startswith("vision_encoder.")
        }
        if not raw_vision_keys:
            return

        target_keys = set(self.vision_encoder.state_dict().keys())
        target_has_wrapper = any(k.startswith("vision_model.") for k in target_keys)

        remapped = {}
        for k, v in raw_vision_keys.items():
            ckpt_has_wrapper = k.startswith("vision_model.")
            if target_has_wrapper and not ckpt_has_wrapper:
                k = "vision_model." + k
            elif not target_has_wrapper and ckpt_has_wrapper:
                k = k[len("vision_model."):]
            remapped[k] = v

        missing, unexpected = self.vision_encoder.load_state_dict(remapped, strict=False)
        real_missing = [k for k in missing if not k.endswith("position_ids")]
        if real_missing:
            import warnings
            warnings.warn(
                f"HayaiModel: vision encoder weight fixup still missing {len(real_missing)} "
                f"keys after remapping: {real_missing[:5]}{'...' if len(real_missing) > 5 else ''}"
            )