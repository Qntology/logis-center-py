import numpy as np
from typing import List, Dict, Tuple, Optional


class FieldScorer:
    def __init__(self, schema: dict):
        self.domain = schema.get("domain", "unknown")
        self.doc_type = schema.get("doc_type", "unknown")
        self.fields = schema.get("fields", {})

    def get_field_names(self) -> List[str]:
        return list(self.fields.keys())

    def get_bias_phrases(self, field_name: str) -> List[str]:
        field = self.fields.get(field_name, {})
        semantic = field.get("semantic", "")
        bias_list = field.get("bias", [])
        phrases = []
        if semantic:
            phrases.append(semantic)
        phrases.extend(bias_list)
        return phrases

    def get_prejudice_phrases(self, field_name: str) -> List[str]:
        field = self.fields.get(field_name, {})
        return field.get("prejudice", [])

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return vec
        return vec / norm

    def score_patches_for_field(
        self,
        patch_embeddings: np.ndarray,
        field_name: str,
        embed_text_fn,
    ) -> Tuple[float, float, float]:
        bias_phrases = self.get_bias_phrases(field_name)
        prejudice_phrases = self.get_prejudice_phrases(field_name)

        if not bias_phrases:
            return 0.0, 0.0, 0.0

        bias_vecs = []
        for phrase in bias_phrases:
            vec = embed_text_fn(phrase)
            if vec is not None and np.any(vec != 0):
                bias_vecs.append(self.l2_normalize(vec))

        if not bias_vecs:
            return 0.0, 0.0, 0.0

        prejudice_vecs = []
        for phrase in prejudice_phrases:
            vec = embed_text_fn(phrase)
            if vec is not None and np.any(vec != 0):
                prejudice_vecs.append(self.l2_normalize(vec))

        bias_scores = []
        for bv in bias_vecs:
            s = self.cosine_similarity(patch_embeddings, bv)
            bias_scores.append(s)
        max_bias = max(bias_scores) if bias_scores else 0.0

        prejudice_scores = []
        for pv in prejudice_vecs:
            s = self.cosine_similarity(patch_embeddings, pv)
            prejudice_scores.append(s)
        max_prejudice = max(prejudice_scores) if prejudice_scores else 0.0

        net_score = max_bias - max_prejudice

        return max_bias, max_prejudice, net_score

    def score_all_patches(
        self,
        patch_embeddings: np.ndarray,
        embed_text_fn,
    ) -> Dict[str, np.ndarray]:
        num_patches = patch_embeddings.shape[0]
        field_scores = {}

        for field_name in self.get_field_names():
            scores = np.zeros(num_patches, dtype=np.float32)
            for i in range(num_patches):
                patch_vec = patch_embeddings[i]
                if np.all(patch_vec == 0):
                    continue
                _, _, net = self.score_patches_for_field(
                    patch_vec, field_name, embed_text_fn
                )
                scores[i] = net
            field_scores[field_name] = scores

        return field_scores