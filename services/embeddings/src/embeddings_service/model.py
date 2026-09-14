"""Model wrapper — the one place that knows about a specific embedding model.

The rest of the service only knows "give me a vector for a text in the role
of a query or a passage" — see docs/architecture/05-contracts.md. Whether
that role needs a prefix, what dimension comes out, and how normalization
works are all decided here, per model.
"""

from __future__ import annotations

from typing import Literal, Protocol

Kind = Literal["query", "passage"]


class EmbeddingModel(Protocol):
    """What the route layer needs — real or fake."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def max_seq_length(self) -> int: ...

    @property
    def tokenizer_name(self) -> str: ...

    @property
    def requires_prefix(self) -> bool: ...

    def embed(self, texts: list[str], kind: Kind) -> list[list[float]]: ...


# e5-family models are trained with "query: " / "passage: " prefixes and lose
# meaningful accuracy without them; paraphrase-mpnet was not trained with any
# and silently ignores the concept. Detected by name because that is the
# only signal the model itself gives us — there is no flag on the weights.
_E5_PREFIXES: dict[Kind, str] = {"query": "query: ", "passage": "passage: "}


def _model_requires_prefix(model_name: str) -> bool:
    return "e5" in model_name.lower()


class SentenceTransformerModel:
    """Loads one SentenceTransformer and serves it for the process lifetime.

    Constructing this is expensive (seconds, once weights are on disk) — see
    docs/architecture/06-deployment.md on why /readyz waits for it. Import of
    sentence_transformers is deferred into __init__ so tests that only need
    the Protocol (via a fake) never pay for torch.
    """

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._requires_prefix = _model_requires_prefix(model_name)
        self._st = SentenceTransformer(model_name, cache_folder=cache_dir, trust_remote_code=False)

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._st.get_sentence_embedding_dimension()

    @property
    def max_seq_length(self) -> int:
        return self._st.get_max_seq_length()

    @property
    def tokenizer_name(self) -> str:
        return type(self._st.tokenizer).__name__

    @property
    def requires_prefix(self) -> bool:
        return self._requires_prefix

    def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        prefixed = self._with_prefix(texts, kind)
        # normalize_embeddings=True: cosine distance and dot product agree,
        # so `embedding <=> :query_vector` in Postgres needs no extra
        # transform — see 05-contracts.md.
        vectors = self._st.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()

    def _with_prefix(self, texts: list[str], kind: Kind) -> list[str]:
        if not self._requires_prefix:
            return texts
        prefix = _E5_PREFIXES[kind]
        return [prefix + t for t in texts]
