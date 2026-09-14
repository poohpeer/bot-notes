"""Pure prefix-detection logic — no model weights involved.

e5-family models need "query: "/"passage: " prefixes; paraphrase-mpnet does
not and ignores the concept — see model.py and 05-contracts.md.
"""

from embeddings_service.model import _model_requires_prefix


def test_e5_model_requires_prefix():
    assert _model_requires_prefix("intfloat/multilingual-e5-base") is True


def test_mpnet_model_does_not_require_prefix():
    assert _model_requires_prefix("paraphrase-multilingual-mpnet-base-v2") is False


def test_detection_is_case_insensitive():
    assert _model_requires_prefix("Intfloat/Multilingual-E5-Base") is True
