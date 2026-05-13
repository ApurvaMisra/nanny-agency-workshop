from unittest.mock import MagicMock, patch
from nanny_workshop.openai_client import CachedOpenAI


@patch("nanny_workshop.openai_client.OpenAI")
def test_completion_caches_on_disk(mock_openai_cls, tmp_path):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="hi"))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    client = CachedOpenAI(cache_dir=tmp_path, api_key="test")
    out1 = client.complete(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    out2 = client.complete(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    assert out1 == "hi"
    assert out2 == "hi"
    assert mock_client.chat.completions.create.call_count == 1  # second served from cache


@patch("nanny_workshop.openai_client.OpenAI")
def test_embeddings_returns_vector(mock_openai_cls, tmp_path):
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    client = CachedOpenAI(cache_dir=tmp_path, api_key="test")
    vec = client.embed(model="text-embedding-3-small", text="hello")
    assert vec == [0.1, 0.2, 0.3]
