"""Tests for batch embedding operations in retrievers."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np


class TestBatchEmbedChunks:
    """Test batch_embed_chunks helper function."""

    def test_batch_embeddings_returns_correct_count(self):
        """Test that batch embedding returns correct number of embeddings."""
        from minions.utils.multimodal_retrievers import (
            batch_embed_chunks,
            MultiModalEmbedder,
        )

        # Mock embedder
        embedder = Mock(spec=MultiModalEmbedder)
        embedder.client = Mock()
        embedder.client.embed = Mock(
            return_value=[
                [0.1] * 768,  # Embedding 1
                [0.2] * 768,  # Embedding 2
                [0.3] * 768,  # Embedding 3
            ]
        )

        chunks = ["chunk 1", "chunk 2", "chunk 3"]
        embeddings = batch_embed_chunks(embedder, chunks, batch_size=32)

        assert len(embeddings) == 3
        assert all(len(emb.embedding) == 768 for emb in embeddings)
        # Verify client.embed was called with the full batch
        embedder.client.embed.assert_called_once_with(content=chunks)

    def test_batch_embeddings_with_small_batch_size(self):
        """Test batching with batch_size smaller than chunk count."""
        from minions.utils.multimodal_retrievers import (
            batch_embed_chunks,
            MultiModalEmbedder,
        )

        embedder = Mock(spec=MultiModalEmbedder)
        call_count = 0
        batches_received = []

        def mock_embed(content):
            nonlocal call_count
            call_count += 1
            batches_received.append(len(content))
            # Return embeddings for batch
            return [[0.1] * 768 for _ in range(len(content))]

        embedder.client = Mock()
        embedder.client.embed = mock_embed

        chunks = ["chunk" + str(i) for i in range(10)]
        embeddings = batch_embed_chunks(embedder, chunks, batch_size=3)

        # Should make 4 API calls: [3, 3, 3, 1]
        assert call_count == 4
        assert batches_received == [3, 3, 3, 1]
        assert len(embeddings) == 10

    def test_batch_embeddings_preserves_order(self):
        """Test that batching preserves chunk order."""
        from minions.utils.multimodal_retrievers import (
            batch_embed_chunks,
            MultiModalEmbedder,
        )

        embedder = Mock(spec=MultiModalEmbedder)

        def mock_embed(content):
            # Return embeddings with index-based values to verify order
            return [[float(i)] * 768 for i in range(len(content))]

        embedder.client = Mock()
        embedder.client.embed = mock_embed

        chunks = ["chunk" + str(i) for i in range(50)]
        embeddings = batch_embed_chunks(embedder, chunks, batch_size=10)

        # Verify order preserved
        assert len(embeddings) == 50
        for i, emb in enumerate(embeddings):
            assert emb.content == f"chunk{i}"

    def test_batch_embeddings_with_empty_chunks(self):
        """Test handling of empty chunk list."""
        from minions.utils.multimodal_retrievers import (
            batch_embed_chunks,
            MultiModalEmbedder,
        )

        embedder = Mock(spec=MultiModalEmbedder)
        embedder.client = Mock()
        chunks = []
        embeddings = batch_embed_chunks(embedder, chunks, batch_size=32)

        assert len(embeddings) == 0
        embedder.client.embed.assert_not_called()

    def test_batch_embeddings_with_single_chunk(self):
        """Test handling of single chunk."""
        from minions.utils.multimodal_retrievers import (
            batch_embed_chunks,
            MultiModalEmbedder,
        )

        embedder = Mock(spec=MultiModalEmbedder)
        embedder.client = Mock()
        embedder.client.embed = Mock(return_value=[[0.1] * 768])

        chunks = ["single chunk"]
        embeddings = batch_embed_chunks(embedder, chunks, batch_size=32)

        assert len(embeddings) == 1
        assert embeddings[0].content == "single chunk"
        embedder.client.embed.assert_called_once_with(content=["single chunk"])

    def test_batch_embeddings_with_file_paths(self):
        """Test batch embedding preserves file paths."""
        from minions.utils.multimodal_retrievers import (
            batch_embed_chunks,
            MultiModalEmbedder,
        )

        embedder = Mock(spec=MultiModalEmbedder)
        embedder.client = Mock()
        embedder.client.embed = Mock(return_value=[[0.1] * 768, [0.2] * 768])

        chunks = ["chunk 1", "chunk 2"]
        embeddings = batch_embed_chunks(
            embedder, chunks, batch_size=32, file_path="/test/path.txt"
        )

        assert len(embeddings) == 2
        assert all(emb.content_path == "/test/path.txt" for emb in embeddings)

    def test_batch_size_boundary_conditions(self):
        """Test edge cases with batch size boundaries."""
        from minions.utils.multimodal_retrievers import (
            batch_embed_chunks,
            MultiModalEmbedder,
        )

        embedder = Mock(spec=MultiModalEmbedder)
        embedder.client = Mock()

        def mock_embed(content):
            return [[0.1] * 768 for _ in range(len(content))]

        embedder.client.embed = mock_embed

        # Test: chunks == batch_size (exactly one batch)
        chunks = ["chunk"] * 32
        embeddings = batch_embed_chunks(embedder, chunks, batch_size=32)
        assert len(embeddings) == 32

        # Test: chunks > batch_size by 1 (two batches: 32 + 1)
        chunks = ["chunk"] * 33
        embeddings = batch_embed_chunks(embedder, chunks, batch_size=32)
        assert len(embeddings) == 33

        # Test: chunks < batch_size (one partial batch)
        chunks = ["chunk"] * 5
        embeddings = batch_embed_chunks(embedder, chunks, batch_size=32)
        assert len(embeddings) == 5


class TestRetrieveChunksChroma:
    """Test retrieve_chunks_from_chroma with batch operations."""

    @pytest.mark.skip(reason="Requires ChromaDB server - integration test")
    def test_chroma_retrieval_uses_batch_embedding(self):
        """Test that chroma retrieval uses batch embedding."""
        from minions.utils.multimodal_retrievers import retrieve_chunks_from_chroma

        chunks = [f"Test chunk {i} with content" for i in range(20)]
        keywords = ["test", "content"]

        # This should use batching internally
        results = retrieve_chunks_from_chroma(
            chunks, keywords, embedding_model="llama3.2", k=5, batch_size=10
        )

        assert len(results) <= 5
        assert all(isinstance(r, str) for r in results)

    @pytest.mark.skip(reason="Performance test - run manually")
    def test_chroma_retrieval_performance(self):
        """Benchmark: batch should be faster than sequential."""
        import time
        from minions.utils.multimodal_retrievers import retrieve_chunks_from_chroma

        chunks = [f"Test chunk {i}" for i in range(100)]
        keywords = ["test"]

        # Sequential (batch_size=1)
        start = time.time()
        results_seq = retrieve_chunks_from_chroma(
            chunks, keywords, embedding_model="llama3.2", k=10, batch_size=1
        )
        time_seq = time.time() - start

        # Batched (batch_size=32)
        start = time.time()
        results_batch = retrieve_chunks_from_chroma(
            chunks, keywords, embedding_model="llama3.2", k=10, batch_size=32
        )
        time_batch = time.time() - start

        speedup = time_seq / time_batch
        print(f"Sequential: {time_seq:.2f}s")
        print(f"Batched: {time_batch:.2f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Expect at least 2x speedup
        assert speedup >= 2.0


class TestRetrieveChunksQdrant:
    """Test retrieve_chunks_from_qdrant with batch operations."""

    @pytest.mark.skip(reason="Requires Qdrant server - integration test")
    def test_qdrant_retrieval_uses_batch_embedding(self):
        """Test that qdrant retrieval uses batch embedding."""
        from minions.utils.multimodal_retrievers import retrieve_chunks_from_qdrant

        chunks = [f"Test chunk {i} with content" for i in range(20)]
        keywords = ["test", "content"]

        results = retrieve_chunks_from_qdrant(
            chunks,
            keywords,
            embedding_model="llama3.2",
            k=5,
            batch_size=10,
            qdrant_url="http://localhost:6333",
        )

        assert len(results) <= 5
        assert all(isinstance(r, str) for r in results)

    @pytest.mark.skip(reason="Performance test - run manually")
    def test_qdrant_retrieval_performance(self):
        """Benchmark: batch should be faster than sequential."""
        import time
        from minions.utils.multimodal_retrievers import retrieve_chunks_from_qdrant

        chunks = [f"Test chunk {i}" for i in range(100)]
        keywords = ["test"]

        # Sequential
        start = time.time()
        results_seq = retrieve_chunks_from_qdrant(
            chunks, keywords, embedding_model="llama3.2", k=10, batch_size=1
        )
        time_seq = time.time() - start

        # Batched
        start = time.time()
        results_batch = retrieve_chunks_from_qdrant(
            chunks, keywords, embedding_model="llama3.2", k=10, batch_size=32
        )
        time_batch = time.time() - start

        speedup = time_seq / time_batch
        print(f"Sequential: {time_seq:.2f}s")
        print(f"Batched: {time_batch:.2f}s")
        print(f"Speedup: {speedup:.2f}x")

        assert speedup >= 2.0


class TestBatchEmbeddingEdgeCases:
    """Test edge cases and error handling."""

    def test_batch_embed_with_very_large_batch_size(self):
        """Test that large batch_size doesn't cause issues."""
        from minions.utils.multimodal_retrievers import (
            batch_embed_chunks,
            MultiModalEmbedder,
        )

        embedder = Mock(spec=MultiModalEmbedder)
        embedder.client = Mock()
        embedder.client.embed = Mock(
            return_value=[[0.1] * 768 for _ in range(10)]
        )

        chunks = ["chunk"] * 10
        # batch_size larger than total chunks
        embeddings = batch_embed_chunks(embedder, chunks, batch_size=1000)

        assert len(embeddings) == 10
        # Should only make 1 call
        embedder.client.embed.assert_called_once()

    def test_batch_embed_maintains_text_content(self):
        """Test that text content is correctly assigned to embeddings."""
        from minions.utils.multimodal_retrievers import (
            batch_embed_chunks,
            MultiModalEmbedder,
        )

        embedder = Mock(spec=MultiModalEmbedder)
        embedder.client = Mock()

        def mock_embed(content):
            return [[float(ord(c[0]))] * 768 for c in content]

        embedder.client.embed = mock_embed

        chunks = ["apple", "banana", "cherry"]
        embeddings = batch_embed_chunks(embedder, chunks, batch_size=2)

        assert embeddings[0].content == "apple"
        assert embeddings[1].content == "banana"
        assert embeddings[2].content == "cherry"
