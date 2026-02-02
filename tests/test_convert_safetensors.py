"""Tests for safetensors conversion utility."""

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from ogm_to_parquet.convert_safetensors import convert_f16_to_f32


class TestConvertSafetensors:
    """Test suite for safetensors conversion utility."""

    def test_convert_f16_to_f32_single_tensor(self):
        """Test conversion of a single F16 tensor to F32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.safetensors"
            output_path = Path(tmpdir) / "output.safetensors"

            # Create F16 tensor
            tensors = {
                "embedding": torch.randn(100, 256, dtype=torch.float16)
            }
            save_file(tensors, str(input_path))

            # Convert
            stats = convert_f16_to_f32(input_path, output_path)

            # Verify stats
            assert stats["converted"] == 1
            assert stats["unchanged"] == 0
            assert stats["total"] == 1

            # Verify output
            output_tensors = load_file(str(output_path))
            assert "embedding" in output_tensors
            assert output_tensors["embedding"].dtype == torch.float32
            assert output_tensors["embedding"].shape == (100, 256)

    def test_convert_f16_to_f32_multiple_tensors(self):
        """Test conversion of multiple tensors with mixed types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.safetensors"
            output_path = Path(tmpdir) / "output.safetensors"

            # Create mixed precision tensors
            tensors = {
                "embeddings": torch.randn(1000, 128, dtype=torch.float16),
                "weights": torch.randn(128, 64, dtype=torch.float16),
                "bias": torch.randn(64, dtype=torch.float32),  # Already F32
                "indices": torch.randint(0, 1000, (100,), dtype=torch.int64),  # Not float
            }
            save_file(tensors, str(input_path))

            # Convert
            stats = convert_f16_to_f32(input_path, output_path)

            # Verify stats
            assert stats["converted"] == 2  # embeddings and weights
            assert stats["unchanged"] == 2  # bias and indices
            assert stats["total"] == 4

            # Verify output types
            output_tensors = load_file(str(output_path))
            assert output_tensors["embeddings"].dtype == torch.float32
            assert output_tensors["weights"].dtype == torch.float32
            assert output_tensors["bias"].dtype == torch.float32
            assert output_tensors["indices"].dtype == torch.int64

            # Verify shapes unchanged
            assert output_tensors["embeddings"].shape == (1000, 128)
            assert output_tensors["weights"].shape == (128, 64)
            assert output_tensors["bias"].shape == (64,)
            assert output_tensors["indices"].shape == (100,)

    def test_convert_f16_to_f32_preserves_values(self):
        """Test that conversion preserves tensor values (within precision limits)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.safetensors"
            output_path = Path(tmpdir) / "output.safetensors"

            # Create F16 tensor with known values
            original = torch.tensor([[1.0, 2.5, -3.75], [0.5, -1.25, 4.0]], dtype=torch.float16)
            tensors = {"data": original}
            save_file(tensors, str(input_path))

            # Convert
            convert_f16_to_f32(input_path, output_path)

            # Load and verify values
            output_tensors = load_file(str(output_path))
            converted = output_tensors["data"]

            # Convert original to F32 for comparison
            expected = original.to(torch.float32)
            assert torch.allclose(converted, expected, rtol=1e-5)

    def test_convert_f16_to_f32_no_f16_tensors(self):
        """Test conversion when all tensors are already F32 or other types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.safetensors"
            output_path = Path(tmpdir) / "output.safetensors"

            # Create only F32 tensors
            tensors = {
                "weights": torch.randn(100, 50, dtype=torch.float32),
                "bias": torch.randn(50, dtype=torch.float32),
            }
            save_file(tensors, str(input_path))

            # Convert
            stats = convert_f16_to_f32(input_path, output_path)

            # Verify stats
            assert stats["converted"] == 0
            assert stats["unchanged"] == 2
            assert stats["total"] == 2

            # Verify output unchanged
            output_tensors = load_file(str(output_path))
            assert output_tensors["weights"].dtype == torch.float32
            assert output_tensors["bias"].dtype == torch.float32

    def test_convert_f16_to_f32_empty_file(self):
        """Test handling of empty safetensors file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.safetensors"
            output_path = Path(tmpdir) / "output.safetensors"

            # Create empty safetensors file
            save_file({}, str(input_path))

            # Convert
            stats = convert_f16_to_f32(input_path, output_path)

            # Verify stats
            assert stats["converted"] == 0
            assert stats["unchanged"] == 0
            assert stats["total"] == 0

    def test_convert_f16_to_f32_input_not_found(self):
        """Test error handling when input file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "nonexistent.safetensors"
            output_path = Path(tmpdir) / "output.safetensors"

            with pytest.raises(FileNotFoundError, match="Input file not found"):
                convert_f16_to_f32(input_path, output_path)

    def test_convert_f16_to_f32_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.safetensors"
            output_path = Path(tmpdir) / "subdir" / "output.safetensors"

            # Create input file
            tensors = {"data": torch.randn(10, 10, dtype=torch.float16)}
            save_file(tensors, str(input_path))

            # Convert (should create subdir)
            convert_f16_to_f32(input_path, output_path)

            # Verify output exists
            assert output_path.exists()
            assert output_path.parent.is_dir()

    def test_convert_f16_to_f32_large_tensors(self):
        """Test conversion with larger tensors (performance check)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.safetensors"
            output_path = Path(tmpdir) / "output.safetensors"

            # Create larger F16 tensor (5000 x 256 = 1.28M elements)
            tensors = {
                "embeddings": torch.randn(5000, 256, dtype=torch.float16)
            }
            save_file(tensors, str(input_path))

            # Convert
            stats = convert_f16_to_f32(input_path, output_path)

            # Verify conversion completed
            assert stats["converted"] == 1
            assert stats["total"] == 1

            # Verify output
            output_tensors = load_file(str(output_path))
            assert output_tensors["embeddings"].dtype == torch.float32
            assert output_tensors["embeddings"].shape == (5000, 256)

    def test_convert_f16_to_f32_invalid_input_file(self):
        """Test error handling with invalid safetensors file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "invalid.safetensors"
            output_path = Path(tmpdir) / "output.safetensors"

            # Create invalid file
            with open(input_path, "w") as f:
                f.write("Not a valid safetensors file")

            with pytest.raises(ValueError, match="Failed to load safetensors file"):
                convert_f16_to_f32(input_path, output_path)
