"""Convert safetensors models between F16 and F32 precision formats.

This utility converts all tensors in a safetensors file from float16 to float32
precision. Useful for ensuring compatibility with systems that don't support
half-precision floats (e.g., some JavaScript/WASM environments).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


def convert_f16_to_f32(input_path: Path, output_path: Path) -> Dict[str, int]:
    """Convert all F16 tensors in a safetensors file to F32.

    Args:
        input_path: Path to input safetensors file (may contain F16 tensors)
        output_path: Path to output safetensors file (all tensors as F32)

    Returns:
        Dictionary with conversion statistics:
        - converted: Number of tensors converted from F16 to F32
        - unchanged: Number of tensors that were already F32 or other types
        - total: Total number of tensors processed

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input file is not a valid safetensors file
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading model from {input_path}")

    try:
        tensors = load_file(str(input_path))
    except Exception as e:
        raise ValueError(f"Failed to load safetensors file: {e}") from e

    if not tensors:
        logger.warning("No tensors found in input file")
        return {"converted": 0, "unchanged": 0, "total": 0}

    converted_tensors = {}
    stats = {"converted": 0, "unchanged": 0, "total": len(tensors)}

    for name, tensor in tensors.items():
        logger.debug(f"  {name}: {tensor.dtype} {list(tensor.shape)}")

        # Convert F16 to F32
        if tensor.dtype == torch.float16:
            converted_tensor = tensor.to(torch.float32)
            logger.debug(f"    -> Converted to {converted_tensor.dtype}")
            stats["converted"] += 1
        else:
            converted_tensor = tensor
            logger.debug(f"    -> Keeping as {tensor.dtype}")
            stats["unchanged"] += 1

        converted_tensors[name] = converted_tensor

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving converted model to {output_path}")
    save_file(converted_tensors, str(output_path))

    logger.info(
        f"Conversion complete: {stats['converted']} converted, "
        f"{stats['unchanged']} unchanged, {stats['total']} total"
    )

    return stats


def main() -> int:
    """Command-line interface for safetensors conversion.

    Returns:
        Exit code: 0 for success, 1 for error
    """
    parser = argparse.ArgumentParser(
        description="Convert safetensors models from F16 to F32 precision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  %(prog)s model.safetensors model_f32.safetensors

  # Convert with custom output name
  %(prog)s tmp/ogm-model/model.safetensors tmp/ogm-model/model_f32.safetensors

  # Enable verbose logging
  %(prog)s -v model.safetensors model_f32.safetensors
        """,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to input safetensors file (F16 or mixed precision)",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Path to output safetensors file (all tensors will be F32)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (shows per-tensor conversion details)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Check if output file exists
    if args.output.exists() and not args.overwrite:
        logger.error(
            f"Output file already exists: {args.output}\n"
            "Use --overwrite to replace it"
        )
        return 1

    try:
        stats = convert_f16_to_f32(args.input, args.output)

        # Print summary
        print("\nConversion Summary:")
        print(f"  Input:  {args.input}")
        print(f"  Output: {args.output}")
        print(f"  Total tensors:     {stats['total']}")
        print(f"  Converted (F16→F32): {stats['converted']}")
        print(f"  Unchanged:         {stats['unchanged']}")

        if stats["converted"] == 0:
            print("\nNote: No F16 tensors found. All tensors were already F32 or other types.")

        return 0

    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
