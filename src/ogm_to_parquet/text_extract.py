"""Text extraction from map images using Ollama vision models.

Downloads images from URLs, processes them, and extracts text using
a local Ollama instance with a vision-language model.
"""

import base64
import json
import logging
import shutil
import sqlite3
import zipfile
from pathlib import Path

import requests
from PIL import Image

# Increase PIL's decompression bomb limit for large map images
# Default is ~179 million pixels; maps can be much larger
Image.MAX_IMAGE_PIXELS = 500_000_000  # 500 million pixels

logger = logging.getLogger(__name__)

# OCR prompt for map text extraction
OCR_PROMPT = """Digitized Map OCR

Your task is to analyze and perform text extraction on this digitized map image.
Focus only on printed text on the map (ignore handwriting unless it clearly functions as a map label).
Preserve original spelling, punctuation, capitalization, accents/diacritics; visible abbreviations; line breaks only when important for meaning (e.g., multi-line titles)
Exclude archival stickers, barcodes, library stamps, scanner metadata, watermarks that are clearly not part of the original cartographic content.
Do not hallucinate text that is not clearly readable. If partially legible, use a ? placeholder only for uncertain characters

Extract these categories into a JSON document with these keys:
1. inset_titles (Inset titles and labels)
2. legend_text (category labels, symbols, explanatory notes)
3. boundaries
4. regions
5. settlements (cities, towns, hamlets, villages)
6. waterbodies
7. roads (streets, paths, highways)
8. railroads
9. buildings (landmarks, buildings, churches, important structures)
10. neighborhoods
11. topographic (mountains, valleys, hills)

Do not include categories without extracted values.
Be as complete as possible.
Pay special attention to roads, try your best to extract as many as possible and do not include road names that are numbers only.

The output should be a valid JSON document in this format:

```
{
  "category1": ["value1", "value2", etc..],
  "category2": ["value3", "value4", etc..],
  …
}
```

Only output the JSON document.
"""

# Valid image extensions
VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".gif",
    ".bmp",
    ".webp",
    ".jp2",
}

# Maximum image size for Ollama (9MB)
MAX_IMAGE_SIZE_BYTES = 9 * 1024 * 1024


class TextExtractor:
    """Extracts text from map images using Ollama vision models."""

    def __init__(
        self,
        db_path: str = "./tmp/text_extraction.db",
        scratch_dir: str = "./tmp/scratch",
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3-vl:30b",
        max_retries: int = 3,
        timeout: int = 300,
    ):
        """Initialize the text extractor.

        Args:
            db_path: Path to SQLite database
            scratch_dir: Directory for temporary file downloads
            ollama_url: Base URL for Ollama API
            model: Ollama model to use for vision tasks
            max_retries: Maximum retries for API calls
            timeout: Request timeout in seconds
        """
        self.db_path = Path(db_path)
        self.scratch_dir = Path(scratch_dir)
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and the model is available.

        Returns:
            True if connection is successful and model is available
        """
        try:
            # Check if Ollama is running by getting the list of models
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            response.raise_for_status()

            models_data = response.json()
            available_models = [m.get("name", "") for m in models_data.get("models", [])]

            logger.info(f"Ollama is running. Available models: {available_models}")

            # Check if our model is available (handle model name with or without tag)
            model_base = self.model.split(":")[0]
            model_available = any(
                m == self.model or m.startswith(f"{model_base}:") for m in available_models
            )

            if not model_available:
                logger.error(
                    f"Model '{self.model}' not found. Available models: {available_models}"
                )
                logger.error(f"Run 'ollama pull {self.model}' to download the model.")
                return False

            return True

        except requests.exceptions.ConnectionError:
            logger.error(
                f"Cannot connect to Ollama at {self.ollama_url}. "
                "Is Ollama running? Start it with 'ollama serve'"
            )
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama connection: {e}")
            return False

    def process_all(self, doc_id: str | None = None) -> dict[str, int]:
        """Process documents in the text_extraction table.

        Args:
            doc_id: Optional document ID to process. If provided, only this document
                   will be processed regardless of its current status. If None,
                   all unprocessed documents will be processed.

        Returns:
            Dictionary with counts of processed, errored, and skipped documents
        """
        # Check Ollama connection first
        if not self.check_ollama_connection():
            logger.error("Aborting: Ollama connection check failed")
            return {"completed": 0, "errors": 0, "skipped": 0}

        # Ensure scratch directory exists
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        stats = {"completed": 0, "errors": 0, "skipped": 0}

        try:
            cursor = conn.cursor()

            if doc_id:
                # Process specific document by ID (regardless of status)
                cursor.execute(
                    "SELECT id, image_url FROM text_extraction WHERE id = ?", (doc_id,)
                )
                documents = cursor.fetchall()
                if not documents:
                    logger.warning(f"Document {doc_id} not found in text_extraction table")
                    return stats
                logger.info(f"Processing single document: {doc_id}")
            else:
                # Get all unprocessed documents
                cursor.execute(
                    "SELECT id, image_url FROM text_extraction WHERE status = 'unprocessed' OR status = 'in_progress'"
                )
                documents = cursor.fetchall()
                logger.info(f"Found {len(documents)} unprocessed documents")

            for doc_id, image_url in documents:
                if not image_url:
                    logger.warning(f"Document {doc_id} has no image_url, skipping")
                    stats["skipped"] += 1
                    continue

                result = self._process_document(conn, doc_id, image_url)
                if result == "completed":
                    stats["completed"] += 1
                elif result == "error":
                    stats["errors"] += 1
                else:
                    stats["skipped"] += 1

        finally:
            conn.close()

        logger.info(f"Processing complete: {stats}")
        return stats

    def _process_document(self, conn: sqlite3.Connection, doc_id: str, image_url: str) -> str:
        """Process a single document.

        Args:
            conn: Database connection
            doc_id: Document ID
            image_url: URL to download image from

        Returns:
            Status string: "completed", "error", or "skipped"
        """
        logger.info(f"Processing document {doc_id}")
        cursor = conn.cursor()
        image_path = None

        try:
            # Mark as in_progress
            cursor.execute(
                "UPDATE text_extraction SET status = 'in_progress' WHERE id = ?",
                (doc_id,),
            )
            conn.commit()

            # Download the file
            image_path = self._download_file(image_url, doc_id)
            if image_path is None:
                self._mark_error(conn, doc_id, f"Failed to download image from {image_url}")
                return "error"

            # Validate and convert image
            processed_path = self._prepare_image(image_path)
            if processed_path is None:
                self._cleanup_file(image_path)
                self._mark_error(conn, doc_id, f"Failed to prepare image: {image_path.name}")
                return "error"

            # If we converted, update the path
            if processed_path != image_path:
                self._cleanup_file(image_path)
                image_path = processed_path

            # Extract text using Ollama
            extracted_json = self._extract_text_with_retry(image_path)
            if extracted_json is None:
                self._cleanup_file(image_path)
                self._mark_error(
                    conn, doc_id, "Failed to extract text after maximum retries"
                )
                return "error"

            # Save result
            cursor.execute(
                "UPDATE text_extraction SET generated_output = ?, status = 'complete' WHERE id = ?",
                (extracted_json, doc_id),
            )
            conn.commit()

            # Cleanup
            self._cleanup_file(image_path)
            logger.info(f"Document {doc_id} completed successfully")
            return "completed"

        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}")
            if image_path:
                self._cleanup_file(image_path)
            self._mark_error(conn, doc_id, str(e))
            return "error"

    def _download_file(self, url: str, doc_id: str) -> Path | None:
        """Download a file from URL to scratch directory.

        Handles zip files by extracting them.

        Args:
            url: URL to download from
            doc_id: Document ID for naming

        Returns:
            Path to downloaded/extracted file, or None on error
        """
        try:
            logger.debug(f"Downloading {url}")
            response = requests.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()

            # Determine file extension from URL or content-type
            url_path = Path(url.split("?")[0])  # Remove query params
            ext = url_path.suffix.lower() or ".tmp"

            # Save to scratch directory
            download_path = self.scratch_dir / f"{doc_id}{ext}"
            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Handle zip files
            if ext == ".zip" or zipfile.is_zipfile(download_path):
                return self._extract_zip(download_path, doc_id)

            return download_path

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None

    def _extract_zip(self, zip_path: Path, doc_id: str) -> Path | None:
        """Extract a zip file and return the first image found.

        Args:
            zip_path: Path to zip file
            doc_id: Document ID

        Returns:
            Path to extracted image, or None if no image found
        """
        try:
            extract_dir = self.scratch_dir / f"{doc_id}_extracted"
            extract_dir.mkdir(exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            # Remove the zip file
            zip_path.unlink()

            # Find image files
            for file_path in extract_dir.rglob("*"):
                if file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    return file_path

            # No image found, clean up
            shutil.rmtree(extract_dir)
            logger.warning(f"No image found in zip for {doc_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to extract zip {zip_path}: {e}")
            return None

    def _prepare_image(self, image_path: Path) -> Path | None:
        """Validate and prepare image for Ollama.

        Converts to JPEG if needed and ensures size is under 9MB.

        Args:
            image_path: Path to image file

        Returns:
            Path to prepared image, or None if invalid
        """
        try:
            # Check if it's a valid image
            if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                logger.warning(f"Invalid image extension: {image_path.suffix}")
                return None

            # Open and validate image
            with Image.open(image_path) as img:
                # Check if already a small enough JPEG
                if (
                    image_path.suffix.lower() in {".jpg", ".jpeg"}
                    and image_path.stat().st_size <= MAX_IMAGE_SIZE_BYTES
                ):
                    return image_path

                # Convert to RGB if necessary (for RGBA, palette images, etc.)
                if img.mode in ("RGBA", "P", "LA"):
                    img = img.convert("RGB")
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Save as JPEG with quality reduction if needed
                output_path = image_path.with_suffix(".jpg")
                quality = 95

                while quality >= 20:
                    img.save(output_path, "JPEG", quality=quality)
                    if output_path.stat().st_size <= MAX_IMAGE_SIZE_BYTES:
                        return output_path
                    quality -= 10

                # If still too large, resize
                return self._resize_image(img, output_path)

        except Exception as e:
            logger.error(f"Failed to prepare image {image_path}: {e}")
            return None

    def _resize_image(self, img: Image.Image, output_path: Path) -> Path | None:
        """Resize image to fit under max size.

        Args:
            img: PIL Image object
            output_path: Path to save resized image

        Returns:
            Path to resized image, or None on error
        """
        try:
            # Start with 80% of original size and reduce
            scale = 0.8
            while scale >= 0.1:
                new_size = (int(img.width * scale), int(img.height * scale))
                resized = img.resize(new_size, Image.Resampling.LANCZOS)
                resized.save(output_path, "JPEG", quality=85)

                if output_path.stat().st_size <= MAX_IMAGE_SIZE_BYTES:
                    return output_path

                scale -= 0.1

            logger.error("Could not resize image to under 9MB")
            return None

        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            return None

    def _extract_text_with_retry(self, image_path: Path) -> str | None:
        """Extract text from image using Ollama with retries.

        Args:
            image_path: Path to prepared image

        Returns:
            JSON string of extracted text, or None on error
        """
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        prompt = OCR_PROMPT

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Ollama API attempt {attempt + 1}/{self.max_retries}")

                # Build request for /api/generate endpoint
                request_data = {
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False,
                }

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=request_data,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                result = response.json()
                content = result.get("response", "")

                # Try to parse as JSON
                json_str = self._extract_json(content)
                if json_str:
                    return json_str

                # Invalid JSON, build follow-up prompt for retry
                logger.warning(f"Invalid JSON response on attempt {attempt + 1}")
                prompt = f"""Your previous response was:
{content}

This is not valid JSON. Please output only a valid JSON document with no additional text or markdown formatting. The format should be:
{{
  "category1": ["value1", "value2"],
  "category2": ["value3", "value4"]
}}

Only output the JSON document."""

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"API error on attempt {attempt + 1}: {e}")

        return None

    def _extract_json(self, content: str) -> str | None:
        """Extract and validate JSON from response content.

        Args:
            content: Raw response content

        Returns:
            Valid JSON string, or None if invalid
        """
        # Try to parse directly
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                json_str = content[start:end].strip()
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    pass

        # Try to extract JSON from generic code blocks
        if "```" in content:
            start = content.find("```") + 3
            # Skip language identifier if present
            newline = content.find("\n", start)
            if newline > start:
                start = newline + 1
            end = content.find("```", start)
            if end > start:
                json_str = content[start:end].strip()
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    pass

        # Try to find JSON object in content
        brace_start = content.find("{")
        brace_end = content.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            json_str = content[brace_start : brace_end + 1]
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass

        return None

    def _mark_error(
        self, conn: sqlite3.Connection, doc_id: str, error_message: str | None = None
    ) -> None:
        """Mark a document as errored in the database.

        Args:
            conn: Database connection
            doc_id: Document ID
            error_message: Optional error message describing the failure
        """
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE text_extraction SET status = 'error', error_message = ? WHERE id = ?",
            (error_message, doc_id),
        )
        conn.commit()
        logger.warning(f"Document {doc_id} marked as error: {error_message}")

    def _cleanup_file(self, file_path: Path) -> None:
        """Clean up a file and its parent directory if it was extracted and empty.

        Args:
            file_path: Path to file to clean up
        """
        try:
            if file_path.exists():
                file_path.unlink()

            # Check if parent is an extraction directory and is now empty
            parent = file_path.parent
            if parent.name.endswith("_extracted") and parent.exists():
                # Only delete if directory is empty (no other files remain)
                if not any(parent.iterdir()):
                    shutil.rmtree(parent)

        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}")


def main():
    """Command-line entry point for text extraction."""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Extract text from map images using Ollama")
    parser.add_argument(
        "--db-path",
        type=str,
        default="./tmp/text_extraction.db",
        help="Path to SQLite database (default: ./tmp/text_extraction.db)",
    )
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default="./tmp/scratch",
        help="Directory for temporary files (default: ./tmp/scratch)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-vl:32b",
        help="Ollama model to use (default: qwen2.5-vl:32b)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for API calls (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Process only this document ID (processes regardless of current status)",
    )

    args = parser.parse_args()

    extractor = TextExtractor(
        db_path=args.db_path,
        scratch_dir=args.scratch_dir,
        ollama_url=args.ollama_url,
        model=args.model,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )

    try:
        results = extractor.process_all(doc_id=args.id)
        print(f"Extraction complete: {results}")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(130)  # Standard exit code for Ctrl+C


if __name__ == "__main__":
    main()
