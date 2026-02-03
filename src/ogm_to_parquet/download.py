"""Repository download and update functionality.

Downloads OpenGeoMetadata repositories from GitHub using shallow clones
for efficient storage and bandwidth usage.
"""

import logging
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Default location of the repos.yaml config file (same directory as this module)
DEFAULT_REPOS_CONFIG = Path(__file__).parent / "repos.yaml"
GITHUB_ORG_URL = "https://github.com/OpenGeoMetadata"


def load_repository_list(config_path: Path | str | None = None) -> list[str]:
    """Load the list of repositories from a YAML config file.

    Args:
        config_path: Path to YAML config file. If None, uses default repos.yaml.

    Returns:
        List of repository names (e.g., ["edu.berkeley", "edu.harvard"])

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is malformed
    """
    config_path = Path(config_path) if config_path else DEFAULT_REPOS_CONFIG

    if not config_path.exists():
        raise FileNotFoundError(f"Repository config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config or "repositories" not in config:
        raise ValueError(f"Invalid config file: missing 'repositories' key in {config_path}")

    repos = config["repositories"]
    if not isinstance(repos, list):
        raise ValueError(f"Invalid config file: 'repositories' must be a list in {config_path}")

    return repos


def clone_repository(repo_name: str, target_dir: Path) -> bool:
    """Clone a repository using shallow clone.

    Args:
        repo_name: Repository name (e.g., "edu.berkeley")
        target_dir: Directory to clone into

    Returns:
        True if successful, False otherwise
    """
    repo_url = f"{GITHUB_ORG_URL}/{repo_name}.git"
    repo_path = target_dir / repo_name

    logger.info(f"Cloning {repo_name} from {repo_url}")

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"Failed to clone {repo_name}: {result.stderr}")
            return False

        logger.info(f"Successfully cloned {repo_name}")
        return True

    except Exception as e:
        logger.error(f"Error cloning {repo_name}: {e}")
        return False


def update_repository(repo_path: Path) -> bool:
    """Update an existing repository with git pull.

    Args:
        repo_path: Path to the repository

    Returns:
        True if successful, False otherwise
    """
    repo_name = repo_path.name
    logger.info(f"Updating {repo_name}")

    try:
        result = subprocess.run(
            ["git", "pull"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"Failed to update {repo_name}: {result.stderr}")
            return False

        # Log whether there were changes
        if "Already up to date" in result.stdout:
            logger.info(f"{repo_name} is already up to date")
        else:
            logger.info(f"Successfully updated {repo_name}")

        return True

    except Exception as e:
        logger.error(f"Error updating {repo_name}: {e}")
        return False


def download_repositories(
    target_dir: Path | str,
    config_path: Path | str | None = None,
    repos: list[str] | None = None,
) -> dict[str, bool]:
    """Download or update OpenGeoMetadata repositories.

    For each repository:
    - If it doesn't exist locally, clone it using shallow clone (--depth 1)
    - If it exists, run git pull to get latest changes

    Args:
        target_dir: Directory where repositories will be stored
        config_path: Path to YAML config file with repository list.
                    If None, uses default repos.yaml.
        repos: Optional list of repository names to override config file.
               If provided, config_path is ignored.

    Returns:
        Dictionary mapping repository name to success status (True/False)
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Get repository list
    if repos is None:
        repos = load_repository_list(config_path)

    logger.info(f"Processing {len(repos)} repositories")

    results = {}
    for repo_name in repos:
        repo_path = target_dir / repo_name

        if repo_path.exists() and (repo_path / ".git").is_dir():
            # Repository exists, update it
            results[repo_name] = update_repository(repo_path)
        else:
            # Repository doesn't exist, clone it
            # Remove partial directory if it exists but isn't a git repo
            if repo_path.exists():
                logger.warning(f"Removing incomplete directory: {repo_path}")
                import shutil

                shutil.rmtree(repo_path)
            results[repo_name] = clone_repository(repo_name, target_dir)

    # Log summary
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    logger.info(f"Repository download complete: {successful} successful, {failed} failed")

    return results
