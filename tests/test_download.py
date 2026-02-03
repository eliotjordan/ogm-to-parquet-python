"""Tests for download module."""

from unittest.mock import MagicMock, patch

import pytest

from ogm_to_parquet.download import (
    DEFAULT_REPOS_CONFIG,
    GITHUB_ORG_URL,
    clone_repository,
    download_repositories,
    load_repository_list,
    update_repository,
)


class TestLoadRepositoryList:
    """Test suite for loading repository list from YAML."""

    def test_load_default_config(self):
        """Test loading the default repos.yaml config."""
        repos = load_repository_list()
        assert isinstance(repos, list)
        assert len(repos) > 0
        assert "edu.berkeley" in repos
        assert "edu.stanford.purl" in repos

    def test_load_custom_config(self, tmp_path):
        """Test loading a custom config file."""
        config_file = tmp_path / "custom_repos.yaml"
        config_file.write_text("repositories:\n  - test.repo1\n  - test.repo2\n")

        repos = load_repository_list(config_file)

        assert repos == ["test.repo1", "test.repo2"]

    def test_load_missing_file(self, tmp_path):
        """Test error when config file doesn't exist."""
        missing_file = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError):
            load_repository_list(missing_file)

    def test_load_invalid_config_no_repos_key(self, tmp_path):
        """Test error when config file is missing repositories key."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("something_else:\n  - item1\n")

        with pytest.raises(ValueError, match="missing 'repositories' key"):
            load_repository_list(config_file)

    def test_load_invalid_config_repos_not_list(self, tmp_path):
        """Test error when repositories is not a list."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("repositories: not-a-list\n")

        with pytest.raises(ValueError, match="must be a list"):
            load_repository_list(config_file)

    def test_load_empty_config(self, tmp_path):
        """Test error when config file is empty."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="missing 'repositories' key"):
            load_repository_list(config_file)


class TestCloneRepository:
    """Test suite for cloning repositories."""

    def test_clone_success(self, tmp_path):
        """Test successful clone."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            result = clone_repository("edu.test", tmp_path)

            assert result is True
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0] == [
                "git",
                "clone",
                "--depth",
                "1",
                f"{GITHUB_ORG_URL}/edu.test.git",
                str(tmp_path / "edu.test"),
            ]

    def test_clone_failure(self, tmp_path):
        """Test failed clone."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=128, stdout="", stderr="Repository not found"
            )

            result = clone_repository("nonexistent.repo", tmp_path)

            assert result is False

    def test_clone_exception(self, tmp_path):
        """Test clone with exception."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Network error")

            result = clone_repository("edu.test", tmp_path)

            assert result is False


class TestUpdateRepository:
    """Test suite for updating repositories."""

    def test_update_success_with_changes(self, tmp_path):
        """Test successful update with new changes."""
        repo_path = tmp_path / "edu.test"
        repo_path.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Updating abc123..def456\n", stderr=""
            )

            result = update_repository(repo_path)

            assert result is True
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0] == ["git", "pull"]
            assert call_args[1]["cwd"] == str(repo_path)

    def test_update_already_up_to_date(self, tmp_path):
        """Test update when already up to date."""
        repo_path = tmp_path / "edu.test"
        repo_path.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Already up to date.\n", stderr=""
            )

            result = update_repository(repo_path)

            assert result is True

    def test_update_failure(self, tmp_path):
        """Test failed update."""
        repo_path = tmp_path / "edu.test"
        repo_path.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="Not a git repository"
            )

            result = update_repository(repo_path)

            assert result is False

    def test_update_exception(self, tmp_path):
        """Test update with exception."""
        repo_path = tmp_path / "edu.test"
        repo_path.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Network error")

            result = update_repository(repo_path)

            assert result is False


class TestDownloadRepositories:
    """Test suite for download_repositories function."""

    def test_download_clones_new_repos(self, tmp_path):
        """Test that new repos are cloned."""
        with patch("ogm_to_parquet.download.clone_repository") as mock_clone:
            mock_clone.return_value = True

            results = download_repositories(tmp_path, repos=["edu.test1", "edu.test2"])

            assert results == {"edu.test1": True, "edu.test2": True}
            assert mock_clone.call_count == 2

    def test_download_updates_existing_repos(self, tmp_path):
        """Test that existing repos are updated."""
        # Create existing repo directories with .git folder
        repo1 = tmp_path / "edu.test1"
        repo1.mkdir()
        (repo1 / ".git").mkdir()

        with patch("ogm_to_parquet.download.update_repository") as mock_update:
            mock_update.return_value = True

            results = download_repositories(tmp_path, repos=["edu.test1"])

            assert results == {"edu.test1": True}
            mock_update.assert_called_once_with(repo1)

    def test_download_mixed_new_and_existing(self, tmp_path):
        """Test handling mix of new and existing repos."""
        # Create one existing repo
        existing_repo = tmp_path / "edu.existing"
        existing_repo.mkdir()
        (existing_repo / ".git").mkdir()

        with (
            patch("ogm_to_parquet.download.clone_repository") as mock_clone,
            patch("ogm_to_parquet.download.update_repository") as mock_update,
        ):
            mock_clone.return_value = True
            mock_update.return_value = True

            results = download_repositories(tmp_path, repos=["edu.new", "edu.existing"])

            assert results == {"edu.new": True, "edu.existing": True}
            mock_clone.assert_called_once()
            mock_update.assert_called_once()

    def test_download_creates_target_dir(self, tmp_path):
        """Test that target directory is created if it doesn't exist."""
        new_dir = tmp_path / "new" / "nested" / "dir"

        with patch("ogm_to_parquet.download.clone_repository") as mock_clone:
            mock_clone.return_value = True

            download_repositories(new_dir, repos=["edu.test"])

            assert new_dir.exists()

    def test_download_removes_incomplete_repo(self, tmp_path):
        """Test that incomplete repo (no .git) is removed before clone."""
        # Create incomplete repo directory (no .git folder)
        incomplete_repo = tmp_path / "edu.incomplete"
        incomplete_repo.mkdir()
        (incomplete_repo / "some_file.txt").write_text("partial content")

        with patch("ogm_to_parquet.download.clone_repository") as mock_clone:
            mock_clone.return_value = True

            results = download_repositories(tmp_path, repos=["edu.incomplete"])

            # clone_repository should be called (not update)
            mock_clone.assert_called_once()
            assert results == {"edu.incomplete": True}

    def test_download_loads_from_config(self, tmp_path):
        """Test loading repos from config file."""
        config_file = tmp_path / "repos.yaml"
        config_file.write_text("repositories:\n  - config.repo1\n  - config.repo2\n")

        with patch("ogm_to_parquet.download.clone_repository") as mock_clone:
            mock_clone.return_value = True

            results = download_repositories(tmp_path, config_path=config_file)

            assert "config.repo1" in results
            assert "config.repo2" in results

    def test_download_partial_failure(self, tmp_path):
        """Test handling of partial failures."""
        with patch("ogm_to_parquet.download.clone_repository") as mock_clone:
            mock_clone.side_effect = [True, False, True]

            results = download_repositories(tmp_path, repos=["repo1", "repo2", "repo3"])

            assert results == {"repo1": True, "repo2": False, "repo3": True}


class TestDefaultReposConfig:
    """Test the default repos.yaml configuration."""

    def test_default_config_exists(self):
        """Test that the default config file exists."""
        assert DEFAULT_REPOS_CONFIG.exists()

    def test_default_config_has_expected_repos(self):
        """Test that default config contains expected repositories."""
        repos = load_repository_list()

        expected_repos = [
            "edu.berkeley",
            "edu.harvard",
            "edu.illinois",
            "edu.indiana",
            "edu.msu",
            "edu.northwestern",
            "edu.nyu",
            "edu.osu",
            "edu.psu",
            "edu.purdue",
            "edu.rutgers",
            "edu.stanford.purl",
            "edu.uiowa",
            "edu.umd",
            "edu.umich",
            "edu.umn",
            "edu.unl",
            "edu.utexas",
        ]

        for expected in expected_repos:
            assert expected in repos, f"Missing expected repo: {expected}"
