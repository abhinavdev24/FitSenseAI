"""Push LoRA adapters and training checkpoints to Google Cloud Storage and HuggingFace Hub.

This script packages a selected LoRA adapter with metadata and pushes it to
Google Cloud Storage (GCS) and optionally to HuggingFace Hub. Also supports
uploading all training checkpoints to GCS. Supports versioning, tagging, and rollback.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_registry_config(config_path: str) -> dict[str, Any]:
    """Load registry push configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values, or empty dict if path is None.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Registry config file not found: {config_path}")
    with path.open("r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    return cfg


def get_git_commit() -> str:
    """Get the short git commit hash.

    Returns:
        Short commit hash (7 chars) or "unknown" if not in a git repo

    Raises:
        subprocess.CalledProcessError: If git command fails unexpectedly
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def validate_inputs(
    adapter_dir: str, metadata_files: list[str] | None, logger: logging.Logger
) -> list[Path]:
    """Validate that all required inputs exist.

    Args:
        adapter_dir: Path to the LoRA adapter directory
        metadata_files: Optional list of metadata file paths
        logger: Logger instance

    Returns:
        List of all files to package (adapter files + metadata files)

    Raises:
        FileNotFoundError: If any required file is missing
    """
    adapter_path = Path(adapter_dir)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    if not adapter_path.is_dir():
        raise ValueError(f"Adapter path is not a directory: {adapter_path}")

    logger.info(f"Validating adapter directory: {adapter_path}")

    # Collect adapter files
    adapter_files = collect_adapter_files(adapter_path)
    if not adapter_files:
        logger.warning(f"No files found in adapter directory: {adapter_path}")

    logger.info(f"Found {len(adapter_files)} adapter files")

    # Validate metadata files
    files_to_package = list(adapter_files)
    if metadata_files:
        for meta_file in metadata_files:
            meta_path = Path(meta_file)
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            files_to_package.append(meta_path)
            logger.info(f"Including metadata file: {meta_path}")

    return files_to_package


def collect_adapter_files(adapter_dir: Path) -> list[Path]:
    """Collect all files from the adapter directory.

    Args:
        adapter_dir: Path to the adapter directory

    Returns:
        List of file paths in the adapter directory
    """
    return [p for p in adapter_dir.rglob("*") if p.is_file()]


def stage_package(
    adapter_dir: str,
    metadata_files: list[str] | None,
    staging_dir: str,
    logger: logging.Logger,
) -> Path:
    """Stage the adapter and metadata files into a staging directory.

    Args:
        adapter_dir: Path to the LoRA adapter directory
        metadata_files: Optional list of metadata file paths
        staging_dir: Base staging directory
        logger: Logger instance

    Returns:
        Path to the staged package directory

    Raises:
        OSError: If file copy fails
    """
    adapter_path = Path(adapter_dir)
    staging_path = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Staging package to {staging_path}")

    # Copy adapter files into final_adapter/ subdirectory
    for src_file in collect_adapter_files(adapter_path):
        rel_path = src_file.relative_to(adapter_path)
        dest_file = staging_path / "final_adapter" / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(src_file, dest_file)
            logger.debug(f"Copied {src_file} -> {dest_file}")
        except OSError as e:
            logger.error(f"Failed to copy {src_file}: {e}")
            raise

    # Copy metadata files
    if metadata_files:
        for meta_file in metadata_files:
            meta_path = Path(meta_file)
            dest_file = staging_path / meta_path.name

            try:
                shutil.copy2(meta_path, dest_file)
                logger.info(f"Copied metadata: {meta_path} -> {dest_file}")
            except OSError as e:
                logger.error(f"Failed to copy metadata {meta_path}: {e}")
                raise

    logger.info(f"Package staged successfully at {staging_path}")
    return staging_path


def write_manifest(
    staging_dir: Path,
    model_name: str,
    version: str,
    metadata_files: list[str] | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Write a manifest.json file to the staging directory.

    Args:
        staging_dir: Path to the staging directory
        model_name: Model identifier
        version: Version tag
        metadata_files: Optional list of metadata file names
        logger: Logger instance

    Returns:
        Manifest dict

    Raises:
        ValueError: If version format is invalid
    """
    if not version.startswith("v"):
        raise ValueError(f"Version must start with 'v': {version}")

    # List all files in staging directory
    files_in_staging = [p.name for p in staging_dir.rglob("*") if p.is_file()]

    # Build metadata files list (excluding adapter files)
    metadata_file_names = []
    if metadata_files:
        metadata_file_names = [Path(f).name for f in metadata_files]

    manifest = {
        "model_name": model_name,
        "version": version,
        "git_commit": get_git_commit(),
        "files": sorted(files_in_staging),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata_files": sorted(metadata_file_names),
    }

    manifest_path = staging_dir / "manifest.json"
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest written to {manifest_path}")
    except OSError as e:
        logger.error(f"Failed to write manifest: {e}")
        raise

    return manifest


def upload_to_gcs(
    staging_dir: Path,
    bucket_name: str,
    model_name: str,
    version: str,
    logger: logging.Logger,
) -> None:
    """Upload all staged files to Google Cloud Storage.

    Args:
        staging_dir: Path to the staging directory
        bucket_name: GCS bucket name (e.g., "fitsense-models")
        model_name: Model identifier
        version: Version tag
        logger: Logger instance

    Raises:
        ImportError: If google-cloud-storage is not installed
        Exception: If upload fails
    """
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required for upload. "
            "Install with: pip install google-cloud-storage"
        ) from exc

    logger.info(f"Uploading to gs://{bucket_name}/{model_name}/{version}/")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Upload all files from staging directory
        uploaded = 0
        skipped = 0
        for local_path in staging_dir.rglob("*"):
            if not local_path.is_file():
                continue

            rel_path = local_path.relative_to(staging_dir)
            blob_path = f"{model_name}/{version}/{rel_path}"
            blob = bucket.blob(blob_path)

            if blob.exists():
                logger.debug(f"Skipping {blob_path} (already exists)")
                skipped += 1
                continue

            try:
                blob.upload_from_filename(str(local_path))
                logger.debug(f"Uploaded {blob_path}")
                uploaded += 1
            except Exception as e:
                logger.error(f"Failed to upload {blob_path}: {e}")
                raise

        logger.info(
            f"Upload complete: {uploaded} uploaded, {skipped} skipped "
            f"(already exist) to {bucket_name}/{model_name}/{version}/"
        )

    except Exception as e:
        logger.error(f"Upload to GCS failed: {e}")
        raise


def update_latest_pointer(
    bucket_name: str,
    model_name: str,
    version: str,
    logger: logging.Logger,
) -> None:
    """Update the latest.json pointer on GCS.

    Args:
        bucket_name: GCS bucket name
        model_name: Model identifier
        version: Version tag
        logger: Logger instance

    Raises:
        Exception: If update fails
    """
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required. "
            "Install with: pip install google-cloud-storage"
        ) from exc

    logger.info(f"Updating latest.json for {model_name}")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        latest_blob_path = f"{model_name}/latest.json"
        latest_data = {
            "version": version,
            "path": f"gs://{bucket_name}/{model_name}/{version}/",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        blob = bucket.blob(latest_blob_path)
        blob.upload_from_string(json.dumps(latest_data, indent=2))

        logger.info(f"Updated {latest_blob_path}")

    except Exception as e:
        logger.error(f"Failed to update latest.json: {e}")
        raise


def update_versions_list(
    bucket_name: str,
    model_name: str,
    version: str,
    logger: logging.Logger,
) -> None:
    """Update or create the versions.json file on GCS.

    Args:
        bucket_name: GCS bucket name
        model_name: Model identifier
        version: Version tag
        logger: Logger instance

    Raises:
        Exception: If update fails
    """
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required. "
            "Install with: pip install google-cloud-storage"
        ) from exc

    logger.info(f"Updating versions.json for {model_name}")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        versions_blob_path = f"{model_name}/versions.json"
        versions_blob = bucket.blob(versions_blob_path)

        # Try to load existing versions
        versions_data = []
        if versions_blob.exists():
            try:
                existing = json.loads(versions_blob.download_as_string())
                versions_data = existing if isinstance(existing, list) else []
                logger.info(f"Loaded {len(versions_data)} existing versions")
            except json.JSONDecodeError:
                logger.warning(f"Could not parse existing {versions_blob_path}, starting fresh")
                versions_data = []

        # Add new version
        new_entry = {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": f"gs://{bucket_name}/{model_name}/{version}/",
        }

        # Check if version already exists
        if not any(v.get("version") == version for v in versions_data):
            versions_data.append(new_entry)
            versions_data.sort(key=lambda x: x["timestamp"], reverse=True)

            # Upload updated versions list
            versions_blob.upload_from_string(json.dumps(versions_data, indent=2))
            logger.info(f"Added version {version} to {versions_blob_path}")
        else:
            logger.info(f"Version {version} already exists in {versions_blob_path}")

    except Exception as e:
        logger.error(f"Failed to update versions.json: {e}")
        raise


def discover_checkpoints(checkpoints_dir: str, logger: logging.Logger) -> list[Path]:
    """Discover all checkpoint subdirectories (e.g. checkpoint-100, checkpoint-200).

    Args:
        checkpoints_dir: Path to directory containing checkpoint-* subdirs
        logger: Logger instance

    Returns:
        Sorted list of checkpoint directory paths
    """
    base = Path(checkpoints_dir)
    if not base.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {base}")

    checkpoints = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    logger.info(f"Discovered {len(checkpoints)} checkpoints in {base}: {[p.name for p in checkpoints]}")
    return checkpoints


def upload_checkpoints_to_gcs(
    checkpoints_dir: str,
    bucket_name: str,
    model_name: str,
    version: str,
    logger: logging.Logger,
) -> None:
    """Upload all checkpoint subdirectories to GCS under {model_name}/{version}/checkpoints/.

    Args:
        checkpoints_dir: Local path containing checkpoint-* subdirs
        bucket_name: GCS bucket name (e.g., fitsense-model)
        model_name: Model identifier
        version: Version tag
        logger: Logger instance

    Raises:
        ImportError: If google-cloud-storage is not installed
        FileNotFoundError: If checkpoints directory is missing
        Exception: If any upload fails
    """
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required for upload. "
            "Install with: pip install google-cloud-storage"
        ) from exc

    checkpoints = discover_checkpoints(checkpoints_dir, logger)
    if not checkpoints:
        logger.warning(f"No checkpoint-* directories found in {checkpoints_dir}; skipping checkpoint upload")
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for ckpt_dir in checkpoints:
        ckpt_name = ckpt_dir.name
        gcs_prefix = f"{model_name}/{version}/checkpoints/{ckpt_name}"
        logger.info(f"Uploading {ckpt_name} -> gs://{bucket_name}/{gcs_prefix}/")

        uploaded = 0
        skipped = 0
        for local_path in ckpt_dir.rglob("*"):
            if not local_path.is_file():
                continue
            rel = local_path.relative_to(ckpt_dir)
            blob_path = f"{gcs_prefix}/{rel}"
            blob = bucket.blob(blob_path)

            if blob.exists():
                logger.debug(f"Skipping {blob_path} (already exists)")
                skipped += 1
                continue

            try:
                blob.upload_from_filename(str(local_path))
                logger.debug(f"Uploaded {blob_path}")
                uploaded += 1
            except Exception as e:
                logger.error(f"Failed to upload checkpoint file {blob_path}: {e}")
                raise

        logger.info(f"Checkpoint {ckpt_name}: {uploaded} uploaded, {skipped} skipped (already exist)")

    logger.info(
        f"All {len(checkpoints)} checkpoints uploaded to "
        f"gs://{bucket_name}/{model_name}/{version}/checkpoints/"
    )


def upload_hparam_search_to_gcs(
    hparam_search_dir: str,
    bucket_name: str,
    model_name: str,
    version: str,
    logger: logging.Logger,
) -> None:
    """Upload the hparam_search directory to GCS under {model_name}/{version}/hparam_search/.

    Args:
        hparam_search_dir: Local path containing hparam search results
        bucket_name: GCS bucket name
        model_name: Model identifier
        version: Version tag
        logger: Logger instance

    Raises:
        ImportError: If google-cloud-storage is not installed
        FileNotFoundError: If hparam_search directory is missing
        Exception: If any upload fails
    """
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required for upload. "
            "Install with: pip install google-cloud-storage"
        ) from exc

    base = Path(hparam_search_dir)
    if not base.exists():
        raise FileNotFoundError(f"Hparam search directory not found: {base}")

    gcs_prefix = f"{model_name}/{version}/hparam_search"
    logger.info(f"Uploading hparam_search -> gs://{bucket_name}/{gcs_prefix}/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    files = [p for p in base.rglob("*") if p.is_file()]
    uploaded = 0
    skipped = 0
    for local_path in files:
        rel = local_path.relative_to(base)
        blob_path = f"{gcs_prefix}/{rel}"
        blob = bucket.blob(blob_path)

        if blob.exists():
            logger.debug(f"Skipping {blob_path} (already exists)")
            skipped += 1
            continue

        try:
            blob.upload_from_filename(str(local_path))
            logger.debug(f"Uploaded {blob_path}")
            uploaded += 1
        except Exception as e:
            logger.error(f"Failed to upload hparam search file {blob_path}: {e}")
            raise

    logger.info(
        f"Hparam search: {uploaded} uploaded, {skipped} skipped (already exist) "
        f"to gs://{bucket_name}/{gcs_prefix}/"
    )


def push_to_huggingface(
    adapter_dir: str,
    repo_id: str,
    version: str,
    token: str | None,
    logger: logging.Logger,
    commit_message: str | None = None,
) -> str:
    """Push a LoRA adapter directory to HuggingFace Hub.

    Args:
        adapter_dir: Local path to the LoRA adapter directory
        repo_id: HuggingFace repo in the form 'username/repo-name'
        version: Version tag used as the git tag on the Hub
        token: HuggingFace API token (falls back to HF_TOKEN env var)
        logger: Logger instance
        commit_message: Optional commit message; defaults to 'Upload {version}'

    Returns:
        URL of the model on HuggingFace Hub

    Raises:
        ImportError: If huggingface_hub is not installed
        ValueError: If repo_id is malformed or token is missing
        Exception: If upload fails
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for HF push. "
            "Install with: pip install huggingface_hub"
        ) from exc

    resolved_token = token or os.environ.get("HF_TOKEN")
    if not resolved_token:
        raise ValueError(
            "HuggingFace token is required. Pass --hf-token or set the HF_TOKEN environment variable."
        )

    if "/" not in repo_id:
        raise ValueError(f"repo_id must be 'username/repo-name', got: {repo_id}")

    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    logger.info(f"Pushing adapter to HuggingFace Hub: {repo_id}")

    api = HfApi(token=resolved_token)

    # Create the repo if it doesn't exist
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=resolved_token)
        logger.info(f"Repository ensured: https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.error(f"Failed to create/verify HF repo {repo_id}: {e}")
        raise

    message = commit_message or f"Upload adapter {version}"

    # Upload the entire adapter directory
    try:
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=message,
        )
        logger.info(f"Adapter uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.error(f"Failed to upload adapter to HF Hub: {e}")
        raise

    # Create a version tag on the Hub
    try:
        api.create_tag(repo_id=repo_id, tag=version, repo_type="model", token=resolved_token)
        logger.info(f"Created HuggingFace tag: {version}")
    except Exception as e:
        # Tags may already exist; log but don't fail
        logger.warning(f"Could not create HF tag {version}: {e}")

    repo_url = f"https://huggingface.co/{repo_id}"
    logger.info(f"HuggingFace push complete: {repo_url}")
    return repo_url


def rollback(
    bucket_name: str,
    model_name: str,
    target_version: str,
    logger: logging.Logger,
) -> None:
    """Rollback latest.json to point to a previous version.

    Args:
        bucket_name: GCS bucket name
        model_name: Model identifier
        target_version: Version to rollback to
        logger: Logger instance

    Raises:
        Exception: If rollback fails
    """
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required. "
            "Install with: pip install google-cloud-storage"
        ) from exc

    logger.info(f"Rolling back {model_name} to version {target_version}")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Update latest.json to point to target version
        latest_blob_path = f"{model_name}/latest.json"
        latest_data = {
            "version": target_version,
            "path": f"gs://{bucket_name}/{model_name}/{target_version}/",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "rollback": True,
        }

        blob = bucket.blob(latest_blob_path)
        blob.upload_from_string(json.dumps(latest_data, indent=2))

        logger.info(f"Rolled back to version {target_version}")

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Push LoRA adapters and checkpoints to GCS (fitsense-models) and HuggingFace Hub."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Model-Pipeline/config/registry_config.yaml",
        help="Path to registry_config.yaml (default: Model-Pipeline/config/registry_config.yaml)",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Path to the LoRA adapter directory (overrides config)",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=None,
        help=(
            "Path to directory containing checkpoint-* subdirs "
            "(e.g., Model-Pipeline/outputs/checkpoints). "
            "Overrides config. All discovered checkpoints are uploaded to GCS."
        ),
    )
    parser.add_argument(
        "--hparam-search-dir",
        default=None,
        help=(
            "Path to hparam_search directory (e.g., Model-Pipeline/outputs/hparam_search). "
            "Overrides config. Uploaded to GCS under {model_name}/{version}/hparam_search/."
        ),
    )
    parser.add_argument(
        "--metadata-files",
        nargs="*",
        default=[],
        help="Additional metadata files to include (best_hparams.json, evaluation_results.json, etc.)",
    )
    parser.add_argument(
        "--gcs-bucket",
        default=None,
        help="GCS bucket name (overrides config; default: fitsense-models)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model identifier (overrides config; e.g., qwen3-4b-fitsense-qlora)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Explicit version tag (e.g., v20260324T120000Z). If not provided, auto-generates.",
    )
    parser.add_argument(
        "--output-dir",
        default="Model-Pipeline/outputs/registry",
        help="Local staging directory (default: Model-Pipeline/outputs/registry)",
    )
    parser.add_argument(
        "--rollback-to",
        default=None,
        help="Rollback latest.json to a previous version (e.g., v20260323T100000Z)",
    )
    parser.add_argument(
        "--hf-repo",
        default=None,
        help=(
            "HuggingFace Hub repository to push the adapter to "
            "(e.g., myuser/qwen3-8b-fitsense-lora). "
            "Skipped if not provided."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace API token. Falls back to HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage locally but do not upload to GCS or HuggingFace",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting registry push process")

        # Load registry config and apply CLI overrides (CLI takes precedence)
        cfg: dict[str, Any] = {}
        if args.config:
            try:
                cfg = load_registry_config(args.config)
                logger.info(f"Loaded registry config from: {args.config}")
            except FileNotFoundError:
                logger.warning(f"Registry config not found: {args.config}; proceeding with CLI args only")

        def _resolve(cli_val, cfg_key: str, fallback=None):
            """Return CLI value if set, else config value, else fallback."""
            return cli_val if cli_val is not None else cfg.get(cfg_key, fallback)

        gcs_bucket = _resolve(args.gcs_bucket, "gcs_bucket", "fitsense-models")
        model_name = _resolve(args.model_name, "model_name")
        adapter_dir = _resolve(args.adapter_dir, "adapter_dir")
        checkpoints_dir = _resolve(args.checkpoints_dir, "checkpoints_dir")
        hparam_search_dir = _resolve(args.hparam_search_dir, "hparam_search_dir")
        metadata_files = args.metadata_files or cfg.get("metadata_files", [])
        output_dir = _resolve(args.output_dir, "output_dir", "Model-Pipeline/outputs/registry")
        hf_repo = _resolve(args.hf_repo, "hf_repo")
        hf_token = args.hf_token
        version = _resolve(args.version, "version")
        rollback_to = args.rollback_to
        dry_run = args.dry_run

        if not model_name:
            raise ValueError("model_name is required (set in registry_config.yaml or via --model-name)")
        if not adapter_dir and not rollback_to:
            raise ValueError("adapter_dir is required (set in registry_config.yaml or via --adapter-dir)")

        # Handle rollback
        if rollback_to:
            logger.info(f"Rollback mode: reverting to {rollback_to}")
            rollback(gcs_bucket, model_name, rollback_to, logger)
            logger.info("Rollback completed successfully")
            return

        # Validate inputs
        logger.info("Validating inputs")
        if not adapter_dir:
            raise ValueError("adapter_dir is required")
        adapter_dir = str(adapter_dir)
        files_to_package = validate_inputs(adapter_dir, metadata_files, logger)
        logger.info(f"Validation passed: {len(files_to_package)} files to package")

        # Generate version if not provided
        version = version or f"v{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        logger.info(f"Using version: {version}")

        # Stage package — remove stale local copy first if it exists
        staging_subdir = Path(output_dir) / model_name / version
        if staging_subdir.exists():
            logger.info(f"Removing existing local staging directory: {staging_subdir}")
            shutil.rmtree(staging_subdir)
        logger.info(f"Staging to {staging_subdir}")
        staged_path = stage_package(
            adapter_dir,
            metadata_files,
            str(staging_subdir),
            logger,
        )

        # Write manifest
        manifest = write_manifest(
            staged_path,
            model_name,
            version,
            metadata_files,
            logger,
        )
        logger.info(f"Manifest: {json.dumps(manifest, indent=2)}")

        if dry_run:
            logger.info("Dry-run mode: skipping GCS and HuggingFace uploads")
        else:
            # --- GCS: upload adapter ---
            logger.info("Uploading adapter to GCS")
            upload_to_gcs(staged_path, gcs_bucket, model_name, version, logger)

            # --- GCS: upload checkpoints ---
            if checkpoints_dir:
                logger.info(f"Uploading checkpoints from {checkpoints_dir} to GCS")
                upload_checkpoints_to_gcs(
                    checkpoints_dir,
                    gcs_bucket,
                    model_name,
                    version,
                    logger,
                )
            else:
                logger.info("No checkpoints_dir configured; skipping checkpoint upload")

            # --- GCS: upload hparam_search ---
            if hparam_search_dir:
                logger.info(f"Uploading hparam_search from {hparam_search_dir} to GCS")
                upload_hparam_search_to_gcs(
                    hparam_search_dir,
                    gcs_bucket,
                    model_name,
                    version,
                    logger,
                )
            else:
                logger.info("No hparam_search_dir configured; skipping hparam search upload")

            # --- GCS: update registry pointers ---
            logger.info("Updating latest.json pointer")
            update_latest_pointer(gcs_bucket, model_name, version, logger)

            logger.info("Updating versions.json")
            update_versions_list(gcs_bucket, model_name, version, logger)

            logger.info(
                f"GCS push complete: gs://{gcs_bucket}/{model_name}/{version}/"
            )

            # --- HuggingFace Hub: push adapter ---
            if hf_repo:
                logger.info(f"Pushing adapter to HuggingFace Hub: {hf_repo}")
                hf_url = push_to_huggingface(
                    adapter_dir,
                    hf_repo,
                    version,
                    hf_token,
                    logger,
                )
                logger.info(f"HuggingFace push complete: {hf_url}")
            else:
                logger.info("No hf_repo configured; skipping HuggingFace push")

        # Clean up local staging directory (remove full output_dir tree)
        output_dir_path = Path(output_dir)
        if output_dir_path.exists():
            shutil.rmtree(output_dir_path)
            logger.info(f"Removed local staging directory: {output_dir_path}")

        logger.info("Registry push completed successfully")

    except (FileNotFoundError, ValueError, OSError, ImportError) as e:
        logger.error(f"Error during registry push: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during registry push: {e}")
        raise


if __name__ == "__main__":
    main()
