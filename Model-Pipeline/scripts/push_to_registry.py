"""Push a LoRA adapter to Google Cloud Storage and register it.

This script packages a selected LoRA adapter with metadata and pushes it to
Google Cloud Storage (GCS) and registers it in GCP Artifact Registry. Supports
versioning, tagging, and rollback.
"""

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    import shutil

    adapter_path = Path(adapter_dir)
    staging_path = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Staging package to {staging_path}")

    # Copy adapter files
    for src_file in collect_adapter_files(adapter_path):
        rel_path = src_file.relative_to(adapter_path)
        dest_file = staging_path / rel_path
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
        for local_path in staging_dir.rglob("*"):
            if not local_path.is_file():
                continue

            rel_path = local_path.relative_to(staging_dir)
            blob_path = f"{model_name}/{version}/{rel_path}"
            blob = bucket.blob(blob_path)

            try:
                blob.upload_from_filename(str(local_path))
                logger.debug(f"Uploaded {blob_path}")
            except Exception as e:
                logger.error(f"Failed to upload {blob_path}: {e}")
                raise

        logger.info(f"All files uploaded successfully to {bucket_name}/{model_name}/{version}/")

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
        description="Push a LoRA adapter to GCS and register it."
    )
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--metadata-files",
        nargs="*",
        default=[],
        help="Additional metadata files to include (best_hparams.json, evaluation_results.json, etc.)",
    )
    parser.add_argument(
        "--gcs-bucket",
        required=True,
        help="GCS bucket name (e.g., fitsense-models)",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model identifier (e.g., qwen3-8b-fitsense-qlora)",
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
        "--dry-run",
        action="store_true",
        help="Stage locally but do not upload to GCS",
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

        # Handle rollback
        if args.rollback_to:
            logger.info(f"Rollback mode: reverting to {args.rollback_to}")
            rollback(args.gcs_bucket, args.model_name, args.rollback_to, logger)
            logger.info("Rollback completed successfully")
            return

        # Validate inputs
        logger.info("Validating inputs")
        files_to_package = validate_inputs(args.adapter_dir, args.metadata_files, logger)
        logger.info(f"Validation passed: {len(files_to_package)} files to package")

        # Generate version if not provided
        version = args.version or f"v{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        logger.info(f"Using version: {version}")

        # Stage package
        staging_subdir = (
            Path(args.output_dir) / args.model_name / version
        )
        logger.info(f"Staging to {staging_subdir}")
        staged_path = stage_package(
            args.adapter_dir,
            args.metadata_files,
            str(staging_subdir),
            logger,
        )

        # Write manifest
        manifest = write_manifest(
            staged_path,
            args.model_name,
            version,
            args.metadata_files,
            logger,
        )
        logger.info(f"Manifest: {json.dumps(manifest, indent=2)}")

        # Upload to GCS (unless dry-run)
        if args.dry_run:
            logger.info("Dry-run mode: skipping GCS upload")
        else:
            logger.info("Uploading to GCS")
            upload_to_gcs(
                staged_path,
                args.gcs_bucket,
                args.model_name,
                version,
                logger,
            )

            logger.info("Updating latest.json pointer")
            update_latest_pointer(args.gcs_bucket, args.model_name, version, logger)

            logger.info("Updating versions.json")
            update_versions_list(args.gcs_bucket, args.model_name, version, logger)

        logger.info("Registry push completed successfully")
        logger.info(
            f"Model available at: gs://{args.gcs_bucket}/{args.model_name}/{version}/"
        )

    except (FileNotFoundError, ValueError, OSError, ImportError) as e:
        logger.error(f"Error during registry push: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during registry push: {e}")
        raise


if __name__ == "__main__":
    main()
