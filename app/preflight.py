import argparse
import logging
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings
from app.db import count_chunks
from app.ingest.embed import embed_texts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_checks(skip_embeddings: bool = False, skip_s3: bool = False) -> list[CheckResult]:
    results = [_check_database()]
    if not skip_embeddings:
        results.append(_check_embedding_provider())
    if not skip_s3:
        results.append(_check_s3())
    return results


def _check_database() -> CheckResult:
    try:
        chunks = count_chunks()
        return CheckResult("database", True, f"connected; chunks={chunks}")
    except Exception as exc:
        return CheckResult("database", False, f"{type(exc).__name__}: {exc}")


def _check_embedding_provider() -> CheckResult:
    try:
        settings = get_settings()
        vector = embed_texts(["preflight"])[0]
        if len(vector) != settings.embedding_dimensions:
            return CheckResult(
                "embeddings",
                False,
                f"expected {settings.embedding_dimensions} dims, got {len(vector)}",
            )
        return CheckResult("embeddings", True, f"{settings.embedding_provider}:{settings.embedding_model}")
    except Exception as exc:
        return CheckResult("embeddings", False, f"{type(exc).__name__}: {exc}")


def _check_s3() -> CheckResult:
    settings = get_settings()
    if not settings.s3_enabled:
        return CheckResult("s3", True, "disabled by S3_ENABLED=false")
    if not settings.s3_bucket.strip():
        return CheckResult("s3", False, "S3_BUCKET is required when S3_ENABLED=true")

    client = boto3.client("s3", region_name=settings.aws_region)
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
        return CheckResult("s3", True, f"bucket exists: {settings.s3_bucket}")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", type(exc).__name__)
        return CheckResult("s3", False, f"{code}: unable to access bucket {settings.s3_bucket}")
    except Exception as exc:
        return CheckResult("s3", False, f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-openai", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-s3", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    results = run_checks(skip_embeddings=args.skip_embeddings or args.skip_openai, skip_s3=args.skip_s3)
    for result in results:
        status = "ok" if result.ok else "fail"
        print(f"{status:4} {result.name}: {result.detail}")
    if not all(result.ok for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
