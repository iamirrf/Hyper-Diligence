import argparse
import logging
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError
from openai import OpenAI

from app.config import get_settings, require_openai_api_key
from app.db import count_chunks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_checks(skip_openai: bool = False, skip_s3: bool = False) -> list[CheckResult]:
    results = [_check_database()]
    if not skip_openai:
        results.append(_check_openai())
    if not skip_s3:
        results.append(_check_s3())
    return results


def _check_database() -> CheckResult:
    try:
        chunks = count_chunks()
        return CheckResult("database", True, f"connected; chunks={chunks}")
    except Exception as exc:
        return CheckResult("database", False, f"{type(exc).__name__}: {exc}")


def _check_openai() -> CheckResult:
    try:
        client = OpenAI(api_key=require_openai_api_key())
        response = client.embeddings.create(model="text-embedding-3-small", input=["preflight"])
        dimensions = len(response.data[0].embedding)
        if dimensions != 1536:
            return CheckResult("openai", False, f"expected 1536 embedding dims, got {dimensions}")
        return CheckResult("openai", True, "embedding request succeeded")
    except Exception as exc:
        return CheckResult("openai", False, f"{type(exc).__name__}: {exc}")


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
    parser.add_argument("--skip-openai", action="store_true")
    parser.add_argument("--skip-s3", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    results = run_checks(skip_openai=args.skip_openai, skip_s3=args.skip_s3)
    for result in results:
        status = "ok" if result.ok else "fail"
        print(f"{status:4} {result.name}: {result.detail}")
    if not all(result.ok for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
