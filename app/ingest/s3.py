import logging

import boto3

from app.config import get_settings

logger = logging.getLogger(__name__)


def put_raw_filing(bucket: str, key: str, content: str) -> str | None:
    settings = get_settings()
    if not settings.s3_enabled:
        logger.warning("s3_disabled_skipping_put", extra={"bucket": bucket, "key": key})
        return None
    if not bucket:
        raise ValueError("S3 bucket is required when S3_ENABLED=true")
    client = boto3.client("s3", region_name=settings.aws_region)
    client.put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"), ContentType="text/html; charset=utf-8")
    logger.info("raw_filing_archived", extra={"bucket": bucket, "key": key})
    return key
