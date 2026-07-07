from functools import lru_cache

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .config import AWS_ACCESS_KEY_ID, AWS_REGION, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME


@lru_cache(maxsize=1)
def get_s3_client():
    if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME):
        raise RuntimeError(
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / S3_BUCKET_NAME are not "
            "fully set in the environment."
        )
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )


def _object_key(user_key: str, filename: str) -> str:
    # e.g. "3f2a.../paper1.pdf" -- mirrors the Alice/ Bob/ folder layout,
    # but keyed by user id so it's stable even if a name/email changes.
    return f"{user_key}/{filename}"


def upload_paper(user_key: str, filename: str, content: bytes) -> str:
    """Uploads a PDF to S3 under the user's folder and returns its storage URL."""
    key = _object_key(user_key, filename)
    client = get_s3_client()
    try:
        client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=content,
            ContentType="application/pdf",
        )
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to upload '{filename}' to S3: {exc}") from exc

    return f"s3://{S3_BUCKET_NAME}/{key}"


def download_paper(user_key: str, filename: str) -> bytes:
    key = _object_key(user_key, filename)
    client = get_s3_client()
    try:
        response = client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        return response["Body"].read()
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to download '{filename}' from S3: {exc}") from exc


def delete_paper_object(user_key: str, filename: str) -> None:
    key = _object_key(user_key, filename)
    client = get_s3_client()
    try:
        client.delete_object(Bucket=S3_BUCKET_NAME, Key=key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to delete '{filename}' from S3: {exc}") from exc