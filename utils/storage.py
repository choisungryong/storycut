
import os
import boto3
from botocore.exceptions import ClientError
from typing import Optional

class StorageManager:
    """
    Manages file interactions with Cloudflare R2 (S3 compatible).
    """

    def __init__(self):
        self.account_id = os.getenv("R2_ACCOUNT_ID")
        self.access_key = os.getenv("R2_ACCESS_KEY_ID")
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("R2_BUCKET_NAME")
        
        self.s3_client = None
        
        if all([self.account_id, self.access_key, self.secret_key, self.bucket_name]):
            try:
                self.s3_client = boto3.client(
                    service_name='s3',
                    endpoint_url=f'https://{self.account_id}.r2.cloudflarestorage.com',
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    region_name='auto'  # Must be 'auto' for Cloudflare R2
                )
                print(f"[StorageManager] Initialized R2 client for bucket: {self.bucket_name}")
            except Exception as e:
                print(f"[StorageManager] Failed to initialize R2 client: {e}")
        else:
            print("[StorageManager] R2 credentials missing. Skipping R2 initialization.")

    def upload_file(self, local_path: str, r2_path: str) -> bool:
        """
        Uploads a local file to R2.
        
        Args:
            local_path: Path to the local file
            r2_path: Destination path in R2 (key)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.s3_client:
            print("[StorageManager] R2 client not available. Upload skipped.")
            return False

        if not os.path.exists(local_path):
            print(f"[StorageManager] Local file not found: {local_path}")
            return False

        try:
            print(f"[StorageManager] Uploading {local_path} to R2://{self.bucket_name}/{r2_path}...")
            self.s3_client.upload_file(local_path, self.bucket_name, r2_path)
            print(f"[StorageManager] Upload successful!")
            return True
        except ClientError as e:
            print(f"[StorageManager] Upload failed: {e}")
            return False
        except Exception as e:
            print(f"[StorageManager] Unexpected error during upload: {e}")
            return False

    def get_object(self, r2_path: str) -> Optional[bytes]:
        """
        R2에서 파일을 다운로드합니다.

        Args:
            r2_path: R2 내 파일 경로 (key)

        Returns:
            파일 바이트 데이터 또는 None
        """
        if not self.s3_client:
            print("[StorageManager] R2 client not available.")
            return None

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=r2_path)
            return response['Body'].read()
        except ClientError as e:
            print(f"[StorageManager] Download failed: {e}")
            return None
        except Exception as e:
            print(f"[StorageManager] Unexpected error during download: {e}")
            return None
