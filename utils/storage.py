
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

    def list_projects(self):
        """
        R2에서 모든 프로젝트 목록을 가져옵니다.
        videos/ 폴더에서 manifest.json 파일들을 찾아 프로젝트 정보를 반환합니다.

        Returns:
            프로젝트 정보 리스트 (최신순 정렬)
        """
        if not self.s3_client:
            print("[StorageManager] R2 client not available.")
            return []

        try:
            projects = []
            
            # R2에서 videos/ 폴더의 모든 객체 나열
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix='videos/')

            for page in pages:
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # manifest.json 파일만 처리
                    if key.endswith('/manifest.json'):
                        try:
                            # manifest.json 다운로드
                            manifest_data = self.get_object(key)
                            if manifest_data:
                                import json
                                manifest = json.loads(manifest_data.decode('utf-8'))
                                
                                # 프로젝트 정보 추출
                                project_id = manifest.get('project_id')
                                is_mv = project_id and project_id.startswith("mv_")
                                status = manifest.get('status', 'unknown')
                                is_completed = status == 'completed'
                                scenes = manifest.get('scenes', [])

                                # 썸네일: 첫 씬 이미지에서 추출
                                thumbnail_url = None
                                for sc in scenes:
                                    img_p = sc.get("image_path", "")
                                    if img_p:
                                        fname = img_p.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                                        thumbnail_url = f"/api/asset/{project_id}/image/{fname}"
                                        break

                                # MV 타이틀: title > concept > 음악 파일명 > 기본값
                                title = manifest.get('title') or ''
                                if is_mv:
                                    concept = manifest.get('concept', '') or ''
                                    if not title and concept:
                                        title = concept
                                    if not title:
                                        # music_analysis.file_path → music_file_path 순으로 폴백
                                        music_path = (manifest.get('music_analysis') or {}).get('file_path') or ''
                                        if not music_path:
                                            music_path = manifest.get('music_file_path') or ''
                                        if music_path:
                                            fname_m = music_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                                            title = fname_m.rsplit(".", 1)[0] if "." in fname_m else fname_m
                                if not title:
                                    title = '제목 없음'

                                proj_info = {
                                    'project_id': project_id,
                                    'title': title,
                                    'type': 'mv' if is_mv else 'video',
                                    'status': status,
                                    'created_at': manifest.get('created_at'),
                                    'last_modified': obj['LastModified'].isoformat() if 'LastModified' in obj else None,
                                    'video_url': (f"/api/mv/stream/{project_id}" if is_mv else f"/api/stream/{project_id}") if is_completed else None,
                                    'download_url': (f"/api/mv/download/{project_id}" if is_mv else f"/api/download/{project_id}") if is_completed else None,
                                    'thumbnail_url': thumbnail_url,
                                    'scene_count': len(scenes),
                                }

                                # MV 추가 메타데이터
                                if is_mv:
                                    ma = manifest.get('music_analysis', {})
                                    proj_info['duration_sec'] = ma.get('duration_sec')
                                    proj_info['genre'] = ma.get('genre')
                                    proj_info['style'] = manifest.get('style')

                                projects.append(proj_info)
                        except Exception as e:
                            print(f"[StorageManager] Error processing {key}: {e}")
                            continue

            # 최신순 정렬 (last_modified 기준)
            projects.sort(key=lambda x: x.get('last_modified', ''), reverse=True)
            
            print(f"[StorageManager] Found {len(projects)} projects in R2")
            return projects

        except Exception as e:
            print(f"[StorageManager] Error listing projects: {e}")
            return []

    def delete_file(self, r2_path: str) -> bool:
        """R2에서 단일 파일 삭제"""
        if not self.s3_client:
            return False
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=r2_path)
            return True
        except Exception as e:
            print(f"[StorageManager] Delete failed: {e}")
            return False

    def delete_old_projects(self, cutoff_date) -> int:
        """cutoff_date 이전 프로젝트의 R2 파일 일괄 삭제"""
        if not self.s3_client:
            return 0
        deleted = 0
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix='videos/'):
                for obj in page.get('Contents', []):
                    if obj['LastModified'].date() <= cutoff_date:
                        self.s3_client.delete_object(Bucket=self.bucket_name, Key=obj['Key'])
                        deleted += 1
        except Exception as e:
            print(f"[StorageManager] Bulk delete error: {e}")
        return deleted
