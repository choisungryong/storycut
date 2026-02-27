"""
STORYCUT File Cleanup Scheduler

Policy:
- Images & intermediate files (video/, rendered/, audio/, subtitles/, characters/, scenes/):
  deleted next day at 03:00
- Final video (final_video.mp4) + manifest.json:
  deleted 3 days later at 03:00
- Empty project directories: removed immediately after cleanup
- R2 cloud storage: videos older than 3 days deleted
"""

import os
import glob
import shutil
import asyncio
from datetime import datetime, date, timedelta
from utils.logger import get_logger
logger = get_logger("cleanup")



def run_cleanup():
    """Run file cleanup based on retention policy. Returns count of deleted items."""
    today = date.today()
    deleted = 0

    # 1) Intermediate files: delete if created before today
    patterns = [
        "outputs/*/media/images/*",
        "outputs/*/media/video/*",
        "outputs/*/media/rendered/*",
        "outputs/*/media/audio/*",
        "outputs/*/media/subtitles/*",
        "outputs/*/media/characters/*",
        "outputs/*/scenes/*",
    ]
    for pattern in patterns:
        for f in glob.glob(pattern):
            try:
                if datetime.fromtimestamp(os.path.getmtime(f)).date() < today:
                    os.remove(f)
                    deleted += 1
            except Exception as e:
                logger.error(f"[CLEANUP] Failed to delete {f}: {e}")

    # 2) Final video + manifest: delete if older than 3 days
    cutoff = today - timedelta(days=3)
    for f in glob.glob("outputs/*/final_video*.mp4") + glob.glob("outputs/*/manifest.json"):
        try:
            if datetime.fromtimestamp(os.path.getmtime(f)).date() <= cutoff:
                os.remove(f)
                deleted += 1
        except Exception as e:
            logger.error(f"[CLEANUP] Failed to delete {f}: {e}")

    # 3) R2 cloud storage: delete videos older than 3 days
    try:
        from utils.storage import StorageManager
        storage = StorageManager()
        r2_deleted = storage.delete_old_projects(cutoff_date=cutoff)
        deleted += r2_deleted
    except Exception as e:
        logger.error(f"[CLEANUP] R2 cleanup error: {e}")

    # 4) Remove empty project directories
    for d in glob.glob("outputs/*"):
        if not os.path.isdir(d):
            continue
        try:
            has_files = any(files for _, _, files in os.walk(d) if files)
            if not has_files:
                shutil.rmtree(d)
                deleted += 1
        except Exception as e:
            logger.error(f"[CLEANUP] Failed to remove directory {d}: {e}")

    return deleted


async def _cleanup_loop():
    """Async loop that runs cleanup at 03:00 daily."""
    while True:
        now = datetime.now()
        next_3am = now.replace(hour=3, minute=0, second=0, microsecond=0)
        if now >= next_3am:
            next_3am += timedelta(days=1)
        wait = (next_3am - now).total_seconds()
        logger.info(f"[CLEANUP] Next run at {next_3am} (in {wait/3600:.1f}h)")
        await asyncio.sleep(wait)
        try:
            deleted = run_cleanup()
            logger.info(f"[CLEANUP] Completed: {deleted} items deleted")
        except Exception as e:
            logger.error(f"[CLEANUP] Error: {e}")


def start_cleanup_scheduler():
    """Register the cleanup loop in the current asyncio event loop."""
    asyncio.ensure_future(_cleanup_loop())
