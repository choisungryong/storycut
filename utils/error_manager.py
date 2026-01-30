import json
import os
import datetime
import traceback
from typing import Dict, Any, List

class ErrorManager:
    """
    Centralized manager for logging and retrieving API errors.
    """
    
    LOG_FILE = "outputs/api_errors.log"
    
    @classmethod
    def log_error(
        cls, 
        service: str, 
        error_message: str, 
        details: Any = None, 
        severity: str = "error"
    ):
        """
        Log an error to the log file.
        
        Args:
            service: Name of the service/agent (e.g., "StoryAgent", "VeoAPI")
            error_message: Brief error description
            details: Additional context or full traceback
            severity: Error severity ("warning", "error", "critical")
        """
        timestamp = datetime.datetime.now().isoformat()
        
        entry = {
            "timestamp": timestamp,
            "service": service,
            "message": error_message,
            "details": str(details) if details else None,
            "severity": severity
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(cls.LOG_FILE), exist_ok=True)
        
        try:
            # Read existing logs
            logs = []
            if os.path.exists(cls.LOG_FILE):
                try:
                    with open(cls.LOG_FILE, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        if file_content.strip():
                            logs = json.loads(file_content)
                except json.JSONDecodeError:
                    logs = [] # Reset if corrupted
            
            # Append new log
            logs.append(entry)
            
            # Key history limited (e.g., last 100 errors)
            if len(logs) > 100:
                logs = logs[-100:]
                
            # Write back
            with open(cls.LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
            print(f"[{severity.upper()}] {service}: {error_message}")
            
        except Exception as e:
            # Fallback print if logging fails
            print(f"[CRITICAL] Failed to write to error log: {e}")
            print(f"Original Error: [{service}] {error_message}")

    @classmethod
    def get_recent_errors(cls, limit: int = 20) -> List[Dict]:
        """Get recent error logs."""
        if not os.path.exists(cls.LOG_FILE):
            return []
            
        try:
            with open(cls.LOG_FILE, 'r', encoding='utf-8') as f:
                logs = json.load(f)
                return sorted(logs, key=lambda x: x['timestamp'], reverse=True)[:limit]
        except Exception:
            return []

    @classmethod
    def clear_logs(cls):
        """Clear the error log file."""
        if os.path.exists(cls.LOG_FILE):
            os.remove(cls.LOG_FILE)
