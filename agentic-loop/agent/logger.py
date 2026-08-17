import sqlite3
import json
import time
from typing import Dict, Any

class StructuredLogger:
    def __init__(self, db_path: str = "agent_logs.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    iteration INTEGER,
                    step_name TEXT,
                    payload TEXT,
                    latency_ms REAL,
                    error TEXT
                )
            ''')
            conn.commit()
            
    def log_step(self, iteration: int, step_name: str, payload: Dict[str, Any], latency_ms: float = 0.0, error: str = None):
        """
        Logs a structured entry to the SQLite database.
        """
        timestamp = time.time()
        payload_str = json.dumps(payload)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO logs (timestamp, iteration, step_name, payload, latency_ms, error)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, iteration, step_name, payload_str, latency_ms, error))
            conn.commit()
            
        print(f"[{step_name.upper()}] Iteration {iteration} | Latency: {latency_ms:.1f}ms" + (f" | ERROR: {error}" if error else ""))
