import datetime
import json
import os
import time

import mysql.connector
from mysql.connector import Error, pooling

_DB_CONFIG = {
    "host": os.getenv("DB_HOST", "mysql"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "banana"),
    "password": os.getenv("DB_PASSWORD", "banana123"),
    "database": os.getenv("DB_NAME", "banana_db"),
}

_pool: pooling.MySQLConnectionPool | None = None
db_available: bool = False


def wait_for_db(retries: int = 8, delay: int = 3) -> None:
    """Block until MySQL accepts connections, then return."""
    for attempt in range(1, retries + 1):
        try:
            conn = mysql.connector.connect(**_DB_CONFIG)
            conn.close()
            print("MySQL is ready.")
            return
        except Error:
            print(f"  Waiting for MySQL... ({attempt}/{retries})")
            time.sleep(delay)
    raise RuntimeError("MySQL did not become ready in time.")


def init_pool(pool_size: int = 5) -> None:
    global _pool, db_available
    _pool = pooling.MySQLConnectionPool(
        pool_name="banana_pool",
        pool_size=pool_size,
        **_DB_CONFIG,
    )
    db_available = True


def _get_conn():
    if _pool is None:
        raise RuntimeError("Database pool is not initialised.")
    return _pool.get_connection()


def save_detection(
    input_filename: str,
    result_filepath: str,
    object_count: int,
    detection_summary: dict,
) -> int:
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO detections
                (input_filename, result_filepath, object_count,
                 detection_summary, upload_timestamp, other_params)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                input_filename,
                result_filepath,
                object_count,
                json.dumps(detection_summary),
                datetime.datetime.now(),
                json.dumps({"framework": "FastAPI", "model": "YOLOv8"}),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_recent_detections(limit: int = 10) -> list:
    conn = _get_conn()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, input_filename, result_filepath, object_count,
                   detection_summary, upload_timestamp
            FROM detections
            ORDER BY upload_timestamp DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        for row in rows:
            if isinstance(row["detection_summary"], str):
                row["detection_summary"] = json.loads(row["detection_summary"])
            if isinstance(row["upload_timestamp"], datetime.datetime):
                row["upload_timestamp"] = row["upload_timestamp"].isoformat()
        return rows
    finally:
        conn.close()
