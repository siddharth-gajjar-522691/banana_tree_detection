CREATE DATABASE IF NOT EXISTS banana_db;
USE banana_db;

CREATE TABLE IF NOT EXISTS detections (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    user_id           VARCHAR(255)    DEFAULT NULL,
    input_filename    VARCHAR(255)    NOT NULL,
    result_filepath   TEXT,
    object_count      INT             NOT NULL DEFAULT 0,
    detection_summary JSON,
    upload_timestamp  DATETIME        NOT NULL,
    other_params      JSON
);
