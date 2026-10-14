import logging
import os
import sys

def setup_logger(name: str = "hybrid_stock", log_file: str = "hybrid_system.log", level: int = logging.INFO) -> logging.Logger:
    """Khởi tạo logger hiển thị console và ghi ra file log."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file:
            os.makedirs("logs", exist_ok=True)
            fh = logging.FileHandler(os.path.join("logs", log_file), encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
