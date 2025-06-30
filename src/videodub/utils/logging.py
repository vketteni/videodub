"""Logging configuration utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog


def configure_logging(
    level: str = "INFO",
    enable_json: bool = True,
    log_file: Optional[Path] = None,
    enable_console: bool = True
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Whether to use JSON formatting
        log_file: Optional log file path
        enable_console: Whether to log to console
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure standard library logging
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=[]  # We'll add handlers manually
    )
    
    # Create processors for structlog
    processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    
    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Set up handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        root_logger.addHandler(console_handler)
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
    
    # Get initial logger to confirm setup
    logger = structlog.get_logger(__name__)
    logger.info(
        "Logging configured",
        level=level,
        json_format=enable_json,
        log_file=str(log_file) if log_file else None,
        console_enabled=enable_console
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LoggingConfig:
    """Configuration class for logging setup."""
    
    def __init__(
        self,
        level: str = "INFO",
        enable_json: bool = True,
        log_directory: Optional[Path] = None,
        enable_console: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize logging configuration.
        
        Args:
            level: Logging level
            enable_json: Use JSON formatting
            log_directory: Directory for log files
            enable_console: Enable console logging
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
        """
        self.level = level
        self.enable_json = enable_json
        self.log_directory = Path(log_directory) if log_directory else None
        self.enable_console = enable_console
        self.max_file_size = max_file_size
        self.backup_count = backup_count
    
    def setup(self) -> None:
        """Set up logging with this configuration."""
        log_file = None
        if self.log_directory:
            self.log_directory.mkdir(parents=True, exist_ok=True)
            log_file = self.log_directory / "pipeline.log"
        
        configure_logging(
            level=self.level,
            enable_json=self.enable_json,
            log_file=log_file,
            enable_console=self.enable_console
        )
    
    def get_file_handler_with_rotation(self) -> Optional[logging.Handler]:
        """
        Get a rotating file handler.
        
        Returns:
            Configured rotating file handler or None
        """
        if not self.log_directory:
            return None
        
        from logging.handlers import RotatingFileHandler
        
        log_file = self.log_directory / "pipeline.log"
        handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        
        handler.setLevel(getattr(logging, self.level.upper(), logging.INFO))
        return handler


def setup_pipeline_logging(
    output_directory: Path,
    level: str = "INFO",
    enable_json: bool = True
) -> None:
    """
    Set up logging specifically for the translation pipeline.
    
    Args:
        output_directory: Pipeline output directory
        level: Logging level
        enable_json: Use JSON formatting
    """
    log_directory = output_directory / "logs"
    
    config = LoggingConfig(
        level=level,
        enable_json=enable_json,
        log_directory=log_directory,
        enable_console=True
    )
    
    config.setup()


def log_pipeline_start(video_id: str, url: str, config: dict) -> None:
    """
    Log pipeline start with context.
    
    Args:
        video_id: Video identifier
        url: Video URL
        config: Pipeline configuration
    """
    logger = get_logger("pipeline.start")
    logger.info(
        "Pipeline started",
        video_id=video_id,
        url=url,
        config=config
    )


def log_pipeline_end(video_id: str, success: bool, duration: float, files: dict) -> None:
    """
    Log pipeline completion with results.
    
    Args:
        video_id: Video identifier
        success: Whether pipeline succeeded
        duration: Processing duration in seconds
        files: Dictionary of generated files
    """
    logger = get_logger("pipeline.end")
    
    if success:
        logger.info(
            "Pipeline completed successfully",
            video_id=video_id,
            duration_seconds=duration,
            generated_files=files
        )
    else:
        logger.error(
            "Pipeline failed",
            video_id=video_id,
            duration_seconds=duration
        )


def log_step_start(step_name: str, video_id: str, **kwargs) -> None:
    """
    Log pipeline step start.
    
    Args:
        step_name: Name of the step
        video_id: Video identifier
        **kwargs: Additional context
    """
    logger = get_logger(f"pipeline.{step_name}")
    logger.info(
        f"Step started: {step_name}",
        video_id=video_id,
        **kwargs
    )


def log_step_end(step_name: str, video_id: str, success: bool, **kwargs) -> None:
    """
    Log pipeline step completion.
    
    Args:
        step_name: Name of the step
        video_id: Video identifier
        success: Whether step succeeded
        **kwargs: Additional context
    """
    logger = get_logger(f"pipeline.{step_name}")
    
    if success:
        logger.info(
            f"Step completed: {step_name}",
            video_id=video_id,
            **kwargs
        )
    else:
        logger.error(
            f"Step failed: {step_name}",
            video_id=video_id,
            **kwargs
        )