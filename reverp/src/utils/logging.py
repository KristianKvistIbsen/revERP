def setup_logging(config: dict) -> None:
    """
    Set up logging configuration based on the provided config dictionary.
    Args:
        config: Configuration dictionary containing logging settings
    The function expects the following structure in config:
    {
        'logging': {
            'level': str,  # e.g., 'INFO', 'DEBUG', 'WARNING'
            'file': str,   # path to log file
            'format': str  # log format string
        }
    }
    """
    import logging
    import os
    from pathlib import Path

    # Extract logging configuration
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO').upper())
    log_file = log_config.get('file', 'logs/reverp.log')
    log_format = log_config.get('format',
                               '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create logs directory if it doesn't exist
    log_dir = Path(log_file).parent
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging with file handler only
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file)
        ]
    )

    # Create a test log entry to verify setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized with level {log_level}")