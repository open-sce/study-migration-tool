import logging

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)

logger = logging.getLogger("main")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s'))
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)
