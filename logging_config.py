import logging
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# https://docs.python.org/3/library/logging.html#logrecord-attributes
formatter = logging.Formatter(fmt="[%(asctime)s] %(levelname)s %(funcName)s(): %(message)s")

handlers = [
    logging.StreamHandler(sys.stdout),   # stdout
    # logging.FileHandler()
]

for h in handlers:
    h.setFormatter(formatter)
    logger.addHandler(h)
