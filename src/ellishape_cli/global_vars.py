from pathlib import Path
from sys import stderr

from loguru import logger as log

NAME = 'Ellishape'

ROOT_DIR = Path().home().absolute() / ('.'+NAME)
if not ROOT_DIR.exists():
    ROOT_DIR.mkdir()
log_file = ROOT_DIR / 'Log.txt'
# detailed file log
log.remove()
log.add(log_file, level='DEBUG', backtrace=True, diagnose=True, filter=NAME,
        mode='a', encoding='utf-8')
# simple stdout log
log.add(stderr, colorize=True, level='INFO')
log.info(f'Start {NAME}')
log.info(f'Log file {log_file}')

