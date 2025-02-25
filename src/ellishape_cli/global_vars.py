from pathlib import Path
from sys import stdout

from loguru import logger as log

NAME = 'Ellishape'

ROOT_DIR = Path().home().absolute() / ('.'+NAME)
if not ROOT_DIR.exists():
    ROOT_DIR.mkdir()
log_file = ROOT_DIR / 'Log.txt'
# detailed file log
log.remove()
fmt = ('<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
       '<level>{level: <8}</level> | '
       # '<cyan>{name}</cyan>:'
       '<cyan>{function}</cyan>:'
       '<cyan>{line}</cyan> - '
       '<level>{message}</level>')
log.add(log_file, format=fmt, level='DEBUG', backtrace=True, diagnose=True,
        filter=NAME,
        mode='a', encoding='utf-8', enqueue=True)
# simple stdout log
# log.add(stdout, format=fmt, colorize=True, level='DEBUG', enqueue=True)
log.add(stdout, format=fmt, colorize=True, level='INFO', enqueue=True)
# log.info(f'Start {NAME}')
# log.info(f'Log file {log_file}')

