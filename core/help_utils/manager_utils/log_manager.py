# @Time: 2022/5/30 9:16
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:log_manager.py

import sys
import logging
from termcolor import colored
from logging.handlers import TimedRotatingFileHandler

__all__ = ['Logger']


class _MyFormatter(logging.Formatter):
    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRNNING ' + msg, 'yellow', attrs=['blink'])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('ERROR ' + msg, 'red', attrs=['blink', 'underline'])
        elif record.levelno == logging.DEBUG or record.levelno == logging.INFO:
            fmt = date + ' ' + colored('INFO ' + msg, 'blue', attrs=['bold'])
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)


class Logger(object):
    def __init__(self, log_file, log_name="AI"):
        self.log_name = log_name
        self.log_file = log_file
        self.logger = logging.getLogger(self.log_name)
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            fh = TimedRotatingFileHandler(self.log_file, when="D", encoding='utf-8', interval=1, backupCount=7)
            fh.suffix = "%Y-%m-%d_%H-%M-%S.log"
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter('[%(asctime)s @%(filename)s:%(lineno)d] %(levelname)s: %(message)s'))
            self.logger.addHandler(fh)

            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(_MyFormatter(datefmt='%y-%m-%d %H:%M:%S'))
            self.logger.addHandler(ch)