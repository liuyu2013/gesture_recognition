import os
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import logger

datas, binaries, hiddenimports = collect_all('util', include_py_files=False)

logger.info('Collecting suanpan datas: {}'.format(datas))
logger.info('Collecting suanpan hiddenimports: {}'.format(hiddenimports))