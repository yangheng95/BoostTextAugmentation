import os

from findfile import rm_dirs

rm_dirs(os.getcwd(), or_key=['cache', '.idea', '.mv'])
