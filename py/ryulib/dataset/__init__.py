import sys, os
import warnings
if(not(os.path.dirname(os.path.realpath(__file__)) in sys.path)):
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import getFilesPath,kanji2unicode
import sscd
import loader
from baseset import RyuImageset