""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict

import json
language = 'english'

try:
    language = json.load(open("language_setting.json", 'r'))['language']
except:
    pass

_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:

if language == 'english':
    symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet
elif language == 'korean':
    _letters_k = ''
    for unicode in range(0x1100, 0x1113):
        _letters_k += chr(unicode)
    for unicode in range(0x1161, 0x1176):
        _letters_k += chr(unicode)
    for unicode in range(0x11A8, 0x11C3):
        _letters_k += chr(unicode)

    # Rule of Seven Jongseong
    # for unicode in (0x11a8, 0x11ab, 0x11ae, 0x11af, 0x11b7, 0x11b8, 0x11bc):
    #    _letters_k += chr(unicode)
    symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters_k)

