import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams["font.family"] = 'NanumGothic'
import os
from collections import Counter
import jamotools

source = 'filelists/custom_audio_text_train_filelist.txt'
target = 'filelists/kss_audio_text_train_filelist.txt'
reduced_target = 'filelists/kss_reduced_audio_text_train_filelist.txt'

with open(source, 'r') as f:
    source_data_list = [line.split('|') for line in f]

with open(target, 'r') as f:
    target_data_list = [line.split('|') for line in f]

source_data_lengths = [(f, script, librosa.core.get_duration(filename=f)) for f, script in tqdm(source_data_list)]
target_data_lengths = [(f, script, librosa.core.get_duration(filename=f)) for f, script in tqdm(target_data_list)]

source_data_lengths.sort(key=lambda x: x[-1])
target_data_lengths.sort(key=lambda x: x[-1])

reduced_target_data_lengths = list()

def find_closest_length(target, sec):

    old_diff = 1000

    for i, t in enumerate(target):
        diff = abs(t[-1] - sec)
        if diff > old_diff:
            return i - 1
        old_diff = diff
    
    return i

def initialize_jamo_counter():
    _letters_k = ''
    for unicode in range(0x1100, 0x1113):
        _letters_k += chr(unicode)
    for unicode in range(0x1161, 0x1176):
        _letters_k += chr(unicode)
    for unicode in range(0x11A8, 0x11C3):
        _letters_k += chr(unicode)
        
    _punctuation = '!\'(),.:;? '

    symbols = list(_letters_k) + list(_punctuation)

    c = Counter()
    for s in symbols:
        c[s] = 0

    return c

def get_script_jamo_counts(data_lengths):
    c = initialize_jamo_counter()
    for f, script, dur in  data_lengths:

        jamos = jamotools.split_syllables(script.strip(), jamo_type="JAMO")

        for j in jamos:
            if j not in c:
                # Join and resplit jamo
                jamos = jamotools.split_syllables(jamotools.join_jamos(jamos), jamo_type="JAMO")
                break

        c.update(jamos)

    return c

for _, _, length in source_data_lengths:
    idx = find_closest_length(target_data_lengths, length)
    value = target_data_lengths.pop(idx)
    reduced_target_data_lengths.append(value)
    # print(f'{length:5.2f}, {value[-1]:5.2f}')

if not os.path.isfile(reduced_target):
    with open(reduced_target, 'w') as file:
        for f, script, _ in reduced_target_data_lengths:
            file.write(f + '|' + script)

plt.figure()
plt.hist([l for (_, _, l) in source_data_lengths], 
         rwidth=0.9, alpha=0.5, label='source', bins=range(0, 12))
plt.hist([l for (_, _, l) in reduced_target_data_lengths], 
         rwidth=0.9, alpha=0.5, label='target', bins=range(0, 12)) 
plt.legend()
plt.savefig('figs/reduced.png')

source_jamo_counter = get_script_jamo_counts(source_data_lengths)
reduced_jamo_counter = get_script_jamo_counts(reduced_target_data_lengths)
del source_jamo_counter['1']
target_jamo_counter = get_script_jamo_counts(target_data_lengths)

print(target_jamo_counter.keys())

# plt.figure(figsize=(18, 6))
# fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
fig, axes = plt.subplots(4, 1, figsize=(18, 12))
axes[0].set_title('Target Jamo Distribution')
axes[0].bar(target_jamo_counter.keys(), target_jamo_counter.values(), alpha=0.3, label='target')

axes[1].set_title('Reduced Target Jamo Distribution')
axes[1].bar(reduced_jamo_counter.keys(), reduced_jamo_counter.values(), alpha=0.3, label='reduced')

axes[2].set_title('Source Jamo Distribution')
axes[2].bar(source_jamo_counter.keys(), source_jamo_counter.values(), alpha=0.3, label='source')

axes[3].set_title('Jamo Distributions')
axes[3].bar(range(len(target_jamo_counter)), target_jamo_counter.values(), alpha=0.3, label='target')
axes[3].bar(range(len(reduced_jamo_counter)), reduced_jamo_counter.values(), alpha=0.3, label='reduced')
axes[3].bar(range(len(source_jamo_counter)), source_jamo_counter.values(), alpha=0.3, label='source')
plt.xticks(range(len(target_jamo_counter)), 
        list(map(jamotools.normalize_to_compat_jamo, target_jamo_counter.keys())))
axes[3].set_yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('figs/jamo_distributions.png')

