import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt

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

for _, _, length in source_data_lengths:
    idx = find_closest_length(target_data_lengths, length)
    value = target_data_lengths.pop(idx)
    reduced_target_data_lengths.append(value)
    # print(f'{length:5.2f}, {value[-1]:5.2f}')

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
