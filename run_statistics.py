from glob import glob
import librosa
import matplotlib.pyplot as plt
# import scipy.io.wavfile

def plot_histogram(lengths, keyword):
    plt.figure(figsize=(6, 6))
    plt.title(f'{keyword} file length histogram', fontsize=18)

    plt.hist(lengths,
            bins=20,
            alpha = 0.5)

    plt.savefig(f'figs/{keyword}_len_hist.png')
    plt.close()
    return

CUSTOM = ''

ljspeech_files = sorted(glob('DUMMY/*.wav'))
kss_files = sorted(glob('DUMMY_K/*/*.wav'))
custom_files = sorted(glob(f'{CUSTOM}/*.wav'))

print(f'LJSpeech files num | {len(ljspeech_files)}')
print(f'KSS files num      | {len(kss_files)}')
print(f'Custom files num   | {len(custom_files)}')

ljspeech_file_lengths = [librosa.core.get_duration(filename=f) for f in ljspeech_files]
kss_file_lengths = [librosa.core.get_duration(filename=f) for f in kss_files]
custom_file_lengths = [librosa.core.get_duration(filename=f) for f in custom_files]

plot_histogram(ljspeech_file_lengths, 'ljspeech')
plot_histogram(kss_file_lengths, 'kss')
plot_histogram(custom_file_lengths, 'custom')

print(f'LJSpeech length sum  | {sum(ljspeech_file_lengths)}')
print(f'LJSpeech length mean | {sum(ljspeech_file_lengths)/len(ljspeech_file_lengths)}')
print(f'KSS      length sum  | {sum(kss_file_lengths)}')
print(f'KSS      length mean | {sum(kss_file_lengths)/len(kss_file_lengths)}')
print(f'Custom   length sum  | {sum(custom_file_lengths)}')
print(f'Custom   length mean | {sum(custom_file_lengths)/len(custom_file_lengths)}')

# LJSpeech length sum  | 86117.07628117915
# LJSpeech length mean | 6.573822616883905
# KSS      length sum  | 46286.282947846
# KSS      length mean | 3.6009244552548623