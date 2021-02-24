import os
import math
import json
import librosa

DATESET_PATH = os.path.join(os.getcwd(), 'Data/genres_original')
JSON_PATH = os.path.join(os.getcwd(), 'data.json')
DURATION = 30  # seconds
SAMPLE_RATE = 22050
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_ftt=2048, hop_length=512, num_segments=5):
    data = {
        "class_names": [],
        "mfcc": [],
        "labels": []
    }
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    for i, (dir_path, dir_names, file_names) in enumerate(os.walk(dataset_path)):
        if dir_path is not DATESET_PATH:

            # genres as class names
            label = dir_path.split('/')[-1]
            data['class_names'].append(label)
            print(f'\nProcessing {label}')
            for file_name in file_names:
                try:
                    # load audio file
                    file_path = os.path.join(dir_path, file_name)
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    for index in range(num_segments):
                        start_index = index * num_samples_per_segment
                        end_index = start_index + num_samples_per_segment
                        meltspec_args = {"n_fft": n_ftt, "hop_length": hop_length, }

                        mfcc = librosa.feature.mfcc(signal[start_index:end_index],
                                                    sr=sr,
                                                    n_mfcc=n_mfcc,
                                                    **meltspec_args).T
                        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                            data['mfcc'].append(mfcc.tolist())
                            data['labels'].append(i - 1)
                            print(f'{file_path}, segment:{index + 1}')
                except Exception:
                    print('shit happens')

    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':
    save_mfcc(DATESET_PATH, JSON_PATH, num_segments=10)
