import os
import tarfile
from random import randint

mono_clean_path = 'recognition_dataset/wordlist_mono_clean.txt'
bi_clean_path = 'recognition_dataset/wordlist_bi_clean.txt'
tarfile_path = 'recognition_dataset/wordlists.tgz'


def create_sequence_dataset():

    with open(mono_clean_path, "w") as file:
        sequence = []
        for i in range(500000):
            text = str(randint(0, 9)) + str(randint(0, 9)) + ':' + str(randint(0, 9)) + str(randint(0, 9)) + ':' + str(
                randint(0, 9)) + str(randint(0, 9)) + '\n'
            sequence.append(text)
        file.writelines(sequence)

    with open(bi_clean_path, "w") as file:
        sequence = []
        for i in range(500000):
            text = str(randint(0, 9)) + str(randint(0, 9)) + ':' + str(randint(0, 9)) + str(randint(0, 9)) + ':' + str(
                randint(0, 9)) + str(randint(0, 9)) + ' ' + str(randint(0, 9)) + str(randint(0, 9)) + ':' + str(
                randint(0, 9)) + str(randint(0, 9)) + ':' + str(randint(0, 9)) + str(randint(0, 9)) + '\n'
            sequence.append(text)
        file.writelines(sequence)
    make_tarfile(tarfile_path)


def make_tarfile(output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(bi_clean_path, arcname=os.path.basename(bi_clean_path))
        tar.add(mono_clean_path, arcname=os.path.basename(mono_clean_path))


if __name__ == "__main__":
    create_sequence_dataset()
    print("Recognition dataset ending successfully")
