import os
import tarfile
import json
from random import randint
from random import choice


class AlphabetDatasetGenerator:

    def __init__(self, n_samples):
        self.n_samples = n_samples
        with open(os.path.join('configuration', 'recognition.json'), 'r') as f:
            recognition_conf = json.load(f)
        self.separators = "/:-"
        self.speeds = ['km/h', 'mp/h', 'kmh', 'mph']
        self.channels = ['CHANNEL', 'CH', 'CAMERA', 'CAM']

        self.time_code_path = os.path.join(recognition_conf["RECOGNITION_FOLDER"],
                                           recognition_conf["TIME_CODE_FILE"])
        self.date_path = os.path.join(recognition_conf["RECOGNITION_FOLDER"],
                                      recognition_conf["DATE_FILE"])
        self.day_of_week_path = os.path.join(recognition_conf["RECOGNITION_FOLDER"],
                                             recognition_conf["DAY_WEEK_FILE"])
        self.speed_path = os.path.join(recognition_conf["RECOGNITION_FOLDER"],
                                       recognition_conf["SPEED_FILE"])
        self.camera_path = os.path.join(recognition_conf["RECOGNITION_FOLDER"],
                                        recognition_conf["CAMERA_FILE"])

    def generate_sequences_dataset(self):
        """ Method for generating mono sequences for time code, speed, camera/channel number and date

        :return: it is predicat
        """
        # Generate Time Code sequences and save them
        time_code_generator = self.time_code_sequence_generator()
        with open(self.time_code_path, "w") as time_code_file:
            for time_code in time_code_generator:
                time_code_file.write(time_code)

        # Generate Date sequences and save them
        date_generator = self.date_sequence_generator()
        with open(self.date_path, "w") as date_file:
            for time_code in date_generator:
                date_file.write(time_code)

        # Generate speed sequneces and save them
        speed_generator = self.speed_sequence_generator()
        with open(self.speed_path, 'w') as speed_file:
            for speed in speed_generator:
                speed_file.write(speed)

        # Generate CAM/CHANNEL sequences and save them
        channel_generator = self.channel_sequence_generator()
        with open(self.camera_path, 'w') as camera_file:
            for channel in channel_generator:
                camera_file.write(channel)

    def time_code_sequence_generator(self):
        for i in range(self.n_samples):
            separator = choice(self.separators)
            time_code = str(randint(0, 9)) + \
                   str(randint(0, 9)) + separator + \
                   str(randint(0, 9)) + \
                   str(randint(0, 9)) + separator + \
                   str(randint(0, 9)) + \
                   str(randint(0, 9)) + '\n'
            yield time_code

    def date_sequence_generator(self):
        for i in range(self.n_samples):
            separator = choice(self.separators)
            date = str(randint(1970, 2030)) + separator + \
                   str(randint(1, 12)) + separator + \
                   str(randint(1, 31)) + '\n'
            yield date

    def speed_sequence_generator(self):
        for i in range(self.n_samples):
            unit = choice(self.speeds)
            speed = str(randint(0, 300)) + ' ' + unit + '\n'
            yield speed

    def channel_sequence_generator(self):
        for i in range(self.n_samples):
            channel = choice(self.channels)
            separator = choice(self.separators)
            speed = channel + separator + str(randint(0, 30)) + '\n'
            yield speed

    # def make_tarfile(self,):
    #     with tarfile.open(self.tarfile_path, "w:gz") as tar:
    #         tar.add(self.bi_clean_path, arcname=os.path.basename(self.bi_clean_path))
    #         tar.add(self.mono_clean_path, arcname=os.path.basename(self.mono_clean_path))


if __name__ == "__main__":
    generator = AlphabetDatasetGenerator(10000)
    generator.generate_sequences_dataset()
    print("Recognition dataset ending successfully")
