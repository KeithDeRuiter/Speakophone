# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:36:53 2019

Speech synthesis testing using C64 allophones.
Allophone dict mapping from: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
Using ARPABet "letters"

@author: Keith
"""

import sounddevice as sd
import numpy as np
from scipy.signal import butter, lfilter
import scipy.io.wavfile as wv
from pathlib import Path
import os
import re



def butter_lowpass(cutoff, fs, order=5):
    """
    Creates a Butterworth Low Pass filter with the specified parameters.
    See scipy.signal.butter for more detail.
    
    Args:
        cutoff: The cutoff frequency in Hz (or whatever the same units as Fs)
        
        fs: The sampling frequency of the system.
        
        order: order of the filter, defaults to 5.
        
    Returns:
        b, a (ndarray): Numerator (b) and Denominator (a) polynomials of the 
        IIR filter.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Applies Butterworth Low Pass filter with the specified parameters to the 
    data passed in.
    See scipy.signal.butter and scipy.signal.lfilter for more detail.
    
    Args:
        data: The input data samples to be filtered.

        cutoff: The cutoff frequency in Hz (or whatever the same units as Fs)
        
        fs: The sampling frequency of the system.
        
        order: order of the filter, defaults to 5.
        
    Returns:
        An array containing the filtered data.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def trim_silence(audio, threshold=200):
    """
    Trims the "silence" at the start and end of the audio clip.  Specifically,
    Removes all samples at the start less than the threshold up until the 
    first occurrane of a sample whose magnitude is greater than the threshold.
    Similarly removes all samples at the end of the data after the last 
    occurance of a sample whose magnitude is larger than the threshold.
    
    Args:
        audio: The NumPy Array containing the audio to trim.
        
        threshold: The magnitude level indicating that the desired signal has 
        started.  Samples below this threshold at the start and end of the 
        data will be removed.
        
    Returns:
        The subset of the original audio with any applicable trimming at the
        beginning and end of the data.  Returns np.empty(1) if all values in
        the audio are less than the threshold.
    """
    if audio.max() < threshold:
        print("Trimming everything below threshold, returning empty")
        return np.empty(1)
    
    #Get index of first/last value larger than 200
    transient_start_index = np.where(abs(audio) > threshold)[0][0]
    transient_stop_index = np.where(abs(audio) > threshold)[0][-1]
    print("trimming to: " + str(transient_start_index) + ":" + str(transient_stop_index))
    
    return audio[transient_start_index:transient_stop_index]


def load_samples(directory):
    """
    Loads the allophone samples from the provided directory.
    These are stored in a dict keyed by filename (minus .wav) as "int16"
    NumPy arrays.  Silence at the start and end is trimmed, and the audio is
    run through a lowpass filter (Fc 5000Hz).  All ".wav" files in the 
    directory are loaded.
    
    Args:
        directory: The directory to search for samples.
        
    Returns:
        allophones: A dict of the loaded allophone samples, keyed by filename
        with a value of the loaded samples in a NumPy array.
    
    """
    allophones = dict()
    
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            joined_path = os.path.join(directory, file)
            p = Path(joined_path)
            allo_name = p.resolve().stem.strip()
            print("Loading file: {0}".format(p))
            
            fs, wav_array = wv.read(joined_path)
            print("{0} with {1} samples at {2}".format(allo_name, len(wav_array), fs))
            wav_array = trim_silence(wav_array, threshold=300)
            wav_array = butter_lowpass_filter(wav_array, 5000, fs, order=6)
            wav_array = wav_array.astype("int16")
            allophones[allo_name] = wav_array
            
    return allophones


def load_cmu_dict(dict_file_path):
    """
    Load the CMU Sphinx Dictionary of allophones for words.  Comment lines 
    starting with ";" are ignored.  Data is returned as a dict with a key of
    the word itself (all caps) and a value of a space-delimited string of the
    phones which make up that word.  The lines in the dictionary should be 
    tab-delimited to separate the word from the phones "list" when loading.
    
    Args:
        dict_file_path: The path to the dictionary file to load.
        
    Returns:
        A dict of the loaded words mapped to their phones as described above.
    """
    cmu_dict = dict()    
    
    with open(dict_file_path, "r") as dict_file:
        print("Loading Dictionary")
        
        for line in dict_file:
            line = line.strip()
            if line.startswith(';'):
                continue
            #print(line)
            word, phones = line.split("\t")
            cmu_dict[word] = phones.strip()
                
        print("Done")
        
    return cmu_dict


def load_allophone_map(phones_file_path):  
    """
    Loads a configuration file of sound mappings.  Each line should contain a
    comma-delimited pair where the first entry is one of the phones used in 
    the CMU Allophone Dictionary, and the second entry is the name of the sound
    it should be mapped to.
    
    Args:
        phones_file_path: The path to the mapping file to load.
        
    Returns:
        A dict of CMU Phones to the Mapped Phones as strings.
    """
    allo_map = dict()
    
    with open(phones_file_path, "r") as phones_file:
        print("Loading Dictionary Allophones")
        for line in phones_file:
            m = line.split(',')
            cmu_phone = m[0].strip()
            c64_phone = m[1].strip()
            
            allo_map[cmu_phone] = c64_phone
        print("Done")
        
    return allo_map



class Speakophone:
    def __init__(self, sample_dir, dict_file_path, allo_map_file_path):
        self.sounds = load_samples(sample_dir)
        self.cmu_dict = load_cmu_dict(dict_file_path)
        self.allo_map = load_allophone_map(allo_map_file_path)
        self.interword_pad = np.zeros(4000).astype("int16")


    def write_sounds_test(self):
        """
        A utility method to write out all of the loaded sounds for examination.
        These will be written with the suffix "_TESTOUT.wav"
        """
        devices = sd.query_devices()
        print(devices)    
        print()
        print(str(len(self.sounds)) + " sounds found")
        
        #Test writing
        for sk, sv in self.sounds.items():
            n = sk + "_TESTOUT.wav"
            wv.write(n, 44100, sv)
        
        
    def generate_audio(self, phrase):
        """
        Generates C64 phone -based audio from the phrase provided, provided 
        that the requisite words and their phe mappings appear in the loaded 
        dictionary.  The prase will be stripped to contain only letters,
        spaces, and apostrophes.  Words will be split based on whitespace.
        
        Args:
            phrase (str): The phrase which should be "spoken" in the 
            generated audio.
            
        Raises:
            ValueError if a word from the provided phrase cannot be found in
            the reference dictionary.
            
        Returns:
            final_audio: The resultant audio generated to say the phrase as a
            NumPy array of samples.
        """
        phrase = phrase.strip().upper()
        phrase = re.sub('[^A-Z \']+', '', phrase)
        words = phrase.split()
        clips = list()

        for w in words:
            print("Saying: {0}".format(w))
            try:
                for phone in self.cmu_dict[w].split(" "):
                    print("DictPhone: {0}\tMapPhone: {1}".format(phone, str(self.allo_map[phone])))
                    s = self.sounds[self.allo_map[phone]]
                    clips.append(s)
            except KeyError:
                raise ValueError("The word \"{0}\" is not in the dictionary".format(w))
            print("Word audio length: {}".format(len(s)))
            clips.append(self.interword_pad)

        #Combine audio from words/sounds together
        final_audio = np.hstack(clips)
        print("Total audio length: {}".format(len(final_audio)))
        return final_audio
    
    
    def output_audio(self, audio, output_file=None, fs=44100):
        """
        Takes the provided audio and outputs it either to a sound device or 
        (if provided) a file.  Auio is written at Fs of 44.1 kHz unless 
        otherwise provided.
        
        Args:
            audio: An array of samples for the audio to be output.
            
            output_file (optional): The file to output the audio to, defaults
            to None and plays to a sound device instead.
            
            fs (optional): The sample frequency (in Hz) to write to an output 
            file, defaults to 44100 Hz.
        """
        if output_file is not None:
            wv.write(output_file, 44100, audio)
        else:
            sd.play(audio)


    def say(self, phrase):
        """
        Convenience method to generate and play aloud some phrase.
        Combines generate_audio and output_audio from this class.
        
        Args:
            phrase (str): The phrase which should be "spoken" in the 
            generated audio.
            
        Raises:
            ValueError if a word from the provided phrase cannot be found in
            the reference dictionary.
        """
        self.output_audio(self.generate_audio(phrase))
        
    
def main():
    print("Running Speakophone")
  
    #di_path = os.path.join(directory, dict_filename)
    
    samp_dir = "../Samples/Keith-AllophonesWords-v2"
    dict_file = "../Samples/CMU-SphinxDict/cmudict_SPHINX_40.txt"
    map_file = "../Samples/Keith-AllophonesWords-v2/SphinxPhones_40__Keith_mapping.txt"
    
    app = Speakophone(sample_dir=samp_dir,
                      dict_file_path=dict_file,
                      allo_map_file_path=map_file)

    
    #Read input and speak
    try:
        keep_speaking = True
        while keep_speaking:
            write_file = False
            phrase = input()
            #end on blank, write to file with >
            if phrase == "":
                break
            if phrase.startswith(">"):
                phrase = phrase[1:]
                write_file = True
            
            spoken_phrase = app.generate_audio(phrase)
            
            if write_file:
                app.output_audio(spoken_phrase, output_file="{0}.wav".format(phrase))
            else:
                app.output_audio(spoken_phrase)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    
    print("Goodbye")
    



if __name__== "__main__":
    main()