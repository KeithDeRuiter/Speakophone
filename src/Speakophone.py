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
from scipy.signal import butter, lfilter, freqz
import scipy.io.wavfile as wv
from pathlib import Path
import os
import re



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def trim_silence(audio, threshold=200):
    if audio.max() < threshold:
        print("Trimming everything below threshold, returning empty")
        return np.empty(1)
    
    #Get index of first/last value larger than 200
    transient_start_index = np.where(abs(audio) > threshold)[0][0]
    transient_stop_index = np.where(abs(audio) > threshold)[0][-1]
    print("trimming to: " + str(transient_start_index) + ":" + str(transient_stop_index))
    
    return audio[transient_start_index:transient_stop_index]


def load_samples(directory):
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
            
            #Hack to make "AND" sound proper TODO remove and edit audio, or accept my fate
            if allo_name == "AX":
                allophones[allo_name] = np.hstack([wav_array, wav_array])
            elif allo_name.startswith("DD"):
                allophones[allo_name] = np.hstack([wav_array, wav_array[2500:]])
            
    return allophones


def load_cmu_dict(dict_file_path):
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
        devices = sd.query_devices()
        print(devices)    
        print()
        print(str(len(self.sounds)) + " sounds found")
        
        #Test writing
        for sk, sv in self.sounds.items():
            n = sk + "_TESTOUT.wav"
            wv.write(n, 44100, sv)
        
        
    def generate_audio(self, phrase):
        phrase = phrase.strip().upper()
        phrase = re.sub('[^A-Z \']+', '', phrase)
        words = phrase.split(" ")
        clips = list()

        for w in words:
            print("Saying: {0}".format(w))
            for phone in cmu_dict[w].split(" "):
                print("DictPhone: {0}\tMapPhone: {1}".format(phone, str(allo_map[phone])))
                s = sounds[allo_map[phone]]
                clips.append(s)
            print("Word audio length: {}".format(len(s)))
            clips.append(self.interword_pad)

        #Combine audio from words/sounds together
        final_audio = np.hstack(clips)
        print("Total audio length: {}".format(len(final_audio)))
        return final_audio
    
    
    def speak_audio(self, audio, output_file=None):
        if output_file is not None:
            out_filename = output_file + ".wav"
            wv.write(out_filename, 44100, audio)
        else:
            sd.play(audio)

    def say(self, phrase):
        self.speak_audio(self.generate_audio(phrase))
        
    
def main():
    print("Running Speakophone")
  
    #di_path = os.path.join(directory, dict_filename)
    
    samp_dir = "../samples/little-scale_SP0256-AL2"
    dict_file = "../samples/CMU-SphinxDict/cmudict_SPHINX_40.txt"
    map_file = "../samples/CMU-SphinxDict/SphinxPhones_40__C64_mapping.txt"
    
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
                app.speak_audio(spoken_phrase, output_file=phrase)
            else:
                app.speak_audio(spoken_phrase)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    
    print("Goodbye")
    
        


sounds = load_samples("../samples/little-scale_SP0256-AL2")
cmu_dict = load_cmu_dict("../samples/CMU-SphinxDict/cmudict_SPHINX_40.txt")
allo_map = load_allophone_map("../samples/CMU-SphinxDict/SphinxPhones_40__C64_mapping.txt")


if __name__== "__main__":
    main()