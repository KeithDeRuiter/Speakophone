# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:15:34 2019

@author: Keith
"""

import json
from random import randint
from random import choice
from Speakophone import Speakophone
import numpy as np
import scipy.io.wavfile as wv
import os
from pathlib import Path
    
ones = {
    0: '', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
    7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve',
    13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',
    17: 'seventeen', 18: 'eighteen', 19: 'nineteen'}
tens = {
    2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty', 6: 'sixty',
    7: 'seventy', 8: 'eighty', 9: 'ninety'}
illions = {
    1: 'thousand', 2: 'million', 3: 'billion', 4: 'trillion', 5: 'quadrillion',
    6: 'quintillion', 7: 'sextillion', 8: 'septillion', 9: 'octillion',
    10: 'nonillion', 11: 'decillion'}


def say_number(i):
    """
    Convert an integer into its word representation.
    e.g. 1 becomes "one"
    Supports all numbers up to "-decillions" as well as negatives.

    Sourced from https://stackoverflow.com/questions/19504350/how-to-convert-numbers-to-words-in-python/42555145

    Args:
        i (int): The integer to convert to words.

    Returns:
        (string) The string representing the words for the number passed in.

    """
    if i < 0:
        return _join('negative', _say_number_pos(-i))
    if i == 0:
        return 'zero'
    return _say_number_pos(i)


def _say_number_pos(i):
    if i < 20:
        return ones[i]
    if i < 100:
        return _join(tens[i // 10], ones[i % 10])
    if i < 1000:
        return _divide(i, 100, 'hundred')
    for illions_number, illions_name in illions.items():
        if i < 1000**(illions_number + 1):
            break
    return _divide(i, 1000**illions_number, illions_name)


def _divide(dividend, divisor, magnitude):
    return _join(
        _say_number_pos(dividend // divisor),
        magnitude,
        _say_number_pos(dividend % divisor),
    )
    
    
def _join(*args):
    return ' '.join(filter(bool, args))
    
    
def roll_dice(num_dice=1, dice_size=6):
    """
    Rolls the specified number of n-sided dice.
    
    Args:
        num_dice (int): The number of dice to roll, defualts to 1.  If 
        num_dice is less than 1, the value is clamped to 1.
        
        dice_size (int): The number of sides on the die being rolled, defaults 
        to 6.  If dice_size is less than 1, the value is clamped to 1.
        
    Returns:
        results (list of int): A list containing all of the results.
    """
    
    results = list()
    dice_size = max(1, dice_size)
    for i in range(0,num_dice):
        results.append(randint(1, dice_size))
    return results


def generate_dice_audio_samples():
    """
    Uses the Speakophone to generate the most basic set of audio for the dice 
    roller sound samples.  These are written to the current directory and 
    named based on the text they say.
    
    """
    
    samp_dir = "../samples/little-scale_SP0256-AL2"
    dict_file = "../samples/CMU-SphinxDict/cmudict_SPHINX_40.txt"
    map_file = "../samples/CMU-SphinxDict/SphinxPhones_40__C64_mapping.txt"
    
    app = Speakophone(sample_dir=samp_dir,
                      dict_file_path=dict_file,
                      allo_map_file_path=map_file)
    
    basic_dice_phrases = [
            "You rolled",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
            "d",
            "and got",
            "and",
            "have a nice day"
            ]
    
    for phr in basic_dice_phrases:
        app.speak_audio(app.generate_audio(phr), output_file=phr)
    

class DiceRoller:
    
    """
    The DiceRoller reads in samples of audio in order to stitch together the 
    phrases required to read out the result of a dice roll.  Which samples it 
    uses to read the results is configurable based on the file passed into 
    __init__ on construction.  This file should be a json file with entries for 
    the overall sample directory, as well as the directories for each of the 
    required phrase components.  Example provided below:
        
    {
        "sample_directory": "../Samples/DiceRoller/Roboid",
        "intro_phrases": "1_Intros",
        "number_phrases": "2_Numbers",
        "d_phrases": "3_Dees",
        "segue_phrases": "4_Segues",
        "joining_phrases": "5_Joiners", 
        "outro_phrases": "6_Outros"
    }
    
    Each provided directory should be located in the sample_directory and
    contain ".wav" files suitable to be randomly selected for that purpose.
    These will all be read in and randomly selected from to assemble the
    proper audio responses for rolls.
    
    Phrases are stitched together as follows:
    
        Intro | num_dice | "Dee" | dice_size | Segue | [Result | "And"] | Outro
        With as many Result/And groups  as num_dice rolled.
    
    
    Args:
        config_file: The path to the configuration file described above.
        
        
    Attributes:
        
        intro_phrases: A list of NumPy arrays for the samples of intro-suitable
        wav files
        
        number_phrases: A dict of NumPy arrays for the number wav files, keyed
        off of the textual form of the word for that number (which should 
        match the filename)
    
        d_phrases: A list of NumPy arrays for the "d" wav files e.g. 2 "d" 6
        
        segue_phrases: A list of NumPy arrays for segue wav files
        
        joining_phrases: A list of NumPy arrays for joining phrase wav files
        such as "and" for between multiple results

        outro_phrases: A list of NumPy arrays for the samples of outro-suitable
        wav files
    
    """
    
    def __init__(self, config_file):
        with open(config_file, 'r') as phrase_config_file:
            phrase_config = json.load(phrase_config_file)
        
        sample_dir = phrase_config["sample_directory"]
        
        self.intro_phrases = self.load_generic_wavs(os.path.join(sample_dir, phrase_config["intro_phrases"]))
        self.number_phrases = self.load_number_wavs(os.path.join(sample_dir, phrase_config["number_phrases"]))
        self.d_phrases = self.load_generic_wavs(os.path.join(sample_dir, phrase_config["d_phrases"]))
        self.segue_phrases = self.load_generic_wavs(os.path.join(sample_dir, phrase_config["segue_phrases"]))
        self.joining_phrases = self.load_generic_wavs(os.path.join(sample_dir, phrase_config["joining_phrases"]))
        self.outro_phrases = self.load_generic_wavs(os.path.join(sample_dir, phrase_config["outro_phrases"]))
        
        
    def load_generic_wavs(self, directory):
        """
        Loads all "*.wav" in the provided directory.
        
        Args:
            directory (string): The directory to search for .wav files.
            
        Returns:
            samples: A list of NumPy arrays of each file's audio samples.
        """
        
        samples = list()
        for file in os.listdir(directory):
            if file.endswith(".wav"):
                joined_path = os.path.join(directory, file)
                fs, wav_array = wv.read(joined_path)
                print("Loaded {0} with {1} samples at {2}".format(joined_path, len(wav_array), fs))
                samples.append(wav_array)
        return samples
    
    
    def load_number_wavs(self, directory):
        """
        Loads in the "*.wav" files for the numbers read by the roller.
        The samples are put in a dict keyed by the filename (minus ".wav").
        This loads all files in the provided directory.
        
        Args:
            directory (string): The directory to search for .wav files.
            
        Returns:
            samples: A dict keyed by the strings that were the filenames (minus .wav)
            with values that are the NumPy array of the audio samples.
        """
        
        samples = dict()
        for file in os.listdir(directory):
            if file.endswith(".wav"):
                joined_path = os.path.join(directory, file)
                p = Path(joined_path)
                number_name = p.resolve().stem.strip()
                fs, wav_array = wv.read(joined_path)
                print("Loaded {0} with {1} samples at {2}".format(joined_path, len(wav_array), fs))
                samples[number_name] = wav_array
        return samples
    
    
    def generate_roll_audio(self, num_dice=1, dice_size=6):
        """
        Generates stitched-together audio based on the samples read in by the
        Dice Roller to read out the provided dice roll's results.
        
        Args:
            num_dice (int): The number of dice to roll, defualts to 1.  Must 
            not be less than 1.
            
            dice_size (int): The number of sides on the die being rolled, 
            defaults to 6.  Must not be less than 1.
            
        Raises:
            ValueError: if num_dice or dice_size are less than 1.
            
        Returns:
            audio: A NumPy array that contains the audio samples of the
            resultant stitched-together audio.
            
        """
        
        if num_dice < 1:
            raise ValueError("num_dice cannot be less than 1")
        if dice_size < 1:
            raise ValueError("dice_size cannot be less than 1")
            
        
        roll_results = roll_dice(num_dice, dice_size)
        
        results = list()
        results.append(choice(self.intro_phrases))
        results.append(self.number_phrases[say_number(num_dice)])
        results.append(choice(self.d_phrases))
        results.append(self.number_phrases[say_number(dice_size)])
        results.append(choice(self.segue_phrases))
        
        for r in roll_results:
            txt = say_number(r)
            results.append(self.number_phrases[txt])
            results.append(choice(self.joining_phrases))
        results = results[:-1]
        
        results.append(choice(self.outro_phrases))        
        
        return np.hstack(results)



if __name__== "__main__":
    samp_dir = "../samples/little-scale_SP0256-AL2"
    dict_file = "../samples/CMU-SphinxDict/cmudict_SPHINX_40.txt"
    map_file = "../samples/CMU-SphinxDict/SphinxPhones_40__C64_mapping.txt"
    
    app = Speakophone(sample_dir=samp_dir,
                      dict_file_path=dict_file,
                      allo_map_file_path=map_file)
    
    num = 1
    size = 6
    
    response_list = list()
    response_list.append("You rolled {0} d {1} and got ".format(say_number(num), say_number(size)))
    for n in roll_dice(num,size):
        response_list.append(say_number(n))
        response_list.append(" and ")
    response_list.append("that is all")
    
    #app.say(''.join(response_list))
    
    
    
    roller = DiceRoller("../samples/DiceRoller/dice_roller_phrases.json")
    #a = roller.generate_roll_audio(randint(1, 4), choice([2,4,6,8,10,12,20,100]))
    a = roller.generate_roll_audio(2, 20)
    app.output_audio(a)
    #app.output_audio(a, "your_roll")
