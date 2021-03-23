from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from PIL import Image

import pandas as pd 
import numpy as np 

import os
from collections import defaultdict

DATA_PATH = os.path.join(os.getcwd(), 'data')


def get_parser():
    """parses terminal arguments"""

    parser = ArgumentParser(description='get parameters', 
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--input_path', metavar='noised_image_path.png', type=str, 
                        required=True, help='path to the noised image file')
    parser.add_argument('-n', '--noise', metavar=0.7, type=float, required=True,
                        help='noise parameter p of Bernoulli distribution')

    return parser


def get_bigram_log_prob(freq_bigrams_path: str) -> defaultdict:
    """
    makes conditional log probabilities from bigram frequencies
    takes str path to json file with bigrams' frequencies
    returns defaultdict with bigrams' conditional log probabilities
    """ 

    df = pd.read_json(freq_bigrams_path, typ='series')

    df = df.to_frame('freq').reset_index()
    df.rename(columns={'index': 'bigrams'}, inplace=True)

    df['alphabet'] = df['bigrams'].apply(lambda x: x[0])

    df = pd.merge(df.groupby(['alphabet']).sum(), df, on='alphabet', 
          suffixes=('_of_alphabet', '_of_bigram'))
  
    df['cond_log_prob'] = np.log(df['freq_of_bigram'] / df['freq_of_alphabet'])

    df = df.set_index('bigrams')
    final_dict = df['cond_log_prob'].to_dict()

    return defaultdict(lambda: -np.inf, final_dict)


def get_alphabet_images(alphabet_path: str) -> dict:
    """
    takes path to the folder with alphabet images
    returns dict with alphabet as keys and np images as values
    """
    
    result = {}

    for base in os.listdir(alphabet_path):
        image = Image.open(os.path.join(alphabet_path, base))
        np_image = np.array(image).astype(dtype=np.int64)
        result[base[:-4]] = np_image

    try:
        result[' '] = result['space']
        result.pop('space')
    except KeyError:
        print('!!!no "space" in the alphabet!!!')

    return result


def get_noisy_images(noisy_path: str) -> list:
    """
    divides the noisy image into (28, 28) shapes
    takes path to the noisy image
    returns array with parts of the divided noisy image a 
    """

    noised_image = Image.open(noisy_path)
    np_noised_image = np.array(noised_image).astype(dtype=np.int64)

    assert not np_noised_image.shape[1] % 28, "!!!the noisy image shape is wrong!!!"

    noised_images = np.split(np_noised_image, 
                             28 * np.arange(1, np_noised_image.shape[1] // 28), axis=1)
    
    return noised_images


def get_unary_constraints(alphabet_images: dict, noisy_images: list, noise: float) -> dict:
    """
    calculates the unary constraints
    takes alphabet and noisy images as np.ndarrays and noise - Bernoulli param
    returns a dict with alphabet as keys and 
    noisy_images' length constraints lists as values
    """
    assert 0 <= noise <= 1, "!!!noise should be in (0,1) range!!!"

    constraints = {}
    for letter in list(alphabet_images):

        letter_constraints = []
        for noisy_image in noisy_images:

            equality = (alphabet_images[letter] == noisy_image)
            non_equality = ~equality
            
            if noise and not noise == 1:
                letter_constraints.append(non_equality.sum() * np.log(noise) + 
                                          equality.sum() * np.log(1 - noise))
            elif noise:
                letter_constraints.append(non_equality.sum())
            else:
                letter_constraints.append(equality.sum())
            
        constraints[letter] = letter_constraints

    return constraints


def find_max_weights(probabilities: defaultdict, constraints: dict) -> dict:
    """
    main function for recognizing algorithm that fullfills the array with maximums
    takes log probabilities and unary constraints
    returns optimal structure - dict with alphabet as keys and lists of tuples 
    as values, where tuples are pairs of max weight and matching letter
    """

    optimal_structure = {} 
    for letter in list(constraints):
        optimal_structure[letter] = [(0, ' ')]

    for idx in range(len(list(constraints.values())[0])):

        for prev_letter in list(constraints):

            all_letter_weights = []
            for letter in list(constraints):
      
                all_letter_weights.append((probabilities[prev_letter + letter] + 
                                           constraints[letter][-1 - idx] + 
                                           optimal_structure[letter][idx][0], letter))
                
            max_weight = max(all_letter_weights, key=lambda x: x[0])
            optimal_structure[prev_letter].append(max_weight) 

    return optimal_structure


def get_recognized_text(structure: dict) -> str:
    """
    goes forward through the structure and collects best letters for recognition
    takes optimal structure from find_max_weights
    returns recognized string
    """

    recognized_text = letter = ' '
    for idx in range(len(list(structure.values())[0])):

        next_letter = structure[letter][-1 - idx][1]
        recognized_text += next_letter 
        letter = next_letter
    
    return recognized_text
    
    
if __name__ == '__main__':
    args = get_parser().parse_args()

    alphabet_images = get_alphabet_images(os.path.join(DATA_PATH, 'alphabet'))
    noisy_images = get_noisy_images(os.path.join(DATA_PATH, 'input', args.input_path))
    bigram_probs = get_bigram_log_prob(os.path.join(DATA_PATH, 'frequencies.json'))
    constraints = get_unary_constraints(alphabet_images, noisy_images, args.noise)

    dictionary_structure = find_max_weights(bigram_probs, constraints)
    result = get_recognized_text(dictionary_structure)

    print(f'\n"{result}"')