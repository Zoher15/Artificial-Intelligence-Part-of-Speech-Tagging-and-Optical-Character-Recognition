#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Ankita, Murtaza, Zoher 
# (based on skeleton code by D. Crandall, Oct 2017)
#

from PIL import Image, ImageDraw, ImageFont
import math
import sys
from decimal import Decimal
import datetime

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

def simplified(e_prob, i_prob):
    result = " Simple: "
    for i in range(len(e_prob)):
        if (i == 0):
            each = [e_prob[i][j] * i_prob[j] for j in range(len(i_prob))]
            result += TRAIN_LETTERS[each.index(max(each))]
        else:
            result += TRAIN_LETTERS[e_prob[i].index(max(e_prob[i]))]
    return result

def isalphabet(ch):
    return (True if (not ch in "0123456789(),.-!?\"' ") else False)

def hmm_ve(characters, i_prob, t_prob, e_prob, f_prob):
    alphamat = [[0 for i in range(72)] for j in range(len(characters))]
    betamat = [[0 for i in range(72)] for j in range(len(characters))]
    for t in range(len(characters)):
        for j in range(72):
            if t==0:
                alphamat[t][j]=e_prob[t][j]*i_prob[j]
            else:
                alphamat[t][j]=e_prob[t][j]*sum([float(alphamat[t-1][i]*t_prob[i][j]) for i in range(72)])
    for t in range(len(characters)-1,-1,-1):
        for j in range(72):
            if t==len(characters)-1:
                betamat[t][j] = f_prob[j]
            else:
                betamat[t][j]=sum([float(betamat[t+1][i]*e_prob[t+1][i]*t_prob[j][i]) for i in range(72)])
    result = [[0 for i in range(72)] for j in range(len(characters))] 
    for i in range(len(alphamat)):
       for j in range(len(betamat[0])):
           result[i][j] = result[i][j] = float(alphamat[i][j] * betamat[i][j])      
    hmm = " HMM VE: "
    for i in range(len(result)):
        hmm += TRAIN_LETTERS[result[i].index(max(result[i]))]
    return hmm
    
def viterbi(len_test_let, i_prob, t_prob, e_prob):
    result = [[0 for j in range(len_test_let)] for i in range(len(TRAIN_LETTERS))]
    memo = [[0 for i in range(len(TRAIN_LETTERS))] for j in range(len_test_let)]
    for j in range(0, len_test_let):
        for i in range(len(TRAIN_LETTERS)):
            if (j == 0):
                memo[j][i] = i_prob[i] * e_prob[j][i]
            else:
                cost = [memo[j - 1][k] * t_prob[k][i] for k in range(len(TRAIN_LETTERS))]
                maxc = max(cost)
                memo[j][i] = e_prob[j][i] * maxc
                result[i][j] = cost.index(maxc)
                
    string = "HMM MAP: "
    string2 = ""
    idx = memo[len_test_let - 1].index(max(memo[len_test_let - 1])) 
    string2 += TRAIN_LETTERS[idx]
    i = len_test_let - 1
    while (i > 0):
        idx = result[idx][i]
        string2 += TRAIN_LETTERS[idx]
        i -= 1
    return string + string2[::-1]

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def match(train_frm, test_frm):
    count = 0
    for i in range(CHARACTER_HEIGHT):
        for j in range(CHARACTER_WIDTH):
            if (train_frm[i][j] == '*' and test_frm[i][j] == '*'):
               count += 20
            elif (train_frm[i][j] == ' ' and test_frm[i][j] == ' '):
               count += 1
            elif (train_frm[i][j] == ' ' and test_frm[i][j] == '*'):
                count -= 20
            else:
                count -= 30
    return count

#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
#
# Find initial, transitional and emission probabilities
initial_prob = [0 for i in range(len(TRAIN_LETTERS))]
initial_count = [0 for i in range(len(TRAIN_LETTERS))]
transition_prob = [[0 for i in range(len(TRAIN_LETTERS))] for j in range(len(TRAIN_LETTERS))]
transition_count = [[0 for i in range(len(TRAIN_LETTERS))] for j in range(len(TRAIN_LETTERS))]
emission_prob = [[0.0 for i in range(len(train_letters))] for j in range(len(test_letters))]
final_prob = [0 for i in range(len(TRAIN_LETTERS))]
final_count = [0 for i in range(len(TRAIN_LETTERS))]

# Dictionary for the TRAIN_LETTERS index
d = {}
i = 0
for ch in TRAIN_LETTERS:
    d[ch] = i
    i += 1

# Calculate initial and transitional probabilities   
numberofline = 0
ifile = open(train_txt_fname, "r")
for line in ifile:
    numberofline += 1
    line = line.lower()
    if (line[0] in TRAIN_LETTERS):
        if (isalphabet(line[0])):
            initial_count[d[line[0]] - 26] += 1
        initial_count[d[line[0]]] += 1
    if (line[-1] in TRAIN_LETTERS):
        if (isalphabet(line[-1])):
            final_count[d[line[-1]] - 26] += 1
        final_count[d[line[-1]]] += 1
    for i in range(len(line) - 1):
        j = i + 1
        if (line[i] in TRAIN_LETTERS and line[j] in TRAIN_LETTERS):
            transition_count[d[line[i]]][d[line[j]]] +=1
            if (isalphabet(line[i])):
                if (isalphabet(line[j])):
                    transition_count[d[line[i]] - 26][d[line[j]]] +=1
                    transition_count[d[line[i]] - 26][d[line[j]] - 26] +=1
                else:
                    transition_count[d[line[i]] - 26][d[line[j]]] +=1                      
            elif (isalphabet(line[j])):
                    transition_count[d[line[i]]][d[line[j]] - 26] +=1

k = 1                 
for i in range(len(TRAIN_LETTERS)):
    initial_prob[i] = float(initial_count[i] + k) / (numberofline + (72 * k))
    final_prob[i] = float(final_prob[i] + k) / (numberofline + (72 * k))
    s = sum(transition_count[i])
    for j in range(len(TRAIN_LETTERS)):
        transition_prob[i][j] = float(transition_count[i][j] + 1) / (s + (72 * k))
        
# Calculate emission probabilities            
num_pixels = CHARACTER_HEIGHT * CHARACTER_WIDTH

for i in range(len(test_letters)):
    j = 0
    for ch in TRAIN_LETTERS:
        emission_prob[i][j] = float(match(train_letters[ch], test_letters[i])) / (num_pixels)
        if (emission_prob[i][j] <= 0):
            emission_prob[i][j] = float(1)/100
        j += 1

print (simplified(emission_prob, initial_prob))
print (hmm_ve(test_letters, initial_prob, transition_prob, emission_prob, final_prob))
print (viterbi(len(test_letters), initial_prob, transition_prob, emission_prob))



