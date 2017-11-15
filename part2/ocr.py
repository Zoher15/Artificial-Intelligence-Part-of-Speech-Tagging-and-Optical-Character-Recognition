#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
# ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Ankita, Murtaza, Zoher 
# (based on skeleton code by D. Crandall, Oct 2017)
#
# Please check the readme file for the description
#
# Approach: We first calculated the initial probabilities, the transition probabilities and emission probabilities based on our training dataset. These probabilities are then passed on to our algorithms (i.e.. Simplified, VE, and MAP).
# Please note our approaches for Simplified, Variable Elimination and Viterbi ARE THE SAME for both programs: 
# For the Simplified algorithm we calculate the max of P(E|S)P(S) we calculate P(S) by mainitaining a table that would contain the count of each label while training.
# For the Variable Elimination algorithm we have used the forward-backward approach. Wherein alpha is P(St,E1,E2,E3...Et) and beta is P(St|Et+1,Et+2,....ET). 
# We calculated the state where the product of alpha and beta is the maximum. We used dynamic programming to maintain tables for both.
# For the Viterbi algorithm, we are calculating probability of being in a state j for a observed value and storing it in table to be used for next observed value probability calculation. 
#This serves as memoization table for us. Along with calculation we are also storing index of max value which gave the probability of the state in a array called result. This array is used 
#at the end to backtrack to find the final sequence.
# For question 2 we are using the following definitions of emissions and transitions:
#        1. Initial: The probability of a charcter occuring as first character of the sentense. We are using Laplacian smoothing to get a better probability distribution and non-zero probability for characters not present in training data at 1st position.
#        1. Emision: The probability of character occuring given an image. This is calculated by pixel matching of the matrix with the different matrices present in the training set. After experimentation and after findng better results, higher weight is assigned for matched pixels and less weight is assigned for matched spaces.
#        2. Transition: The probability of a character occuring given the previous character. P('a'|'b') or P('b'|'c') and so on. We are using Laplacian smoothing for transitional probability as well to get a better probability distribution and non-zero probability for character transactions that were not present in training data.
#
# Assumptions: 1. Training data is a Text representing large number of English words. 2. Test image will always include characters from that training dataset. 

from PIL import Image, ImageDraw, ImageFont
import math
import sys
from decimal import Decimal
import datetime

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

# Calculate sequence of characters using Naive Bayes Classsifier 
def simplified(e_prob, i_prob):
    result = " Simple: "
    for i in range(len(e_prob)):
        if (i == 0):
            each = [e_prob[i][j] * i_prob[j] for j in range(len(i_prob))] # initial and emission prob for 1st charcter
            result += TRAIN_LETTERS[each.index(max(each))]
        else:
            result += TRAIN_LETTERS[e_prob[i].index(max(e_prob[i]))] # only emission probabiity for other characters
    return result

# Check whether a character from traing text document is alphabet or not 
def isalphabet(ch):
    return (True if (not ch in "0123456789(),.-!?\"' ") else False)

# Calculate sequence of characters using Variable Elimination (Forward and backward algorithm)
def hmm_ve(characters, i_prob, t_prob, e_prob, f_prob):
    alphamat = [[0 for i in range(72)] for j in range(len(characters))]
    betamat = [[0 for i in range(72)] for j in range(len(characters))]
    # Calculate alpha from 1st to last character and store in array of size TRAINING_LETTERS * TEST_CHARCTERS
    for t in range(len(characters)): 
        for j in range(72):
            if t==0:
                alphamat[t][j]=e_prob[t][j]*i_prob[j] 
            else:
                alphamat[t][j]=e_prob[t][j]*sum([float(alphamat[t-1][i]*t_prob[i][j]) for i in range(72)])
    # Calculate beta from last to first character
    for t in range(len(characters)-1,-1,-1): 
        for j in range(72):
            if t==len(characters)-1:
                betamat[t][j] = f_prob[j]
            else:
                betamat[t][j]=sum([float(betamat[t+1][i]*e_prob[t+1][i]*t_prob[j][i]) for i in range(72)])
    result = [[0 for i in range(72)] for j in range(len(characters))]
    # Calculate final probability for each character as product of Alpha and Beta
    for i in range(len(alphamat)):
       for j in range(len(betamat[0])):
           result[i][j] = result[i][j] = float(alphamat[i][j] * betamat[i][j])      
    hmm = " HMM VE: "
    for i in range(len(result)):
        hmm += TRAIN_LETTERS[result[i].index(max(result[i]))]
    return hmm

# Calculate sequence of characters using Viterbi Algorithm (Dynamic Programming):    
def viterbi(len_test_let, i_prob, t_prob, e_prob):
    result = [[0 for j in range(len_test_let)] for i in range(len(TRAIN_LETTERS))]
    vtable = [[0 for i in range(len(TRAIN_LETTERS))] for j in range(len_test_let)]
    # Calculate values of Viterbi Table from 1st charcter to last character and then back to find the final sequence.
    for j in range(0, len_test_let):
        for i in range(len(TRAIN_LETTERS)):
            if (j == 0):
                vtable[j][i] = i_prob[i] * e_prob[j][i]
            else:
                cost = [vtable[j - 1][k] * t_prob[k][i] for k in range(len(TRAIN_LETTERS))]
                maxc = max(cost)
                vtable[j][i] = e_prob[j][i] * maxc
                result[i][j] = cost.index(maxc)
                
    string = "HMM MAP: "
    string2 = ""
    index = vtable[len_test_let - 1].index(max(vtable[len_test_let - 1])) 
    string2 += TRAIN_LETTERS[index]
    i = len_test_let - 1
    while (i > 0):
        index = result[index][i]
        string2 += TRAIN_LETTERS[index]
        i -= 1
    return string + string2[::-1]

# Load the images into bits format
def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

# Load the training image 
def load_training_letters(fname):
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

# Match the pixel values for Training image character to test image character and return the count 
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
# Find initial, transitional, emission and final probabilities
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
    # Count initial charcter occurances (For Both lower and upper case)
    if (line[0] in TRAIN_LETTERS):
        if (isalphabet(line[0])):
            initial_count[d[line[0]] - 26] += 1
        initial_count[d[line[0]]] += 1
    # Count final charcter occurances (for Both lower and upper case)
    if (line[-1] in TRAIN_LETTERS):
        if (isalphabet(line[-1])):
            final_count[d[line[-1]] - 26] += 1
        final_count[d[line[-1]]] += 1
    # Count transition from one charcter to the next character 
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

# Laplac Constant                   
k = 1
# Calculate the probabilities based on counts (Using laplacian smoothing to nullify biased training set)
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
        
# Print the final results
print (simplified(emission_prob, initial_prob))
print (hmm_ve(test_letters, initial_prob, transition_prob, emission_prob, final_prob))
print (viterbi(len(test_letters), initial_prob, transition_prob, emission_prob))
