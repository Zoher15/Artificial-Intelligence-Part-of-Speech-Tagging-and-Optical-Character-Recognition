###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids: Murtaza Khambaty - mkhambat@iu.edu
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
import math
import operator
from label import read_data
#import copy

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

initial_state_distribution=[0]*12
emission_count={}
type_of_words=['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']
transition_count=[[0 ]*12 for i in range(12)]
transition_probability=[[0]*12 for i in range(12)]
emission_probability={}
f = read_data("bc.train.txt")
count=float(1)/10000000
emission_count=emission_probability={j:[count]*12 for i in f for j in i[0]}




class Solver:

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    
    # store_data()
    # simplified("Desperately , Nick flashed one hand up , catching Poet's neck in the bend of his elbow .")
    # hmm_ve("Desperately , Nick flashed one hand up , catching Poet's neck in the bend of his elbow .")
    
    def posterior(self, sentence, label):
        return 0

    # Do the training!
    #
    def train(self, f):
        initial_count=[0]*12
        first_word_count=[0]*12
        total_num_words=0
        total_num_sentances=0
        count =0

        for i in range(len(f)):
            total_num_words+=len(f[i][0])
            total_num_sentances+=1
            #Counts used for calculating initial state distribution
            first_word_count[type_of_words.index(f[i][1][0])]+=1
            for j in range(len(f[i][1])):
                index=type_of_words.index(f[i][1][j])
                initial_count[index]+=1
                #Counts used for calculating transition probability
                if(j<len(f[i][1])-1):  
                    transition_count[index][type_of_words.index(f[i][1][j+1])]+=1

                #Counts used for calculating emission probability
                emission_count[f[i][0][j]][index]+=1
                #  implement the below line to make it more efficient with a little less accuracy
                
                emission_probability[f[i][0][j]][index]=float(emission_count[f[i][0][j]][index])/initial_count[index]
                # print emission_probability[f[i][0][j]][index]

        # print emission_probability
        for i in range(12):
            #Initial Probability distribution
            if(first_word_count[i]==0):
                initial_state_distribution[i]=count
            else:
                initial_state_distribution[i]=float(first_word_count[i])/len(f)
            #Calculating transition probabilities
            for j in range(12):
                if(sum(transition_count[i])==0 or transition_count[i][j]==0):
                    transition_probability[i][j]=count
                else:
                    transition_probability[i][j]=float(transition_count[i][j])/sum(transition_count[i])
        # Implement the below lines using dictionary comperehension
        # for i in range(len(emission_probability.keys())):
        #     for j in range(12):
        #         if(initial_count[j]==0 or emission_count.values()[i][j]==0):
        #             emission_probability.values()[i][j]=float(1)/10000
        #         else:
        #             emission_probability.values()[i][j]=float(emission_count.values()[i][j])/initial_count[j]

        # print transition_probability
        # print emission_probability
        # print emission_count
    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        sentence=sentence.lower()
        sentence=sentence.split(" ")
        part_of_speech=[]
        for i in range(len(sentence)):
            most_probable_tag=[0]*12
            for j in range(12):
                most_probable_tag[j]= emission_probability[sentence[i]][j]*initial_state_distribution[j]
            part_of_speech.append(type_of_words[most_probable_tag.index(max(most_probable_tag))])
        print part_of_speech
        print sentence
        return [ "noun" ] * len(sentence)

    def hmm_ve(self, sentence):
        sentence=sentence.lower()
        sentence=sentence.split(" ")
        alphamat=[[0 for i in range(12)] for j in range(len(sentence))]
        betamat=[[0 for i in range (12)] for j in range(len(sentence))]
        for t in range(len(sentence)):
            for j in range(12):
                if t==0:
                    alphamat[t][j]=emission_probability[sentence[t]][j]*initial_state_distribution[j]
                else:
                    alphamat[t][j]=emission_probability[sentence[t]][j]*sum([float(alphamat[t-1][i]*transition_probability[i][j]) for i in range(12)])
        for t in range(len(sentence)-1,-1,-1):
            for j in range(12):
                if t==len(sentence)-1:
                    betamat[t][j]=1
                else:
                    betamat[t][j]=sum([float(betamat[t+1][i]*emission_probability[sentence[t+1]][i]*transition_probability[i][j]) for i in range(12)])
        #print alphamat
        #print betamat
        correct_answer=['adv','.','noun','verb','num','noun','prt','.','verb','noun','noun','adp','det','noun','adp','det','noun','.']
        part_of_speech_ve=[]
        vestates=[[max([[j,float(alphamat[t][j]*betamat[t][j])] for j in range(12)],key=operator.itemgetter(1))[0]] for t in range(len(sentence))]
        for i in range(len(vestates)):
            part_of_speech_ve.append(type_of_words[vestates[i][0]])
        print part_of_speech_ve
        return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):
        return [ "noun" ] * len(sentence)


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"

