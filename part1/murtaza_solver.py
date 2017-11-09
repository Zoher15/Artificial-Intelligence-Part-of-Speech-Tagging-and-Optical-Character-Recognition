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
from label import read_data
#import copy

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#


initial_state_distribution=[0]*12


emission_count={}
#emission_type_count=[0]*12

type_of_words=['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']
transition_count=[[0 ]*12 for i in range(12)]
transition_probability=[[0]*12 for i in range(12)]
emission_probability={}
f = read_data("bc.test.tiny")
count=0

#for i in range(len(f)):
    #for j in range(len(f[i][0])):
        #emission_count[f[i][0][j]]=copy.deepcopy(emission_type_count)
        #emission_probability[f[i][0][j]]=copy.deepcopy(emission_type_count)

#The above 4 lines(Commented) and the below one line performs the same function
#emission_count=emission_probability={j:copy.deepcopy(emission_type_count) for i in f for j in i[0]}
emission_count=emission_probability={j:[0]*12 for i in f for j in i[0]}
#print emission_probability
#emission_probability= {j:copy.deepcopy(emission_type_count) for i in f for j in i[0]}
#print len(emission_probability)


def store_data():
    initial_count=[0]*12
    first_word_count=[0]*12
    total_num_words=0
    total_num_sentances=0

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
        # print emission_probability
        # print "_______________________________________________________"



    for i in range(12):
        #Initial Probability distribution
        initial_state_distribution[i]=float(first_word_count[i])/len(f)
        #Calculating transition probabilities
        for j in range(12):
            if(sum(transition_count[i])==0 or transition_count[i][j]==0):
                transition_probability[i][j]=0
            else:
                transition_probability[i][j]=float(transition_count[i][j])/sum(transition_count[i])


    #Implement the below lines using dictionary comperehension
    # for i in range(len(emission_probability.keys())):
    #     for j in range(12):
    #         if(initial_count[j]==0 or emission_count.values()[i][j]==0):
    #             emission_probability.values()[i][j]=0
    #         else:
    #             emission_probability.values()[i][j]=float(emission_count.values()[i][j])/initial_count[j]
                                                                                
    #print initial_state_distribution
    #print emission_probability
    #print emission_count

store_data()

def simplified(sentence):
    sentence=sentence.lower()
    sentence=sentence.split(" ")
    part_of_speech=[]
    for i in range(len(sentence)):
        most_probable_tag=[0]*12
        for j in range(12):
            most_probable_tag[j]= emission_probability[sentence[i]][j]*initial_state_distribution[j]
        part_of_speech.append(type_of_words[most_probable_tag.index(max(most_probable_tag))])
    # print part_of_speech

simplified("Desperately , Nick flashed one hand up , catching Poet's neck in the bend of his elbow .")

def hmm_ve(sentence):
    sentence=sentence.lower()
    sentence=sentence.split(" ")
    dp_table=[]
    temp=[]
    for i in range(12):
        temp.append(initial_state_distribution[i]*emission_probability[sentence[0]][i])
    dp_table.append(max(temp))
    temp=[]
    for i in range(1,len(sentence)):
        for j in range(12):
            for k in range(12):
                temp.append(dp_table[len(dp_table)-1]*transition_probability[j][k]*emission_probability[sentence[i]][k])

        dp_table.append(max(temp))
    print dp_table
    


hmm_ve("Desperately , Nick flashed one hand up , catching Poet's neck in the bend of his elbow .")


class Solver:

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling

    



    def posterior(self, sentence, label):
        return 0

    # Do the training!
    #
    def train(self, data):

        pass

    

    # Functions for each algorithm.
    #
    def simplified(self, sentence):

        print sentence


        return [ "noun" ] * len(sentence)

    def hmm_ve(self, sentence):
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

