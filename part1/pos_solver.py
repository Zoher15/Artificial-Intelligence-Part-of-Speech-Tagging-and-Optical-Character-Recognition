###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids: Murtaza Khambaty - mkhambat, Ankita Alshi - aralshi, Zoher Kachwala - zkachwal
#
# (Based on skeleton code by D. Crandall)
#
#
####
'''
Approach: We first calculated the initial probabilities, the transition probabilities and emission probabilities based on our 
training dataset. These probabilities are then passed on to our algorithms (i.e.. Simplified, VE, and MAP).

Please note our approaches for Simplified, Variable Elimination and Viterbi ARE THE SAME for both programs: 
For the Simplified algorithm we calculate the max of P(E|S)P(S) we calculate P(S) by mainitaining a table that would contain the count of each label while training.
For the Variable Elimination algorithm we have used the forward-backward approach. Wherein alpha is P(St,E1,E2,E3...Et) and beta is P(St|Et+1,Et+2,....ET). 
We calculated the state where the product of alpha and beta is the maximum. We used dynamic programming to maintain tables for both.
For the Viterbi algorithm 

For question 1 we are using the following definitions of emissions and transitions: 
1. Emision: The probability of word occuring given its label. P('The'|Noun) or P('The'|Verb) or P('The'|Determinant) and so on. 
2. Transition: The probability of a label occuring given the previous label. P(Noun|Verb) or P(Verb|Determinant) and so on.

'''
####

import random
import math
import operator

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

class Solver:

    initial_state_distribution=[0]*12
    final_state_distribution=[0]*12
    emission_count={}
    type_of_words=['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']
    transition_count=[[0 ]*12 for i in range(12)]
    transition_probability=[[0]*12 for i in range(12)]
    emission_probability={}
    count=float(1)/100

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    
    def posterior(self, sentence, label):
        posterior_log=[]
        
        for i in range(len(sentence)):
            posterior_log.append(math.log(self.emission_probability[sentence[i]][Solver.type_of_words.index(label[i])]*self.prior_probability[Solver.type_of_words.index(label[i])],2))            

        return sum(posterior_log)

    # Do the training!
    #
    def train(self, f):
        initial_count=[0]*12
        first_word_count=[0]*12
        last_word_count=[0]*12
        self.total_num_words=0
        total_num_sentances=0
        self.prior_count=[]

        self.emission_count=self.emission_probability={j:[Solver.count]*12 for i in f for j in i[0]}

        for i in range(len(f)):
            self.total_num_words+=len(f[i][0])
            total_num_sentances+=1
            #Counts used for calculating initial state distribution
            first_word_count[Solver.type_of_words.index(f[i][1][0])]+=1
            last_word_count[Solver.type_of_words.index(f[i][1][-1])]+=1
            for j in range(len(f[i][1])):
                index=Solver.type_of_words.index(f[i][1][j])
                initial_count[index]+=1
                #Counts used for calculating transition probability
                if(j<len(f[i][1])-1):  
                    Solver.transition_count[index][Solver.type_of_words.index(f[i][1][j+1])]+=1

                #Counts used for calculating emission probability
                self.emission_count[f[i][0][j]][index]+=1

        for i in range(12):
            #Initial Probability distribution
            # self.prior_count.append(sum(Solver.transition_count[i]))

            if(first_word_count[i]==0):
                Solver.initial_state_distribution[i]=Solver.count
            else:
                Solver.initial_state_distribution[i]=float(first_word_count[i])/len(f)

            if(last_word_count[i]==0):
                Solver.final_state_distribution[i]=Solver.count
            else:
                Solver.final_state_distribution[i]=float(last_word_count[i]/len(f))
            #Calculating transition probabilities
            for j in range(12):
                if(sum(Solver.transition_count[i])==0 or Solver.transition_count[i][j]==0):
                    Solver.transition_probability[i][j]=Solver.count
                else:
                    Solver.transition_probability[i][j]=float(Solver.transition_count[i][j])/sum(Solver.transition_count[i])

        # print len(f[0][1])
        self.prior_probability=[]
        for i in range(12):
            self.prior_probability.append(float(initial_count[i])/self.total_num_words)
        # print sum(self.prior_probability)

        for i in self.emission_probability:
            for j in range(12):
                if(initial_count[j]==0 or self.emission_count[i][j]==0):
                    self.emission_probability[i][j]=Solver.count
                else:
                    self.emission_probability[i][j]=float(self.emission_count[i][j])/initial_count[j]

        
    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        part_of_speech=[]
        for i in range(len(sentence)):
            self.most_probable_tag=[0]*12
            if(sentence[i] not in self.emission_probability):
                #If word not seen before, provide high probability that, that word is a noun.
                self.emission_probability[sentence[i]]=[Solver.count,Solver.count,Solver.count,Solver.count,Solver.count,1-(Solver.count)*11,Solver.count,Solver.count,Solver.count,Solver.count,Solver.count,Solver.count]
            for j in range(12):
                self.most_probable_tag[j]= self.emission_probability[sentence[i]][j]*self.prior_probability[j]
                # Solver.initial_state_distribution[j]
                
            part_of_speech.append(Solver.type_of_words[self.most_probable_tag.index(max(self.most_probable_tag))])

        return part_of_speech
        # return [ "noun" ] * len(sentence)

    def hmm_ve(self, sentence):
        #alphamat calculates and stores the forward probabilities
        alphamat=[[0 for i in range(12)] for j in range(len(sentence))]
        #betamat calculates and stores the backward probabilities
        betamat=[[0 for i in range (12)] for j in range(len(sentence))]
        for t in range(len(sentence)):
            for j in range(12):
                if t==0:
                    alphamat[t][j]=self.emission_probability[sentence[t]][j]*Solver.initial_state_distribution[j]
                else:
                    alphamat[t][j]=self.emission_probability[sentence[t]][j]*sum([float(alphamat[t-1][i]*Solver.transition_probability[i][j]) for i in range(12)])
        
        for t in range(len(sentence)-1,-1,-1):
            
            for j in range(12):
                if t==len(sentence)-1:
                    betamat[t][j]=1
                else:
                    betamat[t][j]=sum([float(betamat[t+1][i]*self.emission_probability[sentence[t+1]][i]*Solver.transition_probability[j][i]) for i in range(12)])
        part_of_speech_ve=[]
        vestates=[]
        posterior_log=[]
        vestates_with_log=[[max([[j,float(alphamat[t][j]*betamat[t][j])] for j in range(12)],key=operator.itemgetter(1))] for t in range(len(sentence))]
                
        for i in range(len(vestates_with_log)):
            vestates.append(vestates_with_log[i][0][0])
            posterior_log.append(vestates_with_log[i][0][1])

        # print posterior_log

        for i in range(len(vestates)):
            part_of_speech_ve.append(Solver.type_of_words[vestates[i]])
        return part_of_speech_ve
        
        # return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):

        T=len(sentence)
        result = [[0 for t in range(T)]for j in range(12)]
        memo = [[0 for i in range(12)]for t in range(T)]
        for t in range(T):
            for j in range(12):
                if t==0:
                    memo[t][j] = Solver.initial_state_distribution[j] * self.emission_probability[sentence[t]][j]
                else:
                    cost = [memo[t - 1][i]*Solver.transition_probability[i][j] for i in range(12)]
                    maxc = max(cost)
                    memo[t][j] = self.emission_probability[sentence[t]][j] * maxc
                    result[j][t] = cost.index(maxc)              
        string2 = []
        idx = memo[T - 1].index(max(memo[T - 1])) 
        string2.append(Solver.type_of_words[idx])
        i = len(sentence) - 1
        
        while (i > 0):
            idx = result[idx][i]
            string2.append(Solver.type_of_words[idx])
            i -= 1
        return string2[::-1]

        # return [ "noun" ] * len(sentence)


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

