# a3

Approach: We first calculated the initial probabilities, the transition probabilities and emission probabilities based on our training dataset. To handle zero probabilities, we've smartly chosen a value of 0.01 which is assigned instead of 0. This value has been derived after performing a number of tests on different arbitrary values. It was found that replacing 0 by 0.01 in the probabilities gives us the most accurate answer. These probabilities are then passed on to our algorithms (i.e.. Simplified, VE, and MAP).

Please note our approaches for Simplified, Variable Elimination and Viterbi ARE THE SAME for both programs:
	For the Simplified algorithm we calculate the max of P(E|S)P(S) we calculate P(S) by mainitaining a table that would contain the count of each label while training
	For the Variable Elimination algorithm we’ve used the forward-backward approach. Wherein alpha is P(St,E1,E2,E3...Et) and beta is P(St|Et+1,Et+2,....ET). We calculated the state where the product of alpha and beta is the maximum.
	For the Viterbi algorithm, we are calculating probability of being in a state j for a observed value and storing it in table to be used for next observed value probability calculation. This serves as memoization table for us. Along with calculation we are also storing index of max value which gave the probability of the state in a array called result. This array is used at the end to backtrack to find the final sequence.	
For question 1 we are using the following definitions of emissions and transitions:
	1. Emision: The probability of word occuring given its label. P('The'|Noun) or P('The'|Verb) or P('The'|Determinant) and so on.
	2. Transition: The probability of a label occuring given the previous label. P(Noun|Verb) or P(Verb|Determinant) and so on.
Here, if our algorithm sees a word it has never observed before, we assign the highest probability to the unknown word being a noun. This word is then added to our training set and the next time this word is seen, our algorithm knows a way to deal with it.

For question 2 we are using the following definitions of emissions and transitions:
	1. Initial: The probability of a charcter occuring as first character of the sentense. We are using Laplacian smoothing to get a better probability distribution and non-zero probability for characters not present in training data at 1st position.
	1. Emision: The probability of character occuring given an image. This is calculated by pixel matching of the matrix with the different matrices present in the training set. After experimentation and after findng better results, higher weight is assigned for matched pixels and less weight is assigned for matched spaces.
	2. Transition: The probability of a character occuring given the previous character. P('a'|'b') or P('b'|'c') and so on. We are using Laplacian smoothing for transitional probability as well to get a better probability distribution and non-zero probability for character transactions that were not present in training data.
	
	
