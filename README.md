# a3

Approach: We first calculated the initial probabilities, the transition probabilities and emission probabilities based on our training dataset. These probabilities are then passed on to our algorithms (i.e.. Simplified, VE, and MAP).

Please note our approaches for Simplified, Variable Elimination and Viterbi are same for both programs:
	For the Simplified algorithm we calculate the max of P(E|S)P(S) we calculate P(S) by mainitaining a table that would contain the count of each label while training
	For the Variable Elimination algorithm we’ve used the forward-backward approach. Wherein alpha is P(St,E1,E2,E3...Et) and beta is P(St|Et+1,Et+2,....ET). We calculated the state where the product of alpha and beta is the maximum.
	MAP uses 
	
For question 1 we are using the following definitions of emissions and transitions:
	1. Emision: The probability of word occuring given its label. P('The'|Noun) or P('The'|Verb) or P('The'|Determinant) and so on.
	2. Transition The probability of a label occuring given the previous lable. P(Noun|Verb) or P(Verb|Determinant) and so on.
