# a3

Approach: We first calculated the initial probabilities of the words, the transition probabilities and emission probabilities based on our training dataset. These probabilities are then passed on to our algorithms (i.e.. Simplified, VE, and MAP).
	Simplified uses the Naïve Bayes probability to predict the part of a speech of a given word. The important assumption made here is that the POS of any given word is independent on the other words in that sentence.
	For the Variable Elimination algorithm we’ve used the forward-backward approach. Wherein alpha is the probability of S2¬|S1, W2,W1 and beta is the probability of S2|S3, W2,W3 (From Fig 1b). 

	MAP uses 
