# Part 1: Part of Speech (POS) Tagging
Natural language processing (NLP) is an important research area in artificial intelligence, dating back to at least the 1950’s. One of the most basic problems in NLP is part-of-speech tagging, in which the goal is to mark every word in a sentence with its part of speech (noun, verb, adjective, etc.). This is a first step towards extracting semantics from natural language text. For example, consider the following sentence:
Her position covers a number of daily tasks common to any social director.
Part-of-speech tagging here is not easy because many of these words can take on different parts of speech depending on context. For example, position can be a noun (as in the above sentence) or a verb (as in “They position themselves near the exit”). In fact, covers, number, and tasks can all be used as either nouns or verbs, while social and common can be nouns or adjectives, and daily can be an adjective, noun, or adverb. The correct labeling for the above sentence is:
	Her	position covers	 a   number  of  daily tasks  common  to   any  social  director.
	DET	NOUN	  VERB  DET  NOUN   ADP   ADJ  NOUN    ADJ    ADP  DET   ADJ     NOUN
where DET stands for a determiner, ADP is an adposition, ADJ is an adjective, and ADV is an adverb. Labeling parts of speech thus involves an understanding of the intended meaning of the words in the sentence, as well as the relationships between the words.

Supervised by **David Crandall** Code by **Murtaza Khambaty** Assisted by **Zoher Kachwala** **Ankita Alshi**

# Part 2: Optical Character Recognition (OCR)
Modern OCR is very good at recognizing documents, but rather poor when recognizing isolated characters. It turns out that the main reason for OCR’s success is that there’s a strong language model: the algorithm can resolve ambiguities in recognition by using statistical constraints of English (or whichever language is being processed). These constraints can be incorporated very naturally using an HMM.
Let’s say we’ve already divided a text string image up into little subimages corresponding to individual letters; a real OCR system has to do this letter segmentation automatically, but here we’ll assume a fixed-width font so that we know exactly where each letter begins and ends ahead of time. In particular, we’ll assume each letter fits in a box that’s 16 pixels wide and 25 pixels tall. We’ll also assume that our documents only have the 26 uppercase latin characters, the 26 lowercase characters, the 10 digits, spaces, and 7 punctuation symbols, (),.-!?’". Suppose we’re trying to recognize a text string with n characters, so we have n observed variables (the subimage corresponding to each letter) O1,...,On and n hidden variables, l1...,ln, which are the letters we want to recognize. We’re thus interested in P(l1,...,ln|O1,...,On). As in part 1, we can rewrite this using Bayes’ Law, estimate P(Oi|li) and P(li|li−1) from training data, then use probabilistic inference to estimate the posterior, in order to recognize letters.

Supervised by **David Crandall** Code by **Ankita Alshi** Assisted by **Zoher Kachwala** **Murtaza Khambaty**

# Team Approach

We first calculated the initial probabilities, the transition probabilities and emission probabilities based on our training dataset. To handle zero probabilities, we've smartly chosen a value of 0.01 which is assigned instead of 0. This value has been derived after performing a number of tests on different arbitrary values. It was found that replacing 0 by 0.01 in the probabilities gives us the most accurate answer. These probabilities are then passed on to our algorithms (i.e.. Simplified, VE, and MAP).

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
	1. Emision: The probability of character occuring given an image. This is calculated by pixel matching of the matrix with the different matrices present in the training set. We have set the probability of a noisy pixel as 0.3. Probability (observed pixel = "*" | training image pixel = "*") = 0.7 and Probability (observed pixel = " " | training image pixel = "*") = 0.3
	2. Transition: The probability of a character occuring given the previous character. P('a'|'b') or P('b'|'c') and so on. We are using Laplacian smoothing for transitional probability as well to get a better probability distribution and non-zero probability for character transactions that were not present in training data.
	
	
