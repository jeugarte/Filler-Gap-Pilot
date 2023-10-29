# Subj / Obj Expectations

- Do RNNs represent filler-gap licensing?
- Do RNNS show a pentalty for violating expectations about locations of gaps?
- Following Stowe (1986), Experiment 1.

Experiment: 2x2x2x3
* Wh-word is present
* Gap is present
* Locaiton of gap is in subject / direct object / PP or indirect object
* Presence of appositive at the start of the embedded clause

1. My brother knows that Ruth will bring us home to mom at Christmas
2. My brother knows who Ruth will bring us home to mom at Christmas
3. My brother knows that __ will bring us home to mom at Christmas
4. My brother knows who __ will bring us home to mom at Christmas

5. My brother knows that despite her poor health Ruth will bring us home to mom at Christmas
6. My brother knows who despite her poor health Ruth will bring us home to mom at Christmas
7. My brother knows that despite her poor health __ will bring us home to mom at Christmas
8. My brother knows who despite her poor health __ will bring us home to mom at Christmas

- These examples show the gap in subject position
- Appositive is added to reduce the local bigram effects of "that will" vs. "who will", which may impact surprisal of the target region.

Dependent Variables:
* (i) Total surprsial of embedded clause
* (ii) Surprisal of the word right after the target NP / gap (e.g. "will" in the examples below)
* (iii) Surprisal from the target NP / gap to the end of the sentence (e.g. "will ... Christmas" in the examples below)
* Surprisal will be taken using the Google 1-billion word benchmark trained LSTM and the Gulordava et. al. (2018) LSTM trained on 90-million tokens of wikipedia.

Predictions:
* If the RNN is learning basic filler/gap dependencies, we expect an interaction between wh-words and gaps, such that when just one of them is present the sentence is worse ((i)-(iii) are higher), but when both are present the sentence is better.
* If the RNN shows human-like behavior, then we expect the interaction to be the strongest for gaps in object position, followed by gaps in subject position, followed by gaps in PP position. 


