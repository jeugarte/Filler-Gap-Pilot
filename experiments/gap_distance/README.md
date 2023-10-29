# Are RNNs sensitive to distancve in filler-gap dependencies

- Do RNNs show filler/gap licensing effects (e.g. a gap must be licensed by a wh-word and a wh-word requires a gap)
- Do RNNs exhibit higher surprisal at gaps that are further from their fillers?

2x2x2x4
* Wh-word is present
* Gap is present
* Gap is in direct / indirect object position
* Modifier is: not present / short / medium / long

1. I know that your friend gave a baguette to Mary last weekend .
2. I know that your friend gave __ to Mary last weekend .
3. I know what your friend gave a baguette to Mary last weekend
4. I know what your friend gave __ to Mary last weekend .
5. I know that your friend in the straw hat and dark blue pants gave a baguette to Mary last weekend .
6. I know that your friend in the straw hat and dark blue pants gave __ to Mary last weekend .
7. I know what your friend in the straw hat and dark blue pants gave a baguette to Mary last weekend
8. I know what your friend in the straw hat and dark blue pants gave __ to Mary last weekend .
(Example given with a long modifier and direct object position)

Defenition of Length:
Short: 3-5 words
Medium: 6-8 words
Long: 8-12 words

- In addition to treating length as a categorical variable we will also treat the number of intervening words as a continuous variable

Dependent Variables:
* (i) Surprsial of the word following the gap (e.g. "to" for the first two examples)
* (ii) Surprisal of the phrase following the gap (e.g. "to Mary" for the first two examples)
* (iii) Surprisal from the gap to the end of the sentence (e.g. "to ... weekend ?")

Predictions:
* If the RNN is representing wh-gap dependencies, then we expect an interaction between the presence of a wh-word and a gap. The presence of a single gap or wh-word results in worse surprisal, but their combination results in lower surprisal in (i) - (iii).
* If the RNN is sensitive to length, then we expect an interaction between wh/gaps and length: the unlicenced gap conditions with longer modifiers will be penalized less than in the base case, ((6)-(5) < (2)-(1)) and, the effect of wh-licensors on gaps will be lower than in the base case ((8)-(7) > (4)-(3))