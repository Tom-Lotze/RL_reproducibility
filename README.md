# Reinforcement learning 2020 - University of Amsterdam
## Reproducibility Project

## Contributors:
- Samantha Biegel
- Berend Jansen
- Tom Lotze
- Eva Sevenster

## Notes Eva: 

Weighted importance SARSA: 
* W: for sarsa, should te w be p(s, a)p(s', a') / p(s, a)p(s', a') to get the relative probability of the whole sequence? 
* How to update if W = 0 and C = 0 (renturns nan)? I've taken it as 1 now 

The environment: 
* According to the book, formally the main difference between ordinary and weigthed importance sampling is in bias and variance. Moreover, it says ordinary sampling is rarely used because of the high variance. Was thinking that if we want to show varinace and bias differences, maybe we should choose an environment where the target and behaviour policy can be quite different (e.g. cliff world?). In the book, I think windy grid world is used mainly to show that on-policy sarsa solves something easy that off-policy and mc might struggle with, but not sure. 

