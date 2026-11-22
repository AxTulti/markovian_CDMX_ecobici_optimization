# Notes on the Markov birth and death model calculations
N(t) has less arrivals on max capacity with A group stations.
The tests work better on A group stations (the ones that usually fail are the same: batch arrivals and no state dependence)

## For which group to optimize model
A: Statistically more stable and well behaved 

B: More useful to model in terms of policy productiom
C: Hubs (yet to test): We would expect a high count of arrivals at max capacity (Unreliable N(t))

### Observations on A group
The model seems to understimate mu, the E[N(t)] is sistematically greater than the median of test days, this 
could mean bicicles are being substracted from the station (yet to investigate)
## Pending Tests
## Proposed Modifications
Adapt model to consider spotaneous substraction/addition of bicicles at fixed times

(Unsure as it implies loseing easy closed form computations) Modify birth and death classic model for a model with off diagonal elements on Q (thus taking into account batch arrivals/ departures wich is the test always failing (altough in some stations not by much))

Treat lambda and mu themselves as random variables (Thus turning the Markov model into a more complex Cox model)
