## Understanding the regret bounds in MuZero

The Muzero planning algorithm learns to play games with a discrete space of moves. It uses a neural network to predict the average score that will be found from a position, but it is also able to look through the tree of possible trajectories, taking moves and then predicting the expected value from *those* positions, and passing that value back down the tree to the original node, and thereby improves the estimation of the value of each move, so that it can eventually return the best.

This makes a lot of sense, and has proved to be a very productive approach, but when you look at the implementation detail of this algorithm, we see the following algorithm, which doesn't look nearly so neat or comprehensible.

![Search Equation](images/search_equation.png)

Firstly, what do the various terms mean? 
- \(a\) denotes an action that's available at node \(s\). Illegal actions are not screened off here - simply penalizing them harshly is enough to quickly prevent them being taken.

- \(Q\) is the mean value of each action as found by previous simulations. Each simulation is played out until a new node is reached, at which point the total expected future value is estimated (using NN) at this final node, and the intermediate reward is also estimated at each node between this new node and the root node, and the sum of these, appropriately discounted, is the value estimate of the action. Q is the average value of all simulations starting with that action, and is 0 if no actions in this direction have yet been taken. All value estimates are linearly normalized to between 0 and 1, and the simulation keeps track of the smallest and largest reward that have been found to maintain this scaling.

- \(P\) is called the policy, but is perhaps better called the prior. This is an estimate (again, using a trained NN) of the distribution of actions that will be taken by the simualator, and is trained (though only trained on distributions of actions at the root node, not the more sparse distrbutions found deeper in the tree). This allows the search tree to be pruned towards those actions that will go well - which is vital here because what's noticably not present here is the estimated value of the action, or the state immediately resulting from the action. This estimate exists, as it has to estimate the value of trajectories, but it only comes into right when a new node is reached (though when a node is first explored, all nodes will be new and so the value function will play a large initial roll in determining the direction of the rollouts.)

- \(N(s,a)\) is the number of simulations from this node thta have tkaen action a, and the \(sum_b{N(s,b)}\) is the total number of simulations that have gone beyond this node. Note that all of this tree structure is built fresh for every step of every game.

- Lastly, the constants \(c_1\) and \(c_2\) are, in the words of the paper, 'used to control the influence of the prior \(P(s,a)\) relative to the value \(Q(s,a)\) as nodes are visited more often. In our experiments \(c_1=1.25\) and \(c_2=19652\).

