## Introduction

When I first heard about EfficientZero, was amazed that it could learn at a speed comparable to humans, and I wanted to practice programming, so I thought I'd make my own version.

The idea here is to give an overview of EfficientZero, and MuZero which it relies upon, but also give a sense of what it looks like to implement in practice.

You can have a look at the full code and run it at on [github](https://github.com/hoagyc/muz).
I'm still tinkering with it and uploading commits of questionable functionality to load into Colab, but this commit [COMMIT] is a pretty stable place to start.

### Basics:
#### Neural Networks
There are three neural networks, the Representation network, the Dynamics network, and the Prediction network. These are separate networks, but they can be neatly combined into two functions, `initial_inference(obsevation)`, and `recurrent_inference(latent_vector)`. 

#### Playing
The core of using the MuZero algorithm to play a game is building the tree structure by which the algorithm explores the tree of possible moves, and therefore decides what to do.

The main interpretation has this section written in C++ to minimize the time taken to create this tree but I've not found that this is a bottleneck.

The basic structure of MuZero is that you have a tree which represents the state of your exploration. The algorithm is built for the case where the action space is finite, and so each node has a slot for each potential child, initialized as `self.children = [None] * action_size`.

The initial node is created by first getting a latent representation of the observation, and using the prediction network to estimate the value and predict the eventual action distribition:

```
frame_t = torch.tensor(current_frame, device=device) # Make tensor from frame array

# These can be brought together as 'initial_inference'
init_latent = mu_net.represent(frame_t.unsqueeze(0))[0]
init_policy_logits, init_val = [x[0] for x in mu_net.predict(init_latent.unsqueeze(0))]
```

and then from the logits of the predicted action distribution (`init_policy`) from the prediction network, we get our final probabilities for how we will begin to explore the tree by:

```
init_policy_probs = torch.softmax(init_policy_logits, 0)
init_policy_probs = add_dirichlet(
	init_policy_probs,
	config["root_dirichlet_alpha"],
	config["explore_frac"],
) # This adds some simple noise to the probabilities of taking an action to encourage exploration

root_node = TreeNode(init_latent, init_policy_probs, ...)
```

With this root node we have the basis for our exploration tree and can begin to populate it:

```
for i in range(config["n_simulations"]):
	# It's vital to have with(torch.no_grad()): or else the size 
	# of the computation graph quickly becomes gigantic
	
	current_node = root_node
	new_node = False # Tracks whether we have reached a new node yet
	search_list = [] 
	# Search list tracks the route of the simulation through the tree
	
	# We traverse the graph by picking actions, until we reach a new node
	# at which point we revert back to the initial node.
	while not new_node: 
		...
```

To traverse the tree within this `while` loop we do the following:

```
action = current_node.pick_action()
if current_node.children[action] is None:
	action_t = nn.functional.one_hot(
		torch.tensor([action], device=device),
		num_classes=mu_net.action_size,
	)
	
	# This can be brought together as 'recurrent_inference'
	latent, reward = [x[0] for x in mu_net.dynamics(latent.unsqueeze(0), action_t)]
	new_policy, new_val = [x[0] for x in mu_net.predict(latent.unsqueeze(0))]
	
else:
	# If we have already explored this node then we take the 
	# child as our new current node
	current_node = current_node.children[action]
	
```

Here the `pick_action` function is doing a lot of work in deciding what this looks like. 

We pick the action with the following function: 

```
def pick_action(self):
	"""Gets the score each of the potential actions and picks the one with the highest"""
	
	total_visit_count = sum([a.num_visits if a else 0 for a in self.children])
	scores = [
		self.action_score(a, total_visit_count) 
		for a in range(self.action_size)
	]
	maxscore = max(scores)
	
	# Need to be careful not to always pick the first action is it common 
	# that two are scored identically
	action = np.random.choice(
			[a for a in range(self.action_size) if scores[a] == maxscore]
		)

return action
```

The action score function has the following formula, but in truth this formula overstates the complexity because the constants used are `c1 = 1.25; c2 = 19652` which means that with on the order of 100 simulations, the final part, or balance term, never differs far from one and is basically ignored. The rest is a balance between the score that has been found so far, and the product of the prior, from the NN, and explore term favouring new actions.

```
def action_score(self, action_n, total_visit_count):
	"""
	Scoring function for the different potential actions, 
	following the formula in Appendix B of MuZero
	"""
	
	child = self.children[action_n]
	
	n = child.num_visits if child else 0
	q = self.minmax.normalize(child.average_val) if child else 0
	
	prior = self.pol_pred[action_n]
	
	# This term increases the prior on those actions which have been taken 
	# only a small fraction of the current number of visits to this node
	
	explore_term = math.sqrt(total_visit_count) / (1 + n)
	
	# This is intended to more heavily weight the prior 
	# as we take more and more actions.
	# Its utility is questionable, because with on the order of 100 
	# simulations, this term will always be very close to 1.
	
	balance_term = c1 + math.log((total_visit_count + c2 + 1) / c2)
	score = q + (prior * explore_term * balance_term)
	
	return score
```


#### Training:

Training a network of this type is quite ordinary in many ways but the structure of the system, in which we learn a recurrent dynamics network, requires a bit of extra work. The network is unrolled to a particular depth, here called `config[rollout_depth]` which is always set to 5, but each individual example in a batch may not be this deep, because the game may end in less than 5 steps. 

We need to do this within a for loop, rather than as a single forward pass, because the dynamics function requires the output of the previous dynamics function, which is why we can see it as an unrolled recurrent network.

```
for i in range(config["rollout_depth"]):
	# This tensor allows us to remove all cases where 
	# there are fewer than i steps of data
	screen_t = torch.tensor(depths) > i 
	
	if torch.sum(screen_t) < 1:
		continue

	target_value_step_i = target_values[:, i]
	target_reward_step_i = target_rewards[:, i]
	target_policy_step_i = target_policies[:, i]
	
	pred_policy_logits, pred_value_logits = mu_net.predict(latents)
	new_latents, pred_reward_logits = mu_net.dynamics(latents, one_hot_actions)
	
	# We scale down the gradient, I believe so that the gradient 
	# at the base of the unrolled network converges to a maximum 
	# rather than increasing linearly with depth
	
	new_latents.register_hook(lambda grad: grad * 0.5)
	
	pred_values = support_to_scalar(
		torch.softmax(pred_value_logits[screen_t], dim=1)
	)
	
	pred_rewards = support_to_scalar(
		torch.softmax(pred_reward_logits[screen_t], dim=1)
	)
	
	val_loss = torch.nn.MSELoss()
	reward_loss = torch.nn.MSELoss()
	
	value_loss = val_loss(pred_values, target_value_step_i[screen_t])
	reward_loss = reward_loss(pred_rewards, target_reward_step_i[screen_t])
	policy_loss = mu_net.policy_loss(
		pred_policy_logits[screen_t], target_policy_step_i[screen_t]
	)  
	
	batch_policy_loss += (policy_loss * weights[screen_t]).mean()
	batch_value_loss += (value_loss * weights[screen_t]).mean()
	batch_reward_loss += (reward_loss * weights[screen_t]).mean()
	
	latents = new_latents
```
This is a little longer but basically what we're doing is to build up the losses by unrolling, screening at each step to remove games that have finished, and scaling down the gradient at each step so that the gradient converges to a finite value rather than scaling lineraly with depth.

##### Support to scalar
One notable detail is the use of `supprt_to_scalar` functions. These are a slightly peculiar piece of MuZero, by which the value and reward functions, although they are ultimately predicting a scalar, actually predict logits of a distribution over numbers, which are roughly proportional to the square of their centered position, so a support of width 5 would correspond to values roughly $[-4, -1, 0, 1, 4]$, and logits which softmax to $[0.5, 0.5, 0, 0, 0]$ would correspond to a final value of $-2.5$ although the details are slightly more complex.

#### Reanalysing:

This is the addition mentioned in MuZero reanalyse, and basically reassesses the values and policies in past games.

More specifically, the target 'value' is the discounted sum of the next `config[value_depth]=5` steps of actual reward, plus the estimated future reward after these 5 steps. While clearly not a perfect picture of value, this is enough to bootstrap the value estimating function. This target value will be worse if the value estimation function is worse, which means that the older value estimates will provide a worse signal, and so the reanalyser goes through old games, and updates the value estimates using the new, updated value function.

Updating these values basically consists of constructing trees exploring the game at each node, just as if we were playing the game 

```
p = buffer.get_reanalyse_probabilities()
ndxs = buffer.get_buffer_ndxs()
ndx = np.random.choice(ndxs, p=p)

game_rec = buffer.get_buffer_ndx(ndx) # Gets the game record at ndx in the buffer
vals = game_rec.values

for i in range(len(game_rec.observations) - 1):
	obs = game_rec.get_last_n(pos=i)
	new_root = search(current_frame=obs, ...)
	vals[i] = new_root.average_val

print(f"Reanalysed game {ndx}")
```
#### Multiactors:
To speed up training and playing, we parallelize by converting the main classes into 'actors', as defined by the `ray` framework. This means wrapping classes with the `ray.remote()` decorator, and then calling their functions with `actor.func.remote(*func_args)` instead of `actor.func(*func_args)`.
<!--ID: 1655227807223-->


The basic classes are the Player, Trainer, and Reanalyser, and each of these have access to a Memory class and a Buffer class from which to pull data. 

### EfficientZero:

#### Value prefix
In MuZero, the network tries to predict the reward at each time point
This apparently causes difficulty due to the 'state aliasing' problem, by which the 
If there is an upcoming 
**LOOK BACK AT PAPER**
The EfficientZero architecture changes the prediction of the reward into the prediction of the value prefix. This is a prefix because the 



#### Consistency Loss
The idea here is that in these deterministic games, the latent vector representing the state of the game as the network expects it to be, after a series of actions (i.e. applying the `represent` network to the initial observation, and then applications of the `dyanmics` network), should be the same as the latent found after that series of actions is actually taken in game, and then the `represent` network is applied to the final observation.
```
class Trainer:
	...
	def train():
		...
		# The target latent is the representation of the observation, i frames on
		# from the initial observation
		if config["consistency_loss"]:
			target_latents = mu_net.represent(images[:, i]).detach()
		...
		# The latent here is the latent that found by applying the dynamics
		# network with the chosen actions to the initial latent i times.
		if config["consistency_loss"]:
			consistency_loss = mu_net.consistency_loss(
				latents[screen_t], target_latents[screen_t]
			)
```
The consistency loss used here is a cosine loss, meaning the cosine of the angle between the `latent` and `target_latent`, interpreted as vectors in $\mathbb{R}^n$. 
#### Off policy correction
This is a simple change that improves the value target.

The idea is simple: that the value target is the discounted sum of the next $n$ steps of observed value, plus the expected value at the $n$th step, but as the policy that generated the actions ages, and diverges from the current policy, the future steps become a less useful proxy for the actions that would be taken now, and so this change shrinks $n$ as the number of steps between the current time and when the trajectory was sampled increases.
```
def get_reward_depth(self, val, tau=0.3, total_steps=100_000, max_depth=5):
	if self.config["off_policy_correction"]:
		# Varying reward depth depending on the length of time 
		# since the trajectory was generated.
		# Follows the formula in A.4 of EfficientZero paper
		
		steps_ago = self.total_vals - val
		depth = max_depth - np.floor((steps_ago / (tau * total_steps)))
		depth = int(np.clip(depth, 1, max_depth))
	
	else:
		depth = max_depth
	
	return depth
```

### Other interesting bits



### Pitfalls:

The most difficult part of the process was 

I had got the code to a point where it was working consistently over long runs, and then set it to perform a test of various hyperparameters, and would find that after several hours, at some points it would just.. die. No error message, no hint of what caused it, the process would just end. Because I was using `ray`, I guessed that there was some kind of problem that broke the sytem in such a way that didn't allow it to exit gracefully, some kind of memory error..

After a lot of frustration and confusion, and self inflicted damage like updating all packages, I started just ignoring it and working on something else, at which point I realised that even trivial errors weren't showing up. Although initialliy this seemed like another blow, actually meant that

 Once I knew that I could replicate the 'no traceback' issue just by introducing a trivial error, I could then easily go back through the commits and find the point at which the traceback disappeared, which made finding the cause super easy.

I'd used `ray.wait()` instead of `ray.get()` to get the final results of the actors, and when one of those actors crashed, `ray.wait()` continued, and immediately hit the end of the script, at which point all the actors were cancelled, before even the error message could be printed! Unfortunately, I'd made this change just after becoming confident that there were no bugs, so instead of being a simle error to find, it was found only after hours of running and many commits.

The main takeaway was not to overly focus in on one possibility 
The worst case scenario, that I had some deep bug that caused an error is such a way that the process immediately died was possible, but I'd far too easily focused on this, instead of the case where I'd caused the lack of traceback by a silly error.

Interesting points for me own:

Found myself naturally converging on similar architectures. When starting off I looked at [open impl] and [muz code] to look for ideas when things weren't working, but I also made a conscious decision not to follow to the way they'd organized their code, and after a while, the differences compounded to the point where I could much directly from their code, even if I wanted to. 

Nonetheless, I often found that I was forced into becoming more similar.

For example, I'd followed [open impl] in converting my classes into Ray actors, which would then concurrently train. At first this was just the player and the trainer, but then having a separarte memory class, and then finding that having the memory

You want to quickly be able to get basic statistics from the memory so you want a lightweight memory actor which is never blocked. However, you also need to do a lot of work to retrieve batches of data from the buffer, and these long operations leave the memory actor blocked, which delays lots of other functions. It's therefore helpful to split the memory into one which stores and return and buffer actor which

The number of cores is important here - very different optimization on the 6 cores of my laptop than on the 2 of Google Colab that I use to run larger models.

Even though we're using the same algorithm for different games, there are differences in the operations - for example doing some basic normalization on the pixel values 
For just one or two games, it's quite easy to add if/else statements to process these differently, but this gets ugly quickly, and so becomes better to wrap the game class into a class which implements the needed functions according to the same arguments (and by taking a `config` argument, this can in effect be overloaded so that we can use the same function call for different 


### Ideas (to be censored just in case they're of value):
 Could computation be saved if we assumed by default that the subtree which contains our actual action was more-or-less correct, perhaps with an estimator (could use simple stats on divergence between hidden state and predicted hidden state, or use ANN) of which nodes are uncertain, and spend more time expanding this tree?)

Could initial learning be faster if we also included a network that attempted to predict the output itself. In the long run it would be counterproductive as it would force the network to try and save all of the information about the observations, instead of only those which are relevant to playing well, (and a randomly generated sliding background would make life difficult) but often the key dimensions of variation are the things that matter, and this would increase the incentives to record these in the initial stages, before the network has completed the more complex task of figuring out if and why they are relevant to playing well.

