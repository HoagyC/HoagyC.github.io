Plan: write up of possible trajectories of GPT-like services

What's the model?

We assume a scenario in which there is money to be made from selling GPT-like services, and the development of these models follows the economic incentive. There is also the possibility to add in the creation of these kinds of models for non-economic reasons such as national security or to build expertise in the expectation that models of greater power will have large returns.

The core parameter is the log of the number of parameters of the major model, and how the value of the system scales with 

Ok so we can start with a super basic model.
The value scales exponentially with the log number of parameters, but with a lower base. Need to calculate what that actually looks like.

We then also have the rise of the underlying power. This is composed of the reduction in the cost to train a model of given underlying parameters, and the increase in the power of the model. 

The power of the model and the effects of greater scaling are the biggest factors here


Let's start off with GPT-3. 2x10^11 parameters, 1x10^7 dollars. If we hypothesize that the

We can imagine that there are three parameters, 

$$c_t$$ is the rate at which costs fall
$$c_v$$ is the rate at which value grows with the power of the model.
$$c_p$$ is the rate at which the power of the model grows with the log of the number of parameters.

If $$c_v > c_p$$ then if there is any size of model which is currently profitable, then there is no size so large that it is unprofitable, and all that remains is to accumulate the will and resources to make the largest possible models.

If $$c_v < c_p$$ then time and multiplayer dynamics come into play. 

If we ignore the diminishing returns of building models equal or weaker than other existing ones, it becomes profitable to build models at the time $$t$$ where $$t - t_0 = \frac{1-c_p - c_v}{c_t}\log{p}$$

This implies a smooth increase in the model size over time.

Now we can introduce the idea that the value is captured by building models which are more powerful than any that currently exist. 

There is the most powerful model currently in existence \tbar{P}. I assume that all value that is captured by making a new $$\tbar{P}$$ is captured permanently (it would be better here to be able to capture the expected time before the model become obsolete, but perhaps that can wait until later).

Now the value of 
