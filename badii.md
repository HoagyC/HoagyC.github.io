I started out reading B&H in the hope of getting a better understanding of the concepts and difficulties around Kolmogorov complexity and its various similar forms. 

What I foun instead was a book which took quite a significantly different tack, to be applied to rather different problems, but nonetheless interesting ideas .

Unfortunately I do not have sufficient background in mathematics and physics to make total sense of the preceding chapters but, since I've made the effort to understand what's going on I may as well give a go at explaining how these concepts are made applicable to physics.

Non-linear dynamics are made amenanable to discrete complexity analysis by the following process: The system is first represented as a phase-space which allows the entire state of the system to be represented by a series of numbers, so that knowing its position in phase space tells you all there is to be known about the state at this time. It seems to be possible to handle an infinite number of dimensions here but I have had to resort to imagining a finite number.

We then take assume that we've found a periodic orbit in this phase space. We take a slice through this phase-space which is called a Poincare section. Since the orbit is periodic, it will obviously pass through this slice and return to the same point. What is required, for it to be a Poincare section, is that for any other point that is infinitessimally close to this periodic point, will also pass through the section, and the slice, taken as a whole 

What this allows is to make a map which describes the dynamics of the system around te point by the map which takes each point on the slice and maps it to the point that its orbit next touches on the slice. The continuous time dimension of this dynamical map is thereby made discrete! What can then be done is divide this surface into cells which give it a discrete coordinate grid, and is this is done correctly ten the order of the cells that a point reaches uniquely defines the dynamics of the system from this point (though this does not seem to be an easy thing to construct in general).

(Tangentially, it's also described in a review by Bactra the way in which lattices are used to model hydrodynamics in place of actually being able to solve the Navier-Stokes equations exactly. To my knowledge the two areas are not particularly connected but it's interesting to see how widely discretization is applied in maths and physics - I'd never preivously understood why the mathematics of lattices was worth studying, but then spin systems, and spin systems in quantum computers..)

They begin by making an important distinction:
> Either one is interested in studying a single item produced by an unknown source, or in modelling the source itself through the properties of the ensemble of all its possible outputs. In both views, however, complexity is naturally related to randomness through the concepts of compressibility or predictability of the data in so far as genuinely random objects are most incompressible and unpredictable. 


