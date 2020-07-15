Mega Models and the future direction of high-end machine learning systems.


## Summary 

A hypothetical scenario and weakly-held prediction that the next few years of AI could consist of huge 'mega models' in the GPT-3 vein, which are offered as APIs not just to use, as with GPT-3, but with the ability to fine train specific versions consisting of only a few altered layers. This ability, combined with increasingly capable models, creates an advanced AI landscape dominated by a few profitable mega models which use this income to finance greater size and training time. I explore the viabilty of this, the economics of such a market, and the hypothesized impact on alignment, concluding that it'd probably be a positive sign to see this play out, though with a lot of uncertainty.

## The basic idea

One of the core features of GPT-2 was that it could be downloaded and fine-tuned to write in a particular style, on particuar topics, with distinctive voice, often to great comic effect, as seen with gwern's work on poetry, various tumblr autoresponders etc.

Now, thanks to the crazy doubling time of the size of these systems, these models have become too big to simply download and run (without pretty serious investment) and the model is to be hosted by OpenAI and accesssed via an API. I'll call these too-big-to-handle models mega models.

This puts a stop to the kind of experimentation we saw with GPT-2, but at the same time, as the power of these huge models grows, the more value there will be in using these mega models over the kind that can fit on a single GPU or normal lab set up (at least without a big step forward in model compression). At the same time, while these models are improving in their breadth of ability, I'd expect that there'd still be value in doing this fine-tuning for domain-specific applications. The question is, how will they do this?

## Implementing

My guess is that this will be resolved by OpenAI, or future builders of mega models, providing an API which offers not just the ability to run a forward pass but the ability to finetune the network on a submitted corpus, or more likely, to only finetune the last few layers of the model to ease the computational requirement for all but the highest-paying customers. You'd have a huge encoder-style network doing probably 99% of the work, and a much smaller, flexible section offering domain-specific performance, and this could all potentially be handled by the owner of the mega model, combined into a slick product. 

Alternatively, they could provide a service in which you submit a dataset, and they return not just the forward pass of the data, but the activations for the last few layers of this forward pass, from which a finetuned last few layers could be trained, probably starting with the actual last few layers of the mega model which would be made public. This was the set-up I initially envisioned but it seems worse for a lot of reasons: more convoluted, less control for the owner of the model, requires engineering and compute from the end-user who will be less specialised in the area and thus more expensive. 

## Consequences

Ultimately, what happens in this scenario depends on how much power these models offer. The more that these models offer genuine awareness and context, and the ability to give 'serious' output when desired, the more money there will be in these services.

One important variable for safety will be the structure of the market for these models. At the moment there doesn't seem to be much possibility for competitive advantage other than model size and competent engineering. I haven't seen much evidence that companies are able to credibly conceal meaningful breakthroughs (though of course publicizing a secret breakthrough would not be an intelligent strategy for keeping something a secret) so I woul expect a level playing fiel on the algorithmic front. The increasing returns to scale suggest the possibility of monopolist emerging, but with a monopoly comes the incentive to charge monopoly prices. 

Economic theory suggests that monopolists can be sustained if there's a credible threat of lowering prices drastically in response to new entrants but to be credible the monopolist must have a much deeper cash pile/credit line than their theoretical competitors. I would expect that the national security value of advanced AI would mean that there would be a lot of funding to create a competitive market in these huge models, and if they are wildly profitable then anyone who trained one would likely attempt to monetize it. 

Resulting in a monopoly seems unlikely - if the value of making these large models totally plateaus and there's never much of a serious market then it could remain a monopoly in that the interest largely dies off and only a few make these models as research projecst then this could be monpoly, but not a very consequential one and this scenario would be falsified. If there's solid power in these models we would expect multiple players to enter for combinations of national security and profit motive. The more profitable it is to scale these models, the more important it would be for major players to compete in this arena so I find it hard to imagine a monopolistic equilibrium. If the returns to scale were so large as to make the competition unable to compete then I imagine we would be hedaing towards foom territory and economic considerations would start to take a back seat to power politics or chaos. 

Overall I would expect this to result in a market with a handful of dominant players backed up by governments or the largest of corporations.

## Will it happen?

It requires AI to spend long enough being smart enough to provide high value assistance without reaching a point where they become so powerful as to make the world go crazy.

It also requires that big systems will dominate smaller systems, which I think is very plausible for use in areas where deep context is very important.

There's something about the way that these systems work which suggests a rather interesting form of takeoff. These systems are more creative than they are correc,t something you prod for inspiration rather than trusting to do the hard yards, but the ability to query them could rise and rise, and the amount that their responses are trusted rises impercetibly over time as Google has done with specific queries. Brb creating a GPT-2 based writing assistant.

## Alignment Considerations

The main safety-relevant part of this to me would be that it would concentrate a lot of power into the hands of one or a few models. To somebody outside of alignment this  sounds pretty negative - certainly it sets some Skynet-style bells ringing - but I imagine that it would actually be beneficial. 

It would allow safety efforts to focus more directly on the properties of one or a few models, giving more time to probe these models in detail. In a best case scenario, these would be unambiguously state-of-the-art in many areas but unable to improve other than by throwing even greater amount of money at training, slowing down the pace of development to something closer to Moore's law, while bringing attention to just how general these models have become.

While it wouldn't allow for control of how the network is actually used, there's potential for verifying that the encodings produced have robust conceptual categories (though it doesn't seem like we've made much progress in how this would be done in concrete terms). If these add-on models were hosted by the owner of the mega model then there would be even greater potential for control, and huge leverage for whoever was in control, which is potentially risky but better than the chaotic case. It would be difficult to check what the outputs are being used for but there would be potential to screen both the kinds of inputs. In the case where there were a few mega models associated with world powers I imagine there would b

A more pessimistic case would be that the returns to scale result in the training of huge models earlier than would otherwise happen, and increase our capacity for scale relative to alignment in a manner even more lopsided than we have now.

## Questions:

Do people think this is a plausible/likely direction for the industry to move in?

What do people think the effects would be?
