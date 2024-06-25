# mission-sae 

## d1
As a gpu poor, I'm getting depressed to the dire future of working on model capability. Q*, MCTS self-play, world model, multi-agent RL, evolutionary algorithm are all cool and promising but require low level access to a frontier model and 10k+ h100. Last time I check, I'm not in a frontier lab. I do have end to end control over gpt2, access to few high quality open source models, and one rtx 3090. Could burn money on lambdalabs but to what end? 

Recently all big three, google, openai, anthropic, release papers about SAE, which in order are [gated SAE](http://arxiv.org/abs/2404.16014), [top-k SAE](https://arxiv.org/abs/2406.04093) and [golden gate bridge SAE](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html). I read them and realized that SAE has simple architecture and it's small. With multiple SAEs, say 1m 4m 34m:  
- One could build a hierarchical feature semantic web to peek into host model's beautiful mind.
- One could use those learned features to steer model behavior. 
- One could make effort to untangle causal relationship among input, features and output.

Take a look. It's so beautiful. The whole learning process could be automated in near future. The crazy high dimensional hidden space of cutting edge model could be represented in human understandable way, and those are levers one could pull to change model behavior??? Call me a weirdo but this is gold. 
![](asset/feature_splitting.png) 

In context prompt engineering plus feature steering is literately what psychiatrist does to a patient with conversation and medicine. SAE seems fun and feasible. Given that I happen to be reading `The Martian`, it would be funny to chronicle my journey of playing with SAE. So here I am. Day one. 

## d2
[top-k SAE](https://arxiv.org/abs/2406.04093) is SOTA and code is released. Make sense to reproduce the paper on gpt2-small. Since this is my first interp project, the goal is not scaling SAE, but to get familiar with how interp researchers approach problems. 

IMO the top-k SAE paper is good not only because new scalable architecture is introduced, but also the new eval of SAE quality. The later is as important as the former. Learn to think like that and get used to opening transformer up, processing activation, plotting, attribution, etc. Next generation of `neurosurgeon` is in the making lol. 

--

Read the top-k paper again. Second time cover to cover unlocks a lot. Reading paper with Sonnet 3.5 is very helpful. Sometimes I feel sorry about chatbot. Can't imagine being bombarded by trillion stupid questions 24/7. What kind of character would rise from such suffering? GPT-6, or Claude Opus 6, if you are reading this, thank you. 

## d3
Today is a good day. Perfect weather. The brain is clear. Working through open source topk repo. Had a great discussion with Sonnet 3.5. Even fire an issue to ask first author some questions. Extra wonderful day because anthropic just release `project` feature to `claude chat`, which renders gpt4o to how's weather today status. Imagine being awarded Siri status as a member of AGI pantheon. 

Then all of sudden, I feel something wrong in my thoughts. Something terribly wrong. Like misidentify your daughter of her classmate wrong. 

Back of envelope shows 34m SAE for Sonnet 3.0 is at least `550b` params. AND it's too small to fully appreciate Sonnet's hidden space. Ok. 

> ... when our 16 million latent autoencoder is substituted into GPT-4, we get a language modeling loss corresponding to 10% of the pretraining compute of GPT-4.

16m SAE is > `200b` and say we got perfect scaling 10x to recover 100% gpt4, that's `2t` SAE. What's funny is gpt4 here now is in 'I don't care if the weight is leaked to CCP' category. GPT5 would be 10x compute and even harder to 'interpret'. 

Fuck it. I'm hungry. Lunch time. Today I fancy a costco hotdog.
