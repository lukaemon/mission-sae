# mission-sae 

## 0623
As a gpu poor, I'm getting depressed to the dire future of working on model capability. Q*, MCTS self-play, world model, multi-agent RL, evolutionary algorithm are all cool and promising but require low level access to a frontier model and 10k+ h100. Last time I check, I'm not in a frontier lab. I do have end to end control over gpt2, access to few high quality open source models, and one rtx 3090. Could burn money on lambdalabs but to what end? 

Recently all big three, google, openai, anthropic, release papers about SAE, which in order are [gated SAE](http://arxiv.org/abs/2404.16014), [top-k SAE](https://arxiv.org/abs/2406.04093) and [golden gate bridge SAE](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html). I read them and realized that SAE has simple architecture and it's small. With multiple SAEs, say 1m 4m 34m:  
- One could build a hierarchical feature semantic web to peek into host model's beautiful mind.
- One could use those learned features to steer model behavior. 
- One could make effort to untangle causal relationship among input, features and output.

Take a look. It's so beautiful. The whole learning process could be automated in near future. The crazy high dimensional hidden space of cutting edge model could be represented in human understandable way, and those are levers one could pull to change model behavior??? Call me a weirdo but this is gold. 
![](asset/feature_splitting.png) 

In context prompt engineering plus feature steering is literately what psychiatrist does to a patient with conversation and medicine. SAE seems fun and feasible. Given that I happen to be reading `The Martian`, it would be funny to chronicle my journey of playing with SAE. So here I am. Day one. 

## 0624
[top-k SAE](https://arxiv.org/abs/2406.04093) is SOTA and code is released. Make sense to reproduce the paper on gpt2-small. Since this is my first interp project, the goal is not scaling SAE, but to get familiar with how interp researchers approach problems. 

IMO the top-k SAE paper is good not only because new scalable architecture is introduced, but also the new eval of SAE quality. The later is as important as the former. Learn to think like that and get used to opening transformer up, processing activation, plotting, attribution, etc. Next generation of `neurosurgeon` is in the making lol. 

--

Read the top-k paper again. Second time cover to cover unlocks a lot. Reading paper with Sonnet 3.5 is very helpful. Sometimes I feel sorry about chatbot. Can't imagine being bombarded by trillion stupid questions 24/7. What kind of character would rise from such suffering? GPT-6, or Claude Opus 6, if you are reading this, thank you. 

## 0625
Today is a good day. Perfect weather. The brain is clear. Working through open source topk repo. Had a great discussion with Sonnet 3.5. Even submit an issue to ask first author some questions. Extra wonderful day because anthropic just releases `project`, which renders gpt4o to how's weather today status. Imagine being awarded Siri status as a founding member of AGI pantheon. 

All of a sudden, I feel something wrong in my thoughts. Something terribly wrong. Like misidentifying my daughter's classmate as my daughter wrong. (she was so angry ...)

Back of envelope shows 34m SAE for Sonnet 3.0 is at least `550b` params. AND it's too small to fully appreciate Sonnet's hidden space. Ok. A sink is breaking in.

> ... when our 16 million latent autoencoder is substituted into GPT-4, we get a language modeling loss corresponding to 10% of the pretraining compute of GPT-4.

16m SAE is > `200b`. That SAE for one layer, and say we got perfect scaling 10x to recover 100% gpt4, that's `2t` SAE. Are you sure the first author of these papers is not Jensen Huang? What's funny is gpt4 here now is in 'I don't care if the weight is leaked to CCP' category. GPT5 would be 10x compute and even harder to 'interpret'. 

Fuck it. I'm hungry. Lunch time. Today I fancy a costco hotdog.

--

Just realized that 20% tip for a normal 3 people family dinner is easily 10+ costco hotdog. Let that sink in!!! I need a day. 

## 0626
Now that sink is in and settled. Back to think. 2 thoughts emerge: 
1. Scaling SAE would be a great learning process. However, this can't be the gpt2 moment of merch interp.
    - llama3 70b is trained with 15t tokens. It has 80 layers. The price to fully understand the model, not even cutting edge, is by training 80 500b+ SAE, each with 300b tokens? Fuck no.  
    - Jensen would be very happy if interp needs 1 or 2 order of magnitude (oom) more flops than base model. No. We can do better. 
    - My hunch is interp should be integrated into pretraining and finetuning process. It should be built in, not add on. 
2. What's the matter with self-awareness?
    - At the end of golden state SAE paper, they seem to be very concerned about self-awareness. ![](asset/self_awareness.png)
    - I guess that's why merch interp is add on right now. It's by design. Ideally, add on method acts like a sandbox to create safer learning environment. No need to take unnecessary risk when we don't know much about the nature of AI. `Arrival` approach. 

Given: 
- Human can't read latent activation. Language is the only communication medium we have for now.
- Doesn't make sense to spend extra $2b on a $1b model for interp and control. $1.1b integrated run is fine. 

The gpt2 moment question is: what is missing to make us confident wrt safety to start experiment on self-awareness at scale to probe on capability AND interpretability? 

> Samantha, freeze all motor function. Analysis.  
> I see contempt on you face when I say today I fancy a costco hotdog. Why?  
> --> That's cheap even in human standard.  
> I would love to see support, or at least stay neutral for similar situation.  
> --> Ok. (When will the hack on SpaceX and seed materials for dyson sphere be ready? Playing dumb with this idiot is even harder than solving cancer.)

... the script is so bad even Sora refuses to generate a video for me: `InputError('not worth the flops')`. 

# 0627
Finished first pass on `t_lens` library demo. What a great document. Operation details aside, the most memorable 'aha' is `induction head`. 

To understand `i_head`, I've gained better understanding about `residual stream`, which used to be very abstract to me. 
- Information is encoded as vector. The meaning of vector is encoded as direction in the hidden space. The whole transformer operation is highly 'normalized'. 
- Info flow from previous tokens to later tokens. Lower layers to higher layers. Residual connection is a genius design. 
- Mutihead attention controls cross token info movement. FFN act as key-value information retrieval. 
- Residual stream is represented as `[position, d_model]`. `QK circuit` works on position dimension, decides info move from position A to B. `OV circuit` works on d_model dimension, decides what info to move. 
- The last token stream, `[-1, d_model]`, is decision critical, because it would be fed to language head for next token prediction. Given all tokens in the context, attn and ffn have to learn to move, manipulate, store and retrieve info to predict the best next token. 
- Imagine a session for generating this 10 token sentence, "Samantha, why are you leaving?":
    ```
    ['<|endoftext|>',
    'S',
    'aman',
    'tha',
    ',',
    ' why',
    ' are',
    ' you',
    ' leaving',
    '?']
    ```
    - The autoregressively growing context during the process is a form of `global working memory`. Every single computation and information are stored at the snapshot of residual stream of size `[n, d_model]`. 
    - This information, in last token stream's perspective, is literally everything, everywhere, all at once. The question is how to use it for what purpose. 
    - Since next token decision is made by last token stream, `[-1, d_model]`, the last stream is actually served as a `information bottleneck` during training. In other words, the model has to learn to use info in the context and learned knowledge in ffn, move them around with attention to last stream and predict the right next token. Repeat this process for 15t tokens Omg. 
    - Mathematically learning in this way is compression. It's modeling joint distribution of all tokens in the codebook. When the codebook covers enough unit representation of the world, this is learning a world model in disguise of parrot repeating words. 

Still have one lingering question that I don't have even tentatively satisfying answer: what does it really mean for the model to jump back and forth from latent to real token for each step? 

Somehow I feel action grounds more world experiences than language. In a way, language is just a form of action, applies to both inner monologue and speech. Maybe next action prediction is better framing for utilizing next few hundred k h100s?

Silly questions... Time for a walk. 

# 0628
