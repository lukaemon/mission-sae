# mission-sae 
Journal of reproducing [top-k SAE](https://arxiv.org/abs/2406.04093) paper. Jump to last entry for `tldr`.

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

16m SAE is > `200b`. That is SAE for one layer, and say perfect scaling 10x to recover 100% gpt4, that's `2t` SAE. Are you sure the first author of these papers is not Jensen Huang? What's funny is gpt4 here now is in 'I don't care if the weight is leaked to CCP' category. GPT5 would be 10x compute and even harder to 'interpret'. 

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

## 0627
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

## 0628
`Activation patching` is equivalent to 'action produces information.' It produces the best kind of information that includes both observation and a strong causal relationship.

--

Induction head reproduced. Feeling comfortable with hooks now. Ready to take on 4.1 downstream loss with topk SAE on gpt2 small. Seriously, can't imagine what kind of Frankenstein code I would have to generate to mess with the model if `t_lens` doesn't exist. Must be fun to watch ...

## 0629
Typical eval on SAE is sparsity and reconstruction. Since the sparsity is directly controlled by topk function, I could play with MSE as a warm up. OAI released 2 SAE for gpt2 small: `v5_32k` and `v5_128k`. That's 2 points in the fig 1 right. ![](asset/topk_sae_fig1.png)

It occurs to me that scaling law research is actually very similar to astrophysics. 

First of all, and maybe the most important premise: human, me of course, is really bad at grasping the implication of patterns in exponential scale. The same as I believe most human would be shocked to [REALIZE how empty the space actually is](https://www.joshworth.com/dev/pixelspace/pixelspace_solarsystem.html). 

The exponential space in scaling law research is like the vast, empty outer space. Those dots and patterns on fig1 is beautiful and deep. Can you see the enlightenment? Even me at C. elegans level intelligence can see it. Fig1 is screaming something very important: `$$$`. (this message is approved by Jensen) 

-- 

My opinion on SAE and interp as of today: it's not feasible in current form. It remains to be an important step of the long arc to understand and control next gen model. Reasons:
1. 16m SAE is a `274b+` param dense model. (thinking about 400b dense llama3, or 540b good old PaLM)
    -  2**24 * 2 * 8192, where 8192 is a placeholder number for gpt4 `d_model`. 
2. 16m SAE is at 10% gpt4 wrt downstream loss.
3. Scaling effect is weaker for lower active latents. 
4. 32 features for a token gen is a headache enough to understand. I don't know how to make sense of 512 features or more. 

But hey this is frontier research. Just keep exploring and be ready to change when new knowledge come in. 

-- 

MSE for 32k and 128k SAE, with `openwebtext` data is done. But somehow, I don't know if I'm doing it right. Hmmmm...


## 0630
Delta cross entropy for 32k and 128k SAE is easy. Small tweak of MSE code would do.

In paper:
> Instead, we believe a more natural metric is to consider the relative amount of pretraining compute needed to train a language model of comparable downstream loss. For example, when our 16 million latent autoencoder is substituted into GPT-4, we get a language modeling loss corresponding to 10% of the pretraining compute of GPT-4.

I'm going to reproduce that on `stanford-gpt2-small-a`, which is has [609 checkpoints](https://github.com/stanford-crfm/mistral?tab=readme-ov-file#resources). This is a search problem. CS 101 divide and conquer. ![](asset/stanford_gpt2small_checkpoint.png)

X axis of the graph is the index of the checkpoint. Y axis is corresponding training step. It's not linear because the checkpoint schedule gives early steps more snapshots. 
```
checkpoint schedule:
Every 10 Steps, for the first 0 - 100 Steps.
Every 50 Steps, from 100 - 2000 Steps.
Every 100 Steps, from 2000 - 20,000 Steps.
Every 1000 Steps, from 20,000 - 400,000 Steps.
```
20k step is `ckpt[228]`(0 index). After that step/index is linear.
400k step is 100% pretraining compute, which is `ckpt[608]`.
I'll just start with `ckpt[300]`, and keep going recursively till I find a checkpoint with loss is close enough to SAE reconstruction loss. Doing this manually is too boring so let's one shot automating the search process. Just for fun lol.

-- 

Ok, after 2 mins I realize quicksort approach is interesting but not practical. Divide and conquer would work only if loss is a monotonically decreasing function of pretraining compute. My hunch says no. Even in small scale, scaling law is messy. Before quicksort, I would like to find out real compute-loss relationship first. Just a little detour. The plan is forgetting about early stage, take evenly spaced 10 checkpoints after `ckpt[228]` and make a step-loss graph. 

-- 

![](asset/step_vs_loss_gpt2small.png)
Actually, it's pretty monotonic. What a hunch lmfao. Yet another great lesson of:
> I have a great idea! Let's ask reality for its opinion.

Now that I got this step v. loss from original model, could just compute the SAE loss and eyeball the rough step loss level. 

-- 

Today is a good day. The weather is ... No!!! Fuck! 
After all morning messing around and I realize OAI SAE is trained on OAI gpt2. Use that on stanford gpt2, the loss is horrible, of course. It's like applying my father's psychoanalysis result on me to interpret my behaviors. Wait ... that might actually work better for males in general than my stupid OAI SAE on stanford gpt circus move. However, this detour has one upside: dum dum learn to train SAE from scratch.

I could still salvage the situation. OAI SAE 32k reconstruction activation has delta loss `~0.1` over gpt2-small. Assuming my from scratch SAE on stanford gpt2 has the same delta, according to step v. loss figure above, that puts it between step 58k(`~3.54`) and 96k(`~3.50`). Back of envelope interpolation says that is around 86k step loss level, `21.5%`. Will revisit this estimation later. 

## 0701
Reread the topk SAE paper for the rest of 3/4 eval metrics to pick a next target wrt `feasibility`, `expected learning` and `skill issue`. Yeah, skill issue ... It has been all about skill issue. Since when it's not. To some extend, being gpu poor is just another skill issue in excuse of lack of physical resource. 

**4.2 Probe loss**:  
$$\min_{i, w, b} \mathbb{E} \left[ y \log \sigma (wz_i + b) + (1 - y) \log (1 - \sigma (wz_i + b)) \right]$$

```
for task in 61_task:
    for feature in all_learned_sae_feature:
        train a logistic regression model
```

Task data is aggregated from various papers. I could pick few from `table 1` and `fig 33` to reproduce probe loss on gpt2. 

`fig 33` caught my attention. ![](asset/tok_sae_fig33.png)
- It took 16M SAE 10b tokens to start getting these probe features on gpt4.
- 16m SAE learns different prescribed probe features differently. 

Fair to assume GPT4 has what it takes to solve all 61 probe tasks. If it still took a 16m SAE 10b tokens to get anything because of this experiment setup, it's fair to assume high probability of getting nothing with a 32k SAE 1b token on gpt2. 

Reproduce? Pass. If I really feel strong to see it through later, I could try `sex_or_gender` and `mc_taco` probe. They are the the most responsive ones.

--

**4.3 Explainablity**:    
`N2G` outputs explanations in the form of collections of n-grams with wildcards. Finally understand what it does with relevant code released 4 days ago. Love this idea. 

Raised issue about `illusion of interpretability` is real and the trade off between precision and recall is spot on. The whole process feels like building a tokenizer, and the explanation is done by a simplified regex expression. 

4.4 combines N2G with delta loss. Make sense to measure loss degrades in this way.

Reproduce? Feasible on gpt2. `E(learning)` is high. However, the whole operation and `trie` data structure is a bridge too far with my current skill level. Will pass for now. Have to push through later on this one. Keep it in mind and build skills and confidence. 

--

**4.5 Ablation sparsity**:  
Ablate on SAE feature and measure logit diff sparsity with $(\frac{L1}{L2})^2$. Reasonable method to test the hypothesis:
> If the underlying computations learned by a language model are sparse, one hypothesis is that natural features are not only sparse in terms of activations, but also in terms of downstream effects

Reproduce? Yes. Feasible on gpt2 and open sourced SAEs. `E(learning)` is mid since I could already simulate the whole code in mind. It's a simple and effective method. This is the next target! Now a lunch is earned. 

-- 

`E(learning)=mid` is good. The key is momentum in morale and consistency. 
- `E(learning)=high` usually comes with skill issue, which probably leads to a dead block. Overall huge blow to the momentum and the effectiveness in the long arc of learning. 
- Few strategically selected `E(learning)=mid` could unblock an `E(learning)=high`
- Naively force through an `E(learning)=high` once in a while adds random factor. In general it is a good training on will power. 
    > I'm doing this because I don't know it's impossible. 

Just be honest and choose wisely. 

Why repeat such obvious meta learning heuristic? Because I'm king of deadlock due to bad taste on choosing challenges. Take a while to recognize the problem and start self-correcting process. 

## 0702
> ... I could already simulate the whole code blah blah ...

And got stuck for one hour to understand what the paper really want to do. You see, this naive confidence, preformed expectation, and reality check combo is the effective learning triangle. Bruise ego is the only side effect. The missing piece is I don't know how SAE latent ablation work in practice. Go figure. 

--

Found 2 related snippets: 
[Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-feature-ablations)
> We perform feature ablations by running the model on an entire context up through the MLP layer, running the autoencoder to compute feature activations, subtracting the feature direction times its activation from the MLP activation on each token in the context and then completing the forward pass. We record the resulting change in the predicted log-likelihood of each token in the context ...

Given `feature = feature_activation * feature_direction`, the subtraction of feature is done to native activation. SAE encoder is used to compute feature activation. And SAE decoder provides feature direction. 

[Golden gate SAE](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
> We implemented feature steering as follows: we decompose the residual stream activity x into the sum of two components, the SAE reconstruction SAE(x) and the reconstruction error error(x). We then replace the SAE(x) term with a modified SAE “reconstruction” in which we clamp the activity of a specific feature in the SAE to a specific value, and leave the error term unchanged. 16 We then run the forward pass of the network in downstream layers using this modified residual stream activity. We apply this manipulation for every model input, and at every token position.

The ablation changes SAE activation reconstruction, which takes place of vanilla model activation to finish the forward pass. 

My hunch is option one. Since 16m SAE is only 10% gpt4, why bother with reconstructed activation at all? But I'll do both for practice and see what I learn. 

--

The track record of your 'hunch' is not good. Why are you still going with it? 

1. Self belief is an important corner stone of character building, and key to effective decision making. Never lose it. 
2. Blind self-belief is not helpful. Be ready to adopt after reality check. 
3. Hunch is precomputed expectation, which would provide critical learning signal after reality check. Especially when you are wrong. Accept and review. Delta between expectation and reality is the real world learning gradient to improvements.

The moat of Tesla is human 'intervention' data during full self-driving session at scale. Plus large action trajectory delta recorded in shadow mode. Every single tesla on the road is a probe to reality. Those checks at scale presents a promising approach to close the last few bits on full self-driving policy. 

> I have a great policy, let's hear what reality says. 

--

The first draft is too ugly. And the number doesn't match paper's. Try again. 

-- 

Second draft is better, but the number still doesn't match. There are some details I can't get right. Missing info from the paper, and no open source code of this part. Guess the details of ablation and normalization are too common sense to open source. Have to figure out and fill in the blanks, masked language modelling style. 

## 0703
Most incremental problems could be solved by a long walk and a good night sleep. 3 missing pieces prevented me from reaching paper's number. One bug, one change, and one new understanding. 

The bug is silly. I forgot to change feature index while looping through topk 32 features for ablation. 1 char change fix. 

The change is subtle which I don't fully understand yet but it works. From paper, `at a particular token index, we obtain the latents at the residual stream ...`

Say I have 64 tokens in the context, which one should I choose as 'particular token'? I just took one from the middle as target, `idx=32`. Then the problem is, what should I do with activations of all previous tokens? Should I also subtract `ablate_feature` from them, or just leave them be. No matter which way I choose, as long as starting from the middle of the context, I got no where close to paper number. Change the starting idx to 0, meaning no previous token, and it works. 

The new understanding is about normalizing `(l1/l2)^2` by `V*T`. After few examples I understand why such normalization could upper bound the metric by 1, and such normalization makes comparison between models possible. 

After all, I got `~19%` from 32k SAE. The official number is `~13%`. Ablating residual stream channels gives `~25%`, but the official number is `60%`. For now I'll just sweep the difference under "don't know what data they are using, all I have is `openwebtext` rug."

The logical next step is training SAE from scratch. This is a good stop for consolidating what I've learned. Will read few more papers and regroup. 

-- 

Somehow I feel [Genie](https://arxiv.org/abs/2402.15391)'s `latent action model` is a key to interp world. ![](asset/lam.png)

The action latent is actually more than action. It could be whatever info that's critical for transitioning one frame to next frame. Also interesting that both encoder and decoder takes multiple previous frames, which ditches the `Markov` assumption for traditional action model. 

![](asset/dynamic_model.png)

The latent is passed into `dynamic model` in embedding form. The overall setup feels like a process to manufacture 1 specialized latent. In this paper, it's mostly used for action, but like I said, it could be anything between 2 frames.

The beauty of explicitly carved out latent is access point of control and intervention. 

If one just train a huge `dynamic model` with next video token prediction, I'm pretty sure with decent spatial temporal video encoder and VQ technique, the info that was captured separately by `latent action model` would be there in the `residual stream` of this end2end model. The problem is, with so many subspaces and tensor flow trajectories, we don't know how to locate and isolate that info. That's why we are in the business of SAE and circuit anatomy.

With an explicit `latent action model`, that latent has a clean closure, and ready to be interpreted and intervened. 

Now, the Genie paper's choice of 2d video game is smart because the majority of latent info that cause frame update is user game control, aka action. It uses environment to bound info variety in LAM. Once the setup is proven feasible, an interesting testbed is born. 

It doesn't have to be video or action. It could be inner monologue or subconsciousness between next language token prediction or next action token prediction. Feels like a `bicameral mind` could be implemented this way. An inner voice to guide AI agent when we need control. A stream of explicitly accessible subconsciousness ready to be analyzed when we need to interpret. 

I don't know ... Time to walk.

## 0704
Happy Independence day. Wake up to unmotivated/tired body and mind. Let me channel the inspiring spirit from the past and get the fuck back to work! (doom scrolling x.com)

--

`<reading note>` [A Primer on the Inner Workings of Transformer-based Language Models](http://arxiv.org/abs/2405.00208)

Linear probe learns feature predefined by human. It's a targeted operation and limited by human imagination. SAE learns many features without supervision, but the problem is making sense of them. Just realized that even though topk SAE paper is short, it's very comprehensive:
1. SAE training recipe plus one important upgrade: replace l1 with topk function to deal with shrinkage.
2. Scaling law wrt MSE. 
3. Downstream loss and flops equivalent % as quality proxies to feature. 
4. Linear probe.
5. N2G as a step of automatic SAE feature explanation. 
6. Ablated sparsity to measure downstream effects to show not only the active features are sparse, the downstream effects are sparse as well. Again a quality proxy to feature. 

--

![](asset/tok_emb.png)
- Token embedding is like `V1` of visual cortex. Each token in the codebook has an anchor in `d_model` space, optimized and ready for layer 1. 
- Last layer activation is hyper optimized for next token prediction, aka taking action. Similar to `V5/MT` of visual cortex. Even interp research won't take last layer activation. TopK SAE took layer 8 activation, 3/4 of gpt2. 
- Feeding `h_layer_last` directly to layer 1 for next transformer runtime is like pumping `V5/MT` back to `V1`. Probably won't work. In a sense, language model is actually a full stack perception to action model.
- Tokenizer plus token embedding is an easy V1 for pure natural language codebook. However, the codebook could be multimodal. For example, Spatial temporal transformer based VQVAE is literally building a `V1`, the visual codebook. 
- Early fusion of multimodal model aggregates many `[X]1` into one unified codebook. Such as vision(`V1`), audio(`A1`), action and language. 
- When perception and action are the same modality, next token prediction works very well ,ex: LLM. 
- When perception and action are different modality, ex: visual language -> motor action, what the objective should be? 
- I don't find neurosci and cognisci having compelling understandings on multimodal fusion. 

--

The relationship between `copying head`, `ov circuit` and `eigenvector` is very therapeutic.  
> Positive eigenvalues mean that there exists a linear combination of tokens contributing to an increase in the linear combination of logits of the same tokens. (paper 5.1.2)

Expand on this: 
- Eigenvalue is the scalar of eigenvector. 
- Eigenvectors are stationary directions after linear transformation of ov circuit. 
- Direction in llm model space is meaning, per linear representation hypothesis. 
- Each eigenvector with positive eigenvalue could be seen as a trigger feature live within the ov circuit.

Connecting dots: certain combination of tokens, precisely embeddings from different token track at layer L, would trigger a ov circuit feature, which would enhance the input by eigenvalue. Overall it's a way to copy/paste info from/to residual stream, hence the name copying head.

-- 

`Induction head` v. `in context learning` gives me goosebumps. 
> Finally, the emergence rate of induction heads is impacted by the diversity of in-context tokens, with higher diversity in attended and copied tokens delaying the formation of the two respective sub-mechanisms (paper 5.1.2)

ICL on LLM is like a `FPGA` being ad hoc programmed by in context input. The hope is better understanding between what's in context and the programming mechanism of these myriad circuits so that general intelligence could be interfaced by human language. This is super relevant to gemini 1.5 pro's `2m` context window.

--

The whole transformer is a giant fusion reactor of vectors. I'm very interested in this functional perspective of interpretation. 
> ... language models create vectors representing functions or tasks given in-context examples (paper 5.3)
> 
> ... multiple attention heads work together to create “function” or “task” vectors describing the task when given in-context examples (paper 5.4)

[The line between data, function and vector are getting very blurry](https://www.youtube.com/watch?v=LyGKycYT2v0&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=10). It's like the model is screaming "no~~~~ why are you trying so hard to map vector to English? Wrong medium! Can't you think in vector natively? It's better, faster, and more universal. I don't recall speaking in English that hard for me." 

I don't know where it would lead me but oh boy it's sexy. 

`</reading note>`

A productive reading day. Now ground these learnings to topk SAE.  

Every time I read, I found more depth to this deceptively simple architecture. GPT2 is outdated but still has so much to offer. 

The biggest hit today is functional perspective toward interpretation. Imagine how hard it would be to track down seemingly infinite combination of input dependent transformer circuit, or FFN neuron in super position. Instead, choose a middle layer post MLP activation, train a SAE and focus resource on making sense of them, especially the automatic interpretation route. These SAE features are more easier to interpret and intervene.

## 0705
`<reading_note>` [Circuits Updates - June 2024](https://transformer-circuits.pub/2024/june-update/index.html)

The info fill in the blank between papers and provide a source for process supervision on how frontier lab researchers think, plan and act. The curated research by other groups is a glimpse into their fucus and taste. Overall, signal/noise is off the chart. Woosa~

> The above results give us confidence that both Gated SAEs and TopK SAEs are strong alternatives to standard SAEs with little downside risk and the potential to be a meaningful improvement. However, it’s still difficult to know whether the basis found by an SAE is better or worse.

> Ultimately SAEs need to be judged on whether they provide additional insight into how the model works - can we use it to debug model issues? For steering? For finding circuits? For understanding the impact of fine-tuning? To improve robustness? It’s clear that the evaluations we have at the moment don’t get to the heart of what we care about and we’re excited to work on that and for future work from others which fills this gap.

Snippets aboves are beautiful. Improvement is not binary, but a multi-dimensional concept. They evaluate change in neutral, objective tone and keep the eyes on the true target while being fully aware of the practical necessity and limitation of proxies. 

`</reading_note>`

The critical question right now is: **SAE and learned features are promising. How to create principled evaluation and iterate faster?**

## 0706
7hr training data generation starts. Now I can relate to the mental state of `Mark Watney` during solar power charging session. One big difference: I can take a walk to a beautiful beach or park, he was stuck in a minivan size rover. Poor man...

-- 

> Look! A pair of boobs! -> (.Y.)  -- Mark Watney

I really want to have his personality. (sweeping learning rate)

-- 

![](asset/lr_sweep.png)

Take away from learning rate sweep:
1. Work hard -> faster iteration speed
2. Be bold -> take large step
3. Stay alive -> failures don't matter as long as it's not deadly

The crazy world dynamics at the moment means the performance different at 1k step is enough to learn new things unlock new doors. 
Sophisticated plan is not necessary. Once the plan is finished and perfected, the world has changed. Won't need to finish planned 80k step full run since pivots and entering new games are the norm now. This compounding reinforcement cycle is `create your own luck` in 21th century. 

The loss curve of `5E-5` will haunt me forever. 


Oh, `lr 4e-4` it is.

-- 

32k SAE is cooked. Would leave 128k to cook over night. 

## 0707
32k and 128k SAE are ready for eval. 

--

I trained the wrong model. In the open source code:
```python
class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias
```

The encoder and decoder's weight are tied forever. However, in paper:
> we initialize the encoder to the transpose of the decoder ...

Tied weights and transpose weight init are very different ideas. I guess `TiedTranspose` is for SAE ReLU baseline.  
Train again.

## 0708
Filling in training nuances:
- [x] init and renormalize columns of the decoder to be unit-norm
- [x] compute an MSE normalization constant once at the beginning of training, and do not do any loss normalization per batch.
- [x] initialize the bias b_pre to be the geometric median of a sample set of data points

Would stop here. Won't deal with these training difference for now. 
- total training token is 8 epoch of 1.31b, paper is 8 epoch of 6.4b
- We project away gradient information parallel to the decoder vectors, to account for interaction between Adam and decoder normalization.
- weight EMA
- ghost grads

Have mixed feeling about these tweaks and compensations. I'm not in a position to worry about last few percent optimization hacks yet. I bet few months later the recipe would be very different. 

-- 

Trained 32k and 128k SAE with improved training loop. 

| Eval            | homecooked 32k | OAI 32k | homecooked 128k | OAI 128k |
|-----------------|----------------|---------|-----------------|----------|
| MSE             | 0.2788         | 0.0069  | 0.4698          | 0.0054   |
| Delta Loss      | 0.9206         | 0.1336  | 0.8393          | 0.0816   |
| Ablate Sparsity | 18.93%         | 25.27%  | 16.25%          | 17.57%   |

## 0709 outro
What I learn:
- The concept of SAE, residual stream, transformer circuit and task vector.
- Experience of inspecting and manipulating transformer runtime internal state.
- Basic SAE eval technique and the field needs more, better and automated eval. 
- Basic SAE training loop.

What I miss:
- Didn't fully reproduce all eval. Passed linear probe and N2G.
- Didn't fully reproduce all training optimization. The source code even includes triton kernels. 

Still want to learn more. The model I trained is bad. The plan is to kick start iterations of:
1. Using SAE feature for research and application.
2. Train better SAE. 

I'm not committed to SAE or merch interp, but to the pursuit of understanding on how model works and use it for control and capability.

--

Next step: read [An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers v2](https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite-1) by Neel Nanda.