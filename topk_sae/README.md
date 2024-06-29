Create virtual environment, `python=3.11` and install `requirements.txt`. 

## Log
- Never dig into scaling law papers before. I can understand the conclusions but having trouble to really appreciate nuances. 
- I don't get 90% of the paper from the first read few days ago. Didn't even have general idea about SAE, and linear representation hypo and superposition hypo. After few more background papers, today on the second read, I feel way better but still lack so much background knowledge mentioned in the evaluation section. It's critical to see what's wrong right, what's missing and propose possible solutions. 
- Like first day to medical school and wow human body is complicated ... 
- Ok, made a huge logical mistake. SAE is not small AT ALL. I was an idiot and no wonder people in frontier labs are screaming engineering challenge of scaling SAE.
    - gpt2-small has `163m` params, with `d_model=768`.
    - The largest SAE has ~128k `2**17` latents, meaning ~`2**17 * 768 * 2 = 200m` params ... 128k is SMALL dictionary. 
    - I don't know `d_model` of Sonnet 3, just assume `8192`. The largest SAE in the paper is ~34m, which means ~`2**25 * 8192 * 2 = 550b` params ... WTH
- The main demo notebook for `transformer_lens` is long and packed. Let's skim through and try to catch main points as much as possible. Then divide and reproduce on the second round. 
- With `t_lens` package, I could see a path to reproduce 4.1 `downstream loss`, and 4.5 `ablation sparsity`. Don't understand the other two enough. Forget about training and scaling law for now. Focus on what's feasible in front of me.
- At this point, I've implemented gpt2 style transformer from scratch many times, and I still feel there is so much I don't understand about the model. Residual stream and autoregressive multihead attention are really ... deep. ![](../asset/unrolled_transformer.png)
- It turns out, studying merch interp really helps with understanding fundamentals about transformer and modern ai. 
- Can't imagine without `t_lens`, I'll have to Frankenstein-ize my `gpt2.py` to what extend to support such level intervention.
- First time witnessing `induction head` with close details. Wow, really gives you hope that understand a complex model is possible. I know it's just 2 circuits and simple pattern, but still love the goosebumps. 
- Reading jupyter notebook of merch interp research is like watching video of neurosurgical operation, or psychiatry session. Same wow level but big differences: controllability and reproducibility. 
> In GPT-2, 50256 signifies both the beginning of sequence, end of sequence and padding token
- I assumed gpt2 has not bos and padding for so long lol. 
- With recent interactions with Sonnet 3.5 and Neel's educational materials, I realize that `talent density` is a moat. Cross pollination and accelerated iteration create real quantum leap over mediocre environments. 
- More on `BOS`, didn't know these before. 
> - attention patterns are a probability distribution and so need to add up to one, so to simulate being "off" they normally look at the first token. Giving them a BOS token lets the heads rest by looking at that, preserving the information in the first "real" token.
> - *some* models are trained to need a BOS token (OPT and my interpretability-friendly models are, GPT-2 and GPT-Neo are not). But despite GPT-2 not being trained with this, empirically it seems to make interpretability easier.
- `BOS` is a meditation token. 'Focus on your breathe...' and don't mess with anything. Om~~~
    - Can confirm, both `om`(296) and ` om`(39030) are tokens in gpt2 code book. Probably not related to om~~~ at all. 
- Low rank and multihead attention. This is simply beautiful. 
    >As argued in [A Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html), an unexpected fact about transformer attention heads is that rather than being best understood as keys, queries and values (and the requisite weight matrices), they're actually best understood as two low rank factorized matrices.
    >* **Where to move information from:** $W_QK = W_Q W_K^T$, used for determining the attention pattern - what source positions to move information from and what destination positions to move them to.
    >    * Intuitively, residual stream -> query and residual stream -> key are linear maps, *and* `attention_score = query @ key.T` is a linear map, so the whole thing can be factored into one big bilinear form `residual @ W_QK @ residual.T`
    >* **What information to move:** $W_OV = W_V W_O$, used to determine what information to copy from the source position to the destination position (weighted by the attention pattern weight from that destination to that source).
    >    * Intuitively, the residual stream is a `[position, d_model]` tensor (ignoring batch). The attention pattern acts on the *position* dimension (where to move information from and to) and the value and output weights act on the *d_model* dimension - ie *what* information is contained at that source position. So we can factor it all into `attention_pattern @ residual @ W_V @ W_O`, and so only need to care about `W_OV = W_V @ W_O`
    >* Note - the internal head dimension is smaller than the residual stream dimension, so the factorization is low rank. (here, `d_model=768` and `d_head=64`)
- Having trouble grasping nuances between ov, qa circuit, low rank, eigenvalue and eigenvector. 
    - Like: "what happens to the eigenvectors" is a decent proxy for what happens to an arbitrary vector.
    - How does it make sense? e-vec would only scale, while general vec would go through many other transforms. 
- With `HookedRootModule` and `HookPoint`, I adopt general model inference code to support merch interp. Model codes are getting hyper specialized. A model could be trained in pure c. A version tailor to inference throughput or latency to first token. Yet another version for merch interp. It's the same set of underlying matrices. 
- The first pass is brutal but rewarding. Second pass for reproduction would be even more exciting. The third pass would be enlightening, as usual. 
- Can totally see why interp on LM could be a heaven for neuroscientist. The level of control and feasible experimentation is no comparison. If I were a neuroscientist, would definitely jump camp with no hesitation. 
- `run_with_cache` is access to activation of all hooks. Great for offline analysis. 
- `run_with_hooks` is taking action during runtime. The action, aka hook function, could be passive, simply observe the activation, do some analysis and go. It could also change the activation directly. 
- [tag] with runtime activations, static param weight as functional circuits, and tools like `t_lens` and more, merch interp is the process to study model by the loop of where to look, poke with what, why does it matter.
- I'm not familiar with the tool yet but that's not important. What's important is to build the solid scientific reasoning framework to ground intervention and making conclusions. Don't know where to poke is ok, making wrong conclusion, believe it, use it to guide future experiment and learning would be lethal.
- Eval section would be more important than architecture of training detail for me for a while. Learn to ask good question, and make logically sound conclusion first. Tools would catch up later automatically. 
- Now I know what I miss, oh boy. `activation patching`'s operation totally makes no sense to me, but the conclusion is so understandable. 
- Second pass is way better. Doing this IOI and `activation patching` work is way more software engineering and scientific method than flop power or deep math. 
