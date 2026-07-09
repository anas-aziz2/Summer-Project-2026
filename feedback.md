# Feedback

## General

**Add a `.gitignore` for the `data/` folder.** The MNIST binary files (`data/MNIST/raw/`) are committed to the repository. Large binary data files don't belong in version control — they bloat the repo and don't need to be tracked. Add a `.gitignore` that excludes the `data/` directory.

**Fix `environment.yml`.** The file contains a Windows-specific absolute path (`C:\Users\lenovo\miniconda3\...`) as the `prefix` field, which makes it non-portable for anyone else trying to use it. Remove that line.

**Fill in the README.** Right now it only has the repo title. The README is the first thing someone sees when they open the repository — use it to briefly describe what the project is, what each notebook covers, and how to set up the environment.

---

## Week 1

### Heat Conduction Simulation

Good work setting up the finite-difference solver and the plots. A few things to think about:

**The "no for loop" comment** — ignore that one. I wrote that comment while thinking in JAX, which has `lax.scan` for sequential operations like time-stepping. In NumPy, the time loop genuinely can't be vectorized because each step depends on the previous result (`T[i]` depends on `T[i-1]`). The spatial update inside the loop is already vectorized, which is the important part.

**The geometry/discretization cell** — you already used `np.linspace`, so the "see if you can do this by creating a numpy array" note is already satisfied.

---

## Week 2

### first_try.ipynb

Looks good.

---

### gradients.ipynb

Generally looks good. A few notes:

The manual-vs-autograd comparison is exactly the right approach. Building that verification habit before relying on the framework is good practice.

One thing worth understanding explicitly: PyTorch **accumulates gradients by default**, which is why `zero_()` is needed at the end of each iteration. You got it right here, but forgetting to zero gradients in a training loop is one of the most common PyTorch bugs, so good to keep that in mind.

Before moving on, please watch the 3Blue1Brown series on deep learning. It will give you a strong intuitive foundation for MLPs and backpropagation that will make the next steps much clearer:
[Neural Networks — 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
I think Chapter 1 through 4 of the series are relevant. 

---

### trial_sin_fit.ipynb

Technically looks good. Going forward though, please include plots that explicitly show what you are observing — not just the final fit, but visualizations that support your conclusions.

---

### generalized_sine_fit.ipynb

Technically looks good, and it's great that you wrote down observations about learning rate and network width vs. depth. However, the conclusions in the comments are not backed up by any plots or analysis in the notebook. For each claim you make, there should be a corresponding figure that shows it clearly. For example:

- A plot of training loss over epochs for several different learning rates on the same axes
- A comparison of final fits (or loss curves) for networks of varying width vs. depth

More broadly: getting into the habit of documenting your observations with supporting visuals will make it much easier for us to communicate. When we move into real research, I'll need to be able to look at your analysis, understand what you're seeing, and ask questions — and well-labeled plots make that conversation a lot more productive than written summaries alone.

---

### mnist_classifier.ipynb

Good first pass at a full PyTorch training pipeline. A few things to address:

**Add a training curve.** You print the loss each epoch but never plot it. A loss-vs-epoch plot is the most basic diagnostic for any training run — please include one.

**Track test accuracy over training, not just at the end.** Your comment says "as the number of epochs increase, the test accuracy is observed to be increasing" — but there's no plot to support that. Move the evaluation inside (or alongside) the training loop and plot both train loss and test accuracy together so the claim is actually shown.

**Look at the predictions.** Show a sample of correctly and incorrectly classified digits. Seeing where the model fails is one of the most useful things you can do, and it makes the results feel more concrete. A confusion matrix would be a natural extension if you want to go further.

**Experiment with architecture size.** Your current network (784 → 32 → 10) has one narrow hidden layer. Try varying the width and depth, and plot how test accuracy changes — this connects directly to the observations you made in `generalized_sine_fit` and will give you a much richer sense of what the architecture choices are doing.

---

## On the previous feedback

Before getting into Week 3 and 4 — you did a really nice job acting on the earlier feedback. You cleaned up `environment.yml`, added the `.gitignore` for the data folder, and filled in the README. More importantly, the plotting and analysis suggestions clearly landed: Week 3 and especially `DSE_effects.ipynb` are full of exactly the kind of comparison plots and parameter sweeps I was asking for. That's great to see.

---

## Week 3

### 1D_Langevin.ipynb

This is excellent. Overlaying the sampled histogram against the analytic Boltzmann distribution is the right way to validate a Langevin simulation, and it's a great demonstration of the core idea that a noisy gradient-following process samples from a known equilibrium density. A couple of small things:

- You discard the first 10,000 samples as burn-in, which is sensible, but it's currently a magic number (maybe a guess). It's worth a quick look at the trajectory (or a running estimate) to justify that the chain has actually equilibrated.  There are good ways to check this rigorously, make sure you are familiar with them. 
- Consecutive samples in a Langevin chain are **correlated**, and this matters as soon as you want to put an error bar on anything. Your histogram is really a set of expectation values (each bin estimates the probability of landing in that bin) computed from a correlated chain. The point estimate is fine, but a naive √N error estimate will be far too optimistic, because successive samples are not independent — the effective number of independent samples is much smaller than the 3M steps you took. Please read about **blocking / reblocking** (also called the Flyvbjerg–Petersen method): you group the chain into blocks of increasing size, recompute the variance of the block means, and watch for a plateau — that plateau gives you a correlation-aware error estimate. This is a habit you'll need constantly in real research: never quote a number from a correlated simulation without an error bar that accounts for autocorrelation.

### 2D_Langevin.ipynb

Nice extension to a 2D mixture potential, and the temperature sweep is a great way to build intuition — the spreading of the density as T increases shows up clearly in the 5-panel comparison. Refactoring into a `langevin_2d(Temp)` function was the right call. One thought: at the highest temperatures the sampler explores far beyond the wells, which is why you needed the manual `set_xlim` adjustments — it might be worth a sentence noting *why* that happens physically.

---

## Week 4

### DSE.ipynb

Very nice — a working score-based generative model built from first principles, tying together everything from the previous weeks. Training a network to learn the score via denoising, then reusing your Langevin sampler to generate from noise, is exactly the idea we were building toward. The quiver plot of the learned score field over the data is a nice touch.

I think there is one bug to fix. In the training cell:

```python
for batch in loader:
    x = batch[0]          # this leaves x set to the LAST batch only

model = nn.Sequential(...)
...
for epoch in range(num_epochs):
    for batch in loader:        # you loop over batches...
        eps = torch.randn_like(x)
        x_noisy = x + sigma*eps  # ...but use the stale `x`, not batch[0]
```

The inner loop iterates over `loader`, but the body uses `x` (the leftover last batch) instead of `batch[0]`. So you're actually training on the same 100 points every step rather than the full dataset. The fix is to assign `x = batch[0]` *inside* the training loop. Good news: you already did this correctly in `DSE_effects.ipynb` — the `train_model()` function pulls `x = batch[0]` inside the loop — so the refactor fixed it. Worth correcting here too so the two notebooks agree.

**A question to think about — the empty stretches of the roll.** Take a close look at your generated samples overlaid on the Swiss roll: there are whole blocks along the roll that your samples never visit. Why do you think that is? What is it about how the score was trained, and about how the Langevin sampler explores, that would leave certain regions unvisited? Think about where your training signal actually lives, and what the sampler can and can't do in a fixed number of steps from a Gaussian start. I have some ideas, but I'd like to hear your reasoning first — it's a genuinely instructive failure, not just a glitch.

**A related next step — multiple noise scales.** This connects directly to the question above. You train with a single, small noise level (σ = 0.05). This learns the score well *near* the data manifold, but far from it (where your Langevin sampling starts, from pure Gaussian noise) the score is poorly estimated. The standard fix is annealed Langevin dynamics over a *range* of noise scales (Song & Ermon, 2019) — start with large σ to capture coarse structure, then anneal down. Worth noting where you are conceptually: right now you have a *single fixed noise level* and a *single, time-independent* score field s(x) applied identically at every step. Full diffusion / score-SDE models generalize this by conditioning the score on the noise level or time, s(x, σ) or s(x, t), with a proper noise schedule. That's the direction this is heading.

### DSE_effects.ipynb

Very well done — this is exactly the kind of systematic study I was hoping to see. Wrapping training in a single `train_model()` function and sweeping optimizer, batch size, width, depth, learning rate, and σ (with loss curves for each, and side-by-side generated samples for the σ study) is clean, reusable, and exactly how you'd want to approach this in real research. A few suggestions to take it further:

- Add a sentence or two of written takeaways under each plot, the way you did in Week 2. The plots are great; pairing them with "here's what I conclude" makes the analysis complete.
- For the σ comparison, consider quantifying sample quality rather than only eyeballing it — even a rough metric helps.
- Since the loss values differ in meaning across σ (the target `-eps/sigma` scales with σ), be a little careful comparing loss magnitudes across different noise levels directly.

---

## Follow-up on Week 3 (1D_Langevin.ipynb)

I want to call this out separately because you went back and did a really thorough job on the blocking/reblocking suggestion. A few things I especially liked:

- Starting the chain far from equilibrium (`x[0] = 5`), then using a **running-mean convergence plot** to *justify* the 10,000-step burn-in, turned a magic number into a defended choice. That's exactly the habit I was hoping to instill.
- You implemented **Flyvbjerg–Petersen** reblocking correctly from scratch, and the plateau in the error curve makes the point vividly — the effective number of independent samples is far smaller than the 3M steps.
- Putting a correlation-aware **error bar on every histogram bin** (treating each bin as an indicator-function expectation) went beyond what I asked, and it's exactly the right instinct. This is publication-grade thinking about uncertainty.

One small thing to tighten: your per-bin error picker takes the plateau from around the `argmax` of the reblocking curve. Because the error estimate itself gets noisy at large block sizes (few blocks remain), `argmax` can occasionally land on a spuriously high point. A slightly more robust choice is to look for where the curve *flattens* (successive values agree within their own uncertainty) rather than where it's maximal. Minor, but worth knowing.

---

## Week 5

Excellent work — you hit essentially every point in the plan, including the two things I flagged as most likely to trip you up.

### discrete langevin (multi-scale, both versions)

- **Noise conditioning done right.** Feeding σ in by concatenation (input dim 3) is exactly the pragmatic choice for a 2D toy — no need for anything fancier.
- **You got the σ² loss weighting.** This was the gotcha I was most worried about — without it the small scales dominate and the large-σ score (the part that fills the empty regions) never trains. You nailed it.
- **σ selection from the paper.** Choosing σ_max from the maximum pairwise distance and building the geometric ladder per the "Improved Techniques" recommendations is precisely what I was pointing you to. I also liked that you were explicit about the `_test` version *not* using this and the `_efficient` version *doing* so — that kind of self-documentation is a good habit.
- **The empty regions are filled, and you measured it.** L1 dropping from 1214 (single-scale) to ~729–740 (multi-scale) quantifies the improvement rather than just asserting it. This closes the loop on the empty-blocks question from Week 4 — well done.

Two things to take further:
- **Step size across scales.** During annealed sampling you use a *constant* η at every noise level. Song & Ermon recommend scaling the step per level (roughly η_i ∝ σ_i² / σ_L²) so the dynamics stay well-conditioned as σ shrinks. Worth trying — it often noticeably improves the low-σ refinement.
- **File naming.** `discrete langevin_test.ipynb` / `..._efficient.ipynb` have spaces and somewhat vague names. Underscores and clearer names (e.g. `multiscale_langevin.ipynb`) will save you headaches later, especially when importing or running from the command line.

### The metric itself

Your L1 density difference is a good, simple choice and the *relative* trend across methods is trustworthy. Just be aware it's binning-dependent and noisy in absolute terms — the number changes if you change bins. If you want a more principled scalar later, look at **maximum mean discrepancy (MMD)** or **energy distance**, which compare samples directly without gridding. Not necessary now, but good to have on your radar.

---

## Week 6

### reverse_sde.ipynb

This is genuinely impressive — a correct continuous-time VE SDE, from the schedule all the way through reverse-time sampling.

- σ(t) = σ_min (σ_max/σ_min)^t with a **time-conditioned** score s(x, t), and the VE diffusion coefficient g(t) = σ(t)·√(2 ln(σ_max/σ_min)) — that's the right g for a geometric schedule, and it's easy to get wrong. You derived/matched it correctly.
- **Reverse-time Euler–Maruyama** with initialization from N(0, σ_max² I): the signs, the g² factor on the score term, and the g√dt on the noise all check out.
- And the three-way comparison lands the whole arc: **1214 → ~729 (multi-scale) → 606 (continuous SDE)**. A clean, monotonic, *measured* improvement across the three samplers is exactly the deliverable I wanted.

Where to go from here: rather than doing more on the 2D toy, we're going to make the jump to a **physical system** next — that's where the real interest of this project lives, and you've clearly extracted the core lessons from the Swiss roll. A couple of things carry forward:
- **Written derivation (do this now).** Pair your SDE code with a short written derivation of the reverse-time SDE (following Anderson) so the math and the implementation sit side by side. This is what we'll build on at the whiteboard, and it's cheap to do while it's fresh.
- **Predictor–corrector (coming soon, as a tool).** You have the plain reverse-SDE (predictor) sampler. Adding Langevin "corrector" steps (Song et al.'s PC sampler) is a natural extension — but hold off doing it on the toy. On the higher-dimensional, multimodal physical distribution we're moving to, the corrector is often what makes sampling work *at all*, so we'll introduce it there where the need is real.
- **Probability-flow ODE (later).** The deterministic ODE with the same score gives another sampler and, importantly, *exact likelihoods* — which for a physical system means access to **free energies**. Worth keeping on your radar for down the road.

Overall: you've gone from a single fixed noise scale to a continuous-time score SDE, validated at each step with a consistent metric. That's the full conceptual arc of modern score-based generative modeling, and you built it from first principles. Really nice work — and a great launchpad for moving to a real physical system.
