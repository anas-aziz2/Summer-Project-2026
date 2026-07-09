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
