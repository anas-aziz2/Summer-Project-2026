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

---

## Week 7

### Lennard_Jones_13.ipynb

Good, solid start on the physical system — and given that you were learning Monte Carlo essentially from scratch this week, this is decent progress. The LJ potential and the Metropolis move are implemented correctly, and I'm especially pleased that you carried your Week 3 tools forward: the running-mean burn-in diagnostic and the Flyvbjerg–Petersen reblocking both reappear here, applied to the energy time series. That transfer of habits is exactly what I was hoping to see.

Below is a fair amount, but most of it is one connected story about **two different quantities that are easy to confuse**. Keep that spine in mind as you read: (i) how *wide* the energy distribution physically is, and (ii) how *well you've measured its mean*. Almost everything below is one or the other.

**1. Acceptance ratio.** Your 29.75% is fine — but as a rule of thumb we aim for ~50%, with roughly 25–65% being acceptable in practice. You're on the low side, which means your moves are a touch too large; shrinking the maximum displacement (currently 0.07) will push acceptance up toward 50%. Try tuning it and watch the acceptance respond.

**2. Variance vs. standard deviation vs. standard error — make sure this distinction is crystal clear.** These get conflated constantly, and they mean genuinely different things:
- The **variance** and its square root the **standard deviation** are *physical properties of the distribution* — they describe how wide your energy histogram genuinely is. As you collect more samples, these do **not** shrink; they converge to a fixed, temperature-dependent value.
- The **standard error of the mean** ($\sigma/\sqrt{N_\text{eff}}$) describes *how well you have pinned down the mean* $\langle U\rangle$. This **does** shrink as you sample more, like $1/\sqrt{N}$.

A good exercise: plot all three (variance, std dev, std error) as a function of the number of samples on the same figure. You should see two of them plateau and only one — the standard error — decrease. Note the $N_\text{eff}$, not $N$: because your samples are correlated (see point 4), the effective number of independent samples is much smaller than the raw count, and that's what actually sets your error bar.

**3. What the mean and the variance should be — and why you never see the ground state.** First, a note on notation so we're consistent: the quantity you sample is the *potential* energy, which I'll call $U$ throughout (there's no kinetic energy in configurational Monte Carlo). And I work in reduced LJ units — energies in units of $\varepsilon$ and $k_B = 1$, so temperature is measured in energy units — matching your $\varepsilon = 1$ in code. That's why $k_B$ doesn't appear in the equipartition formulas just below; to restore physical units there, replace each $T \to k_B T$.

Here's the satisfying part: in the harmonic (low-temperature) limit, *both* the mean and the width of your energy histogram are predictable from the temperature alone, via equipartition. With $d = 3N-6 = 33$ vibrational modes (the 3 translations and 3 rotations of the whole cluster carry no energy, hence $3N-6$, not $3N$):

$$\langle U\rangle - U_\text{min} \approx \tfrac{d}{2} T \qquad \mathrm{Var}(U) \approx \tfrac{d}{2} T^2$$

For LJ-13 at $T=0.10$, with $U_\text{min} = -44.327$, this predicts a mean of $\approx -42.68$ (you got $-42.53$) and a standard deviation of $\approx 0.41$ (you got $0.44$). Both match well — and the fact that yours sit *slightly* above the harmonic prediction is the first fingerprint of anharmonicity. **Predict these two numbers from $T$ and overlay them on your histogram** — it's a great validation of your sampler. (The general statement is $\mathrm{Var}(U) = T^2 C_v$ — with $k_B$ restored, $\mathrm{Var}(U) = k_B T^2 C_v$ — i.e. energy fluctuations *are* the configurational heat capacity. The harmonic result is just $C_v = d/2$.)

Now the deeper point. Notice that your lowest sampled energy is $-43.90$, but the true global minimum (the Mackay icosahedron) is $-44.3268$. **Look up that reference value and check against it.** You never reach it — and you *shouldn't* expect to, for a beautiful reason. The energy distribution is the product of the density of states (entropy) and the Boltzmann factor:

$$P(U) \propto g(U) e^{-U/T} \qquad g(U)\propto (U-U_\text{min})^{d/2-1}$$

which is a Gamma distribution that *vanishes* at $U = U_\text{min}$. So although the Boltzmann factor *per configuration* is largest exactly at the global minimum (for any temperature), the ground state is a "lonely island": it is a single point of essentially zero phase-space volume. It has the highest probability of any individual microstate in the universe, yet the *energy* histogram peaks well above it because there are so many more configurations at higher energy. This is the microstate-vs-macrostate distinction in action, and it's worth sitting with. **Experiment:** rerun at lower temperatures and watch $\langle U\rangle$ move toward $U_\text{min}$ as the harmonic formula predicts — but note you'll still never sample $U_\text{min}$ exactly, and that acceptance will collapse at low $T$ (the walker gets stuck), which is itself an instructive difficulty.

**4. Autocorrelation and the sampling interval.** You reblocked the energy, which is great, but two things to add:
- **Make an autocorrelation plot.** Plot the normalized autocorrelation $C(\tau)$ of the energy and extract the integrated autocorrelation time $\tau_\text{int}$. This gives you the correlation time directly, the effective sample size $N_\text{eff} = N/(2\tau_\text{int})$, and an independent cross-check on your reblocking plateau. These are complementary views of the same thing.
- **Justify your save interval from it.** You currently save a configuration every 2000 steps, and your reblocking suggests the energy correlation time is *also* roughly 1000–2000 steps — so your saved configurations are separated by only about one correlation time and are still mildly correlated. Use the *measured* correlation time to *choose* the interval, rather than picking it by hand. This closes the loop on exactly why we learned reblocking in the first place.

  On the reblocking plot itself: plotting block size on a $\log_2$ axis is exactly right (each step doubles the block). It plateaus at a fairly large block size because you reblocked the *per-step* series, so block size is measured in single MC steps and the correlation time is ~1000s of steps (only 1 of 13 atoms moves per step, so the configuration decorrelates slowly) — that's sensible, not surprising. Don't over-interpret the far right of the curve: with few blocks left, the error estimate itself becomes noisy (its own uncertainty grows like $1/\sqrt{2(N_\text{blocks}-1)}$).

**5. Efficiency — the classic Monte Carlo speedup you're missing.** You recompute the *full* $O(N^2)$ energy at every step. But a trial move displaces only *one* atom, so the energy *change* $\Delta E$ only involves that atom's pair interactions — an $O(N)$ incremental update. Computing $\Delta E$ directly (rather than $E_\text{new} - E_\text{old}$ from two full evaluations) is ~13× faster here and much more for LJ-38. This is the single most valuable practical MC lesson before you scale up, so please refactor to incremental updates.

**6. Reproducibility.** The notebook reads `LJ13.xyz`, but that file isn't committed to the repo (it's not in `data/` either), so the notebook won't run as pushed. Please add the starting-structure file (or generate it in-notebook) so the analysis is reproducible.

Genuinely nice work for a first MC week — the fact that you reused your own reblocking machinery on a brand-new system is the kind of thing that will serve you throughout research.

---

## Week 8

This was a big week and you handled it really well. Two things up front: first, you went back and addressed essentially the *entire* Week 7 feedback list — I'll acknowledge that below because it's worth it. Second, the score-model baseline behaves exactly the way we expected a non-invariant baseline to behave — it "fails" in the instructive way, which is the point. Let's go through both.

### Cleaning up Week 7 (Lennard_Jones_13.ipynb)

You closed out almost every open item, and did it properly:

- **Incremental $\Delta E$** — you added `atom_energy(...)` and now update only the moved atom's contribution ($O(N)$ instead of $O(N^2)$). That's the classic MC speedup, and it's exactly why you were able to push to $10^8$ steps. Nicely done.
- **Acceptance tuning** — dropping the max displacement from 0.07 to 0.043 put you at **49.94%**, essentially dead on the 50% target. Good.
- **Autocorrelation** — you implemented $C(\tau)$, plotted it (with a zoom), and computed the integrated autocorrelation time $\tau_\text{int}$. This is the complementary view I was hoping for.
- **Variance / std-dev / std-error vs. $N$** — you built the plot and drew the right conclusion: variance and standard deviation plateau, only the standard error keeps falling. That distinction is now demonstrated, not just asserted.
- **Harmonic overlay** — you coded up $\langle U\rangle = U_\text{min} + \tfrac{d}{2}k_B T$ and $\mathrm{Var} = \tfrac{d}{2}k_B T^2$, overlaid the Gaussian, and correctly attributed the small deviation to anharmonicity. Your numbers (mean $-42.53$, std $0.44$) sit right on the predictions.
- **Reproducibility + units** — `LJ13.xyz` is committed, the dataset is saved, and you added `k_b = 1.0` explicitly. 

One thing to reconcile (a good learning point, not a mistake): your three correlation diagnostics don't quite agree. You get $\tau_\text{int} \approx 134$, but you read the reblocking plateau at block size $\approx 4000$ and say the autocorrelation decays by lag $\approx 1000$–$2000$. Those should be roughly consistent. The likely culprit is that you sum $C(\tau)$ all the way out to lag 5000, which accumulates a lot of tail *noise* into $\tau_\text{int}$ — the standard fix is an automatic windowing/truncation (look up "Sokal windowing"): stop summing once the autocorrelation is buried in noise. It doesn't hurt you in practice — saving a configuration every 4000 steps when $\tau_\text{int}\approx 134$ gives very well-decorrelated data — but it's worth understanding *why* the three numbers should line up and getting them to.

### The baseline score model (LJ_13_Baseline_Score_Model.ipynb)

The structure here is exactly right, and I'm pleased you carried forward two things we discussed:
- You **brought the predictor–corrector sampler forward** from the plan, and — importantly — you *noticed and documented* that the corrector blew up and needed the step size clamped. That instinct (watch for the explosion, diagnose it, note it honestly) is exactly right. The blow-up itself is informative: it usually means the corrector step is too aggressive relative to the score magnitude, which ties into the SNR calibration.
- Your evaluation is honest and quantitative: energy histograms, pairwise-distance distributions, and the headline number that only **~5.4% of generated configurations have $U < 0$**. That's the expected outcome of a non-invariant baseline, and *measuring* the failure rather than hand-waving it is the right scientific move.

Now the things to fix or think about:

1. **There's a bug in your energy-comparison plot.** In the cell where you build `training_energies`, you compute it from `configs_generated`, not from the Monte Carlo `configs`. So in the comparison histogram, the curve you've labeled "Monte Carlo" is actually your *generated* data plotted against itself — the figure isn't showing what the label claims. Change that line to use the MC `configs` and the comparison will be meaningful. (Your pairwise-distance comparison a few cells later *does* use `configs` correctly, so it's just that one line.)

2. **Your `sigma_max = 10` hack and the center-of-mass are the same story.** You capped $\sigma_\text{max}$ at 10 because the max pairwise distance came out around 82. That 82 is not physical structure — it's because the raw configurations aren't centered, so the whole cluster has random-walked around in absolute space, and the distance between two configuration *vectors* is dominated by where the cluster happens to sit, not by its shape. The correct fix is exactly what you started doing in the Modified notebook: **remove the center of mass.** Once you do, the data lives in a compact region, $\sigma_\text{max}$ from the max pairwise distance becomes sensible, and you won't need the hack. Worth connecting those two dots explicitly — the symptom (huge $\sigma_\text{max}$) and the cure (COM removal) are the translational-invariance lesson in action.

3. **Guard `lj_energy` against collapsed atoms.** For generated configs where two atoms nearly coincide, $r\to 0$ makes $(\sigma/r)^{12}$ overflow. Your $U<0$ filter sidesteps it, but a small guard (or clipping $r$) will keep the numbers clean and avoid `inf`/`nan` surprises.

### Where you are, and the road ahead

You've done exactly what Week 8 asked: a working non-invariant baseline, evaluated against the Week 7 physical criteria, with the predictor–corrector brought in — and the failure it exhibits (5.4% physical) is precisely the motivation for the symmetry work. And you've *already* started that in the Modified notebook by handling translation. The next steps are rotation and, the hard one, permutation — which is where the equivariant (EGNN-style) architecture comes in. Take your time with the EGNN; it's the conceptual heart of this part of the project, and it's worth understanding deeply rather than rushing. Really strong week.

## Week 9

Good job - you built a working equivariant score model, and we quickly see how this helps the physics. The failure mode we spent Week 8 diagnosing is essentially gone. You also went back and closed out the Week 8 list, again. Very well done.

### Closing out Week 8

- **The `training_energies` bug** — fixed, and I appreciate the note you left in the notebook confirming it. 
- **`sigma_max` and the center of mass** — you did this the right way round. Instead of patching the number, you removed the COM and *re-derived* it: the max pairwise distance drops from $\approx 82$ to $6.44$. Then you went back and retro-fitted $\sigma_\text{max} = 6.44$ into the baseline notebook too, so the two models are now compared on equal footing. That is exactly the right instinct.
- **Sokal windowing** — implemented, giving $\tau_\text{int} \approx 129.6$ with a window of 648. And you were honest that this still doesn't reconcile with the blocking plateau. We will come back to this.

### The equivariant score model (LJ_13_Modified_Score_Model.ipynb)

This is the hard, central piece of the whole project, and you got it working! The EGNN structure is right: you build displacements $\mathbf{r}_{ij} = \mathbf{x}_i - \mathbf{x}_j$, feed only the *squared distances* into the message MLP, and update coordinates as $\mathbf{r}_{ij}$ times a learned scalar. That is precisely the construction that provides rotational equivariance, and summing over neighbours gives permutation equivariance. You got the architecture right.

The results back it up:

- **Essentially every generated configuration is now physical.** The baseline gave $U < 0$ for about 5.4% of samples. Here the generated energies run from $-43.5$ to $-33.4$, against a training range of $-43.8$ to $-40.3$, with means of $-41.98$ versus $-42.53$. That is a very different situation from what we were seeing before.  You can see the symmetry argument paying off.
- **The pairwise-distance distributions genuinely overlay very well.** You were right to say so.
- **You added a diagnostic I hadn't asked for**, comparing the learned score directly against $-\beta \nabla U$: cosine similarity $0.970$, with norms of $170.4$ (learned) against $224.2$ (true). Inventing your own validation metric is a good step towards independence, and this particular metric is well chosen — it separates *direction* from *magnitude*, which turns out to matter a lot.

A few things to think about.

**1. Your loss function and your architecture disagree about what the natural variable is.**

As you noted, your direction is excellent (cosine $0.97$) but your magnitude is systematically 24% low. That is not a coincidence, and it is not a tuning problem — it is telling you something structural about how you parametrized the network. 

In your training loop you draw a clean configuration $\mathbf{x}$, a time $t \sim U(0,1)$, and a noise level $\sigma(t) = \sigma_\text{min}(\sigma_\text{max}/\sigma_\text{min})^t$ running from $0.01$ to $6.44$. You then draw Gaussian noise (your `noise` variable) and form the corrupted configuration

$$\tilde{\mathbf{x}} = \mathbf{x} + \sigma \mathbf{z}, \qquad \mathbf{z} \sim \mathcal{N}(0, I)$$

*A notation warning:* Here I am calling the noise $\mathbf{z}$, not $\epsilon$. In most diffusion papers it is written $\epsilon$ as you have done here — but remember that here we are working in reduced Lennard-Jones units, where $\epsilon$ is the well depth that sets your energy scale. If you write both in the same document, there will be a notation issue. So I suggest picking $\mathbf{z}$ and staying with it.

Since $\tilde{\mathbf{x}} \mid \mathbf{x} \sim \mathcal{N}(\mathbf{x}, \sigma^2 I)$, the conditional score is

$$\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = -\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2} = -\frac{\mathbf{z}}{\sigma}$$

which is your `target = -noise / sigma_expanded`. That line is correct. It works because the per-sample target is extremely noisy, but its conditional mean is exactly the score of the smoothed density, so training against it is unbiased.

The problem: your configuration space has $D = 3 \times 13 = 39$ dimensions, so $\|\mathbf{z}\| \approx \sqrt{39} \approx 6.2$ and

$$\|\text{target}\| \approx \frac{\sqrt{39}}{\sigma}$$

At $\sigma = 6.44$ that is about $0.97$. At $\sigma = 0.01$ it is about $624$. **Your network is being asked to produce outputs spanning a factor of 644, and the only thing telling it which regime it is in is a single scalar fed in as a feature.** This is a very hard thing for a network to learn.  A stack of SiLU layers has no built-in mechanism for that. What makes it even harder, you initialize the final `coord_mlp` layer at gain $0.001$ — sensible for stability, but it means the network starts near zero and has to discover that entire dynamic range on its own.

The fix is to hand the network the divergence analytically:

$$s_\theta(\tilde{\mathbf{x}}, \sigma) = \frac{f_\theta(\tilde{\mathbf{x}}, \sigma)}{\sigma}$$

Now $f_\theta$ is chasing $-\mathbf{z}$, which has unit variance at *every* noise level. Same network, same loss, but the hard part is built in, instead of learned by gradient descent. (While you are there: you feed raw $\sigma$ as a feature, but your own schedule is geometric in $\sigma$ — so $\log \sigma$ is the natural input. Small change, same spirit.)

BTW -- I think you already had a sense of this issue on your own.  I say this because you have already written it down without noticing. Look at your own weighting:

```python
loss_per_sample = ((score_pred - target) ** 2).sum(dim=(1,2))
weights = sigma.squeeze() ** 2
loss = (weights * loss_per_sample).mean()
```

That $\sigma^2$ is correct and important. Without it, the loss at $\sigma = 0.01$ is of order $39/\sigma^2 \approx 3.9 \times 10^5$ while at $\sigma = 6.44$ it is of order $1$ — so the gradient would be *entirely* determined by the smallest noise levels and the large-$\sigma$ end would never train at all. But watch what the weight actually does algebraically:

$$\sigma^2 \left\| s_\theta + \frac{\mathbf{z}}{\sigma} \right\|^2 = \left\| \sigma s_\theta + \mathbf{z} \right\|^2$$

Your weighted loss is *exactly* an unweighted squared error on the quantity $\sigma s_\theta$ against the target $-\mathbf{z}$. In other words: **your loss function already decided that the natural object is $\sigma s_\theta$, not $s_\theta$.** The $\sigma^2$ weight is that statement. But your architecture still asks the network to represent $s_\theta$ directly. Setting $f_\theta \equiv \sigma s_\theta$ simply makes the network parametrize the same thing the loss is already scoring. The two halves of your code currently disagree, and the 24% is (most probably) where that disagreement surfaces — note that you measured it at $\sigma_\text{min}$, the very hardest end of that 644-fold range.

One free bonus falls out of this algebra. Since the loss is now $\|\sigma s_\theta + \mathbf{z}\|^2$ and $\mathbb{E}\|\mathbf{z}\|^2 = D = 39$, a model that outputs *identically zero* scores exactly $39$. So your loss has an absolute scale:

- $39$ means the model has learned nothing
- your converged $\approx 15.3$ means you are capturing roughly **61%** of the noise variance

That is far more useful than "the loss went down," and it gives you a fixed yardstick. Re-run with the $1/\sigma$ parametrization and see where that number lands. My prediction is that it improves, and that it improves *preferentially at small $\sigma$* — so break your loss down by noise level (bin it in $\log \sigma$) and check. If I am right, the norm discrepancy should shrink along with it. If the loss barely moves but the norm error does, that is interesting too, and I would want to know.

**2. Your score is not center-of-mass free.**

This is a subtle point, and *not* a bug but more of a design choice. Your training is internally consistent. It is a design choice that is costing you something. The reason I want to spend space on it is that it illustrates a nice conceptual point.

Translational symmetry has **two** consequences for the score, and you secured one, but not the other:

1. **The score field is translation invariant:** $s(\mathbf{x} + \mathbf{a}) = s(\mathbf{x})$. You have this guaranteed, because only the differences $\mathbf{x}_i - \mathbf{x}_j$ ever enter your network.
2. **The score output sums to zero:** $\sum_i s_i = 0$. You do not have this.

It is very natural to assume the second follows from the first, but it doesn't. Here is why — your coordinate update is

$$\Delta \mathbf{x}_i = \frac{1}{N} \sum_j (\mathbf{x}_i - \mathbf{x}_j) w_{ij}$$

where the scalar weight $w_{ij}$ is what your `coord_mlp` returns for the pair $(i,j)$, from the inputs $[h_i, h_j, d_{ij}^2, \sigma]$.

Sum that over $i$, then relabel $i \leftrightarrow j$ in the same sum and average the two versions:

$$\sum_i \Delta \mathbf{x}_i = \frac{1}{2N} \sum_{ij} (\mathbf{x}_i - \mathbf{x}_j)(w_{ij} - w_{ji})$$

So the net translation vanishes **if and only if $w_{ij} = w_{ji}$**. But your `coord_mlp` receives $[h_i, h_j, d^2, \sigma]$ in that order, which is not symmetric under exchanging $i$ and $j$.

Now look at what that implies layer by layer (this is the fun part). In your **first** layer, every $h_i$ is identical — you build them with `torch.ones(...)` through a shared embedding — so $w_{ij}$ depends only on $d_{ij}^2$, which *is* symmetric. Layer 1 is exactly COM-free. From the **second** layer onward, the $h_i$ have been updated by per-atom aggregated messages and now differ from each other, so $w_{ij} \neq w_{ji}$ and the sum is no longer zero.

That is a prediction about your own code, and I would like you to test it. Print

```python
score.mean(dim=1).norm()      # compare against score.norm()
```

for your trained 3-layer model, then rebuild with `num_layers=1` and print it again. The prediction is that it is nonzero for the 3-layer model and drops to machine precision for the 1-layer one. If that is what you see, you have just confirmed a statement about a neural network by doing algebra on paper, which is a genuinely satisfying thing to be able to do.

**Why it is not a bug.** Your data lies exactly on the 36-dimensional plane $\sum_i \mathbf{x}_i = 0$ after you center it, but you add isotropic noise in all 39 dimensions, which pushes the noisy configurations off that plane. The smoothed density then factorizes as

$$p_\sigma(\tilde{\mathbf{x}}) = q(\tilde{\mathbf{x}}_\perp) \cdot \mathcal{N}(\mathbf{c}; 0, \sigma^2/N), \qquad \mathbf{c} = \frac{1}{N} \sum_i \tilde{\mathbf{x}}_i$$

so the exact score genuinely *does* have a COM component, equal to $-N\mathbf{c}/\sigma^2$, whenever $\mathbf{c} \neq 0$. Your network needs one to fit those points. Everything is consistent.

The wasted part is that this component is **analytically known**. You are spending network capacity learning $-N\mathbf{c}/\sigma^2$, a formula we can just write down. And note it vanishes exactly at $\mathbf{c} = 0$ — which is where your centered training data sits and where you ran your score comparison.

**The cleaner design** is to work entirely inside the COM-free subspace and never leave it. Three one-line changes:

- project the noise when you draw it: `z = z - z.mean(dim=1, keepdim=True)`
- project the score at the end of `ScoreNetwork.forward`, the same way
- draw the prior COM-free, rather than `torch.randn(n_samples, 13, 3) * sigma_max`

Note that this is standard practice in molecular diffusion models — look up the "zero center-of-mass subspace" in Hoogeboom et al.'s equivariant diffusion paper (EDM), which handles it exactly this way.

You will notice that your sampler is *already* patching this by hand: you call `x = x - x.mean(dim=1, keepdim=True)` after every predictor step and every corrector step. That works, but you built the workaround before fully recognizing what it was working around. Doing the projection properly means you no longer need those lines.

Two connections here. First, this is the same issue as your **prior**: if the model lives in the COM-free subspace, the noise you start sampling from should live there too, and that subspace has $3(N-1) = 36$ dimensions, not 39. Second, it feeds straight back into item 1 — if $\mathbf{z}$ is COM-free then $\mathbb{E}\|\mathbf{z}\|^2 = 36$, so the "learned nothing" reference value for your loss becomes **36 rather than 39**. 

One thing to note: this does not explain your 24% norm deficit. A spurious COM component would make the learned norm too *large*, not too small. So item 1 is still the main explanation for that. These are two independent findings that happen to both live in the same function.

**3. You haven't actually tested that your network is equivariant.**

I'm making this point for the sake of developing good habits.Since our entire argument is that equivariance fixes the physics, you want to explicitly test/show that this is true.  All of your evidence for equivariance is *indirect* -- the energies improved, the distance histograms overlay. Both of those are consistent with a correctly equivariant network. But they are also consistent with a network that has, say, a sign error in `diff` and simply trains well anyway. You do not currently have any direct evidence that the network is equivariant.

The direct test is about five lines, and for a notebook whose whole thesis is a symmetry claim, it is important. Please add it.

What I would assert, reporting each as a **relative** error, $\|s(g \cdot x) - g \cdot s(x)\| / \|s(x)\|$:

- **Rotation:** $s(xQ) = s(x)Q$ for $Q \in SO(3)$. Expect $\sim 10^{-6}$ in float32.
- **Translation:** $s(x + \mathbf{a}) = s(x)$. Expect $\sim 10^{-7}$.
- **Permutation:** $s(Px) = P s(x)$ for a random permutation $P$. Expect $\sim 10^{-7}$.
- **Center of mass:** $\sum_i s_i = 0$. This one should *fail*, per item 2 — that is the point.

```python
Q = torch.linalg.qr(torch.randn(3,3))[0]
Q = Q * torch.sign(torch.det(Q))          # ensure a proper rotation, det = +1
rel = (model(x @ Q, sigma) - model(x, sigma) @ Q).norm() / model(x, sigma).norm()
```

Two practical notes. Generate $Q$ properly — a random matrix is not a rotation, so QR-decompose and fix the determinant to $+1$ (or use `scipy.spatial.transform.Rotation.random()`). And keep $\mathbf{a}$ modest in the translation test: a very large offset destroys precision in $\mathbf{x}_i - \mathbf{x}_j$ through float32 cancellation, and you would misread that as a broken model.

**The conceptual point I most want you to take:** equivariance is a property of the *architecture*, not of the training. It holds at random initialization, before you have taken a single gradient step. So run this test on an **untrained** model. That cleanly separates "did I implement the architecture correctly" from "did the training work" — and those are exactly the two things your current indirect evidence doesn't distiguish. 

It is also a real bug-catcher for common typos like a transposed `diff`, a swapped `hi`/`hj` expansion, an aggregation over the wrong axis. You very likely have none of these, since your model works. But you do not currently *know* that, and when we move to, e.g., LJ-38 you will not be able to distinguish a quiet indexing bug from a genuinely hard sampling problem. 

**A bonus you may not have noticed: you get reflections for free.** Because only $d_{ij}^2$ and $\mathbf{x}_i - \mathbf{x}_j$ enter your network, it is equivariant under the *full* orthogonal group $O(3)$, not merely $SO(3)$. Inversion sends $\mathbf{r}_{ij} \to -\mathbf{r}_{ij}$, leaves $d_{ij}^2$ untouched, and flips the sign of the coordinate update, so $s(-x) = -s(x)$ exactly. Your model cannot tell a cluster from its mirror image -- fine here.  Interesting if we ever work on chiral materials.

For Lennard-Jones that is exactly **right** — the potential depends only on distances, so the true Boltzmann distribution really is mirror symmetric, and distinguishing enantiomers would be wasted capacity. But it is worth knowing that this is a genuine limitation for chiral systems, and it is an active discussion point in the EGNN literature. It costs you nothing here, and could be a real bug in a different problem.

**Finally, a key number that really illustrates the benefit of equivariance.** Permutation invariance means your network treats all $13! = 6{,}227{,}020{,}800$ relabelings of the atoms as a single configuration. Your baseline model had to learn that equivalence from 25,000 samples — and learn all of $O(3)$ on top of it. That is hopeless, and it is why the baseline produced $U < 0$ only 5.4% of the time. The EGNN gets all of it by construction, for free, before training starts.

I think that factor of $6.2 \times 10^9$ is the clearest way to see *why* the equivariant approach worked. It turns "symmetry is good practice" into an arithmetic statement about how much of configuration space your training set was being asked to cover on its own. 

**One small thing while you are in there.** In `ScoreNetwork.forward` you build `atom_features = torch.ones(x.shape[0], x.shape[1], 1)` with no `device=` or `dtype=`. That is fine on CPU, but it will throw a device-mismatch error the moment you move to a GPU — which you will want for LJ-38. Build it with `torch.ones_like(x[..., :1])` instead and it will follow the input automatically.

**4. Use the global minimum as a thermometer - you can use it to explain your energy shift.**

LJ-13 has a known global minimum: the icosahedron, at $U_\text{min} = -44.3268$ in reduced units. (This is a standard benchmark that we looked up the Cambridge Cluster Database, and Wales and Doye's work on LJ clusters.) Draw that as a vertical line on your energy histograms. It costs one line of code and it converts both of your distributions from "curves that roughly overlap" into quantitative statements about physics.

What you can measure: A 13-atom cluster has $3N - 6 = 33$ vibrational modes, so equipartition gives $\langle U \rangle = U_\text{min} + \frac{33}{2} T$ — the relation you already coded up in Week 7. Invert it, and any mean energy becomes an **effective temperature**:

| Ensemble | $\langle U \rangle$ | $T_\text{eff}$ |
|---|---|---|
| Your MC training data | $-42.526$ | $0.109$ |
| Your generated samples | $-41.976$ | $0.1425$ |
| Nominal | — | $0.100$ |

Two things fall out immediately. First, your training data reads $0.109$ rather than $0.100$; that 9% is anharmonicity, the same effect you already saw in Week 7 when your measured standard deviation ($0.44$) came out above the harmonic prediction ($0.406$). Good — the two observations are consistent.

Second, and more useful: **your generated ensemble is about 30% too hot** relative to properly calibrated training data. That is a much sharper and more precise statement than "the histograms do not quite match," and unlike a visual comparison it is a single number you can track as you make changes.

**Something to think through.**

Suppose your learned score is systematically too small by a uniform factor, $s_\theta \approx \alpha \nabla \log p$ with $\alpha < 1$. Langevin dynamics driven by that score does not sample $p$. It samples the distribution $q$ satisfying $\nabla \log q = \alpha \nabla \log p$, which means $q \propto p^{\alpha}$. And for a Boltzmann distribution $p \propto e^{-U/T}$, that is

$$q \propto e^{-\alpha U / T} = e^{-U/(T/\alpha)}$$

which is just *another Boltzmann distribution, at a higher temperature*: $T_\text{eff} = T/\alpha$.

You measured $\alpha$ yourself. It is $170.43 / 224.19 = 0.760$. So the prediction is

$$T_\text{eff} = 0.109 / 0.760 = 0.1434$$

and what you actually observe is $0.1425$. That is agreement to better than 1%.

Think about that. A deficiency you measured with one diagnostic — the norm of the score at $\sigma_\text{min}$, from item 1 — predicts, through two lines of statistical mechanics, the energy shift you measured with a completely independent diagnostic. The score parametrization issue from item (1) is very likely the explanation for the physics discrepancy you observed.

Some caveats: you measured $\alpha$ only at $\sigma_\text{min}$ and only on training configurations, the harmonic $d = 33$ is an approximation, and your predictor-corrector sampler is not a pure Langevin chain at fixed $\alpha$. So landing within 1% is partly luck. But the mechanism is  right, and it gives you a concrete prediction to test: **fix the $1/\sigma$ parametrization from item 1, and $T_\text{eff}$ should fall from $0.1425$ toward $0.109$.** Please report that number when you re-run.

**And now the correlation-time question, which I think we can close.**

We have been thinking through this since Week 8: $\tau_\text{int} \approx 130$, a blocking plateau at block size $\approx 4000$, and an autocorrelation that looks like it reaches zero by lag 1000-2000. We concluded these disagree, but on further thinking actually I do not think they disagree.

The blocking plateau does **not** appear at $B \approx \tau_\text{int}$. The bias in the blocked variance decays as $O(\tau_\text{int}/B)$, so a visually flat plateau requires $B \approx 20$ to $50$ times $\tau_\text{int}$. With $\tau_\text{int} = 130$, that predicts a plateau somewhere between about 2600 and 6500. You read it at 4000, right in the middle of that range.

Your autocorrelation plot agrees too. For $\tau \approx 130$, $C(1000) = e^{-7.7} \approx 5 \times 10^{-4}$. "Reaches zero around 1000-2000" is simply where an exponential with $\tau = 130$ falls below the noise in your plot. All three diagnostics have been telling you the same thing.

You can settle this with arithmetic on numbers you have **already printed**. The blocking plateau error should exceed the naive ($B = 1$) error by exactly

$$\sqrt{2 \tau_\text{int}} = \sqrt{260} \approx 16.1$$

You already have the `for b,e in zip(block_sizes, errors)` loop in the notebook. Take the plateau error, divide by the first error, and see whether you get about 16. If you do, the two methods agree exactly and we can close the outstanding question.

One caution: $\tau_\text{int}$ and your block sizes are both in units of *recorded samples*, while your configurations are saved on a stride in MC steps. State that stride explicitly in the notebook. Mixing "per step" with "per recorded sample" is exactly how people manufacture spurious factor-of-1000 disagreements for themselves.

### Where you are, and where we go next

You built the equivariant model, and it worked — and the reason it worked is not mysterious. Permutation invariance alone hands your network $13! \approx 6.2 \times 10^9$ configurations for the price of one, with all of $O(3)$ on top. Your baseline was trying to learn that from 25,000 samples.

What is most satisfying is that the remaining discrepancy is not mysterious either. Your model is 24% low on the score magnitude, and it produces an ensemble 30% too hot, and those two numbers are the same fact viewed from two directions. That is a rare and good position to be in: you know what is wrong, you know why, and you have a specific prediction to test.

So for next week, in priority order:

1. The $1/\sigma$ parametrization (item 1), then report the loss against its $36$ baseline and the new $T_\text{eff}$.
2. The COM-free subspace (item 2), including the one-layer test.
3. The symmetry test suite (item 3) — run it on an untrained model.
4. The $U_\text{min}$ line and the $\sqrt{2\tau_\text{int}}$ check (item 4).
5. Look up the two LJ-38 funnel structures and put images of them in your notebook — see below for what to look for.

**LJ-38 is where we go next, in the fall.** It is the opening problem of the research project rather than a last week of summer, because doing it properly means implementing and tuning replica exchange, regenerating the dataset, retraining, and checking that *both* funnel populations come out with the right relative weight. That is a month of work, and it deserves to be done at its own pace. What is left of the summer goes into making LJ-13 genuinely finished, plus the talk and getting the report started.

The reason LJ-38 is the right next problem comes directly out of your own correlation analysis. Your energy decorrelates in about 130 samples. But *basin identity* — which funnel the cluster is sitting in — does not. Escaping a funnel means crossing a barrier that ordinary energy fluctuations never resolve, so its correlation time can be enormous. Your chain can look beautifully converged in $U$ while never once having visited the second funnel.

This is also an important lesson: **the number of independent samples is a property of the observable, not of the chain.** LJ-38 has a double-funnel landscape — the global minimum is a truncated octahedron sitting at the bottom of a *narrow* funnel, while a broader, entropically favoured icosahedral funnel lies slightly higher in energy — and it is the textbook case where this bites. That is why we will need replica exchange. Your well-behaved LJ-13 diagnostics are the perfect contrast to set it against.

**Please go and look at these two structures before we next talk.** I thought that the Cambridge Cluster Database (the Wales group at Cambridge) has coordinates and images for every LJ cluster -- but I could be wrong.  Please check. The Wales and Doye papers on LJ-38 are the standard references. Put the two pictures side by side in your notebook and describe, in your own words, what distinguishes them. What to look for:

- **The truncated octahedron** is a fragment of an **FCC crystal** — the same packing as bulk copper or solid argon. You should be able to pick out flat square and hexagonal faces, and the whole cluster looks like something cut out of a lattice. Its point group is $O_h$, order 48.
- **The icosahedral structure** is built around a 13-atom icosahedral core — the same motif as your LJ-13 cluster — with a partial second shell on top. Look for the **fivefold symmetry axes**. That is the giveaway: fivefold rotational symmetry is *impossible* in any crystal lattice, so this structure has no bulk counterpart at all. Its point group is $C_{5v}$, order 10.

While you are at it, find the **disconnectivity graph** for LJ-38. It is the standard way of drawing an energy landscape, and it shows both funnels and the barrier between them in a single picture. It is worth learning how to read one.

The two minima have *different* energies, roughly $-173.93$ vs $-173.25$, a gap of about $0.68$ in reduced units. Two configurations related by a group element must have identical energy. These do not. So they lie in different orbits, and no equivariant architecture can conflate them. They are different structures in the full sense.

What your network *does* collapse is the **permutational isomers** within each funnel. A structure with point group order $h$ corresponds to $N!/h$ distinct labelled minima, so the truncated octahedron accounts for $38!/48$ of them and the icosahedral minimum for $38!/10$. Since $38! \approx 5 \times 10^{44}$, your equivariant network folds something like $10^{43}$ labelled configurations into a single point *within each funnel*, while correctly keeping the two funnels apart. That is exactly the behaviour you want, and it is your $13!$ result scaled up to something ridiculous.

**A warning to carry into the fall.** Equivariance and ergodicity are separate problems, and you have solved only one of them:

- Equivariance solves *"the same structure written down many different ways."* Done, by construction, for free.
- The double funnel is *"genuinely different structures separated by a barrier."* Your equivariant network is exactly as helpless against this as your baseline was.

A generative model can only reproduce what its training distribution contains. If your MC chain never crosses into the second funnel, your score model will never generate that funnel — and the failure will be **silent**. The energy histogram will look clean, the pair distances will overlay, every diagnostic you currently have will pass, and the model will be confidently sampling one basin out of two. This is why replica exchange is needed to fix the *training data*, not the model. Do not expect the architecture to rescue you here.

One last thing, because I think you will enjoy it. That point group order $h$ does double duty. Your network divides out the $N!/h$ permutational isomers. But in the harmonic superposition approximation that very same multiplicity enters the partition function — each minimum contributes with weight proportional to $(N!/h_i) e^{-\beta U_i}$. Lower symmetry means more isomers, which means more entropy. The icosahedral structure, with $h = 10$ against the octahedron's $48$, therefore gains $k_B \ln(48/10) \approx 1.57 k_B$ of entropy purely by being *less* symmetric — one of the ingredients that lets the higher-energy structure win at finite temperature. The same symmetry number, doing two entirely different jobs: telling your network what to ignore, and telling the physics which funnel to prefer.

Two documents are coming separately: a plan for the fall research project, and a scaffold for the report. Start the report in the transition week and write the **validation methodology** section first — it is your strongest material, and it is the part of this work that most people simply do not do.

Really strong work this week.
