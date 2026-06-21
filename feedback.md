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
