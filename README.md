# missing-wedge-ei

A small PyTorch study of **equivariant imaging** on a toy version of the cryo-electron-tomography *missing wedge* problem. It is inspired by the equivariant imaging framework of Chen, Tachella and Davies (ICCV 2021). The setting here is much simpler than real cryo-ET: 2D images, no noise, MNIST digits.

**Status:** work in progress. The measurement operator, data loader, metrics and baseline are implemented and tested. The reconstruction model and the training methods are not yet.

## The idea

We hide part of the information of an image in a precise way, then train a neural network to guess what was hidden. The question is whether this can be done **without ever showing the network a complete image**, using only the fact that a rotated digit is still a plausible digit.

## Where the problem comes from

In cryo-electron tomography, a microscope images a frozen biological sample from several angles, and a computation reconstructs a 3D volume. The sample cannot be tilted through all angles, so part of the information is never measured, and the reconstructed structures are distorted. This is the *missing wedge*.

In the real setting there is no complete "true" image to compare with. If there were, the microscope would not be needed. This is the motivation for methods that do not need ground truth.

## The toy version

The microscope is replaced by a fixed linear operator $A$, and the 3D volume by a 2D image $x$ of a handwritten digit (MNIST, padded to 32x32).

An image can be written as a sum of waves, and its Fourier transform says how much of each wave it contains. A wave has a size (frequency) and an orientation, given by a vector $(k_x, k_y)$. The operator $A$ removes all waves whose vector lies in a **double cone** of half-angle 30 degrees around the vertical frequency axis. It keeps every other wave, and it keeps the zero frequency, which is the mean brightness of the image. The measurement is $y = Ax$.

Properties, each covered by a test in `tests/`:
- $A$ is a projection: $A(Ax) = Ax$, because the mask contains only 0 and 1.
- The mask is symmetric ($M(-k) = M(k)$), so the output is real.
- The mean of the image is preserved.

## Why $A$ cannot simply be inverted

Two different images can give exactly the same measurement: it is enough that they differ only by waves inside the cone, which $A$ does not see. No formula can recover $x$ from $y$. Filling the cone requires a prior on what plausible images look like, and a neural network $f$ can learn such a prior.

## Goal

Train $f$ so that $\hat{x} = f(y)$ is close to $x$. Quality is measured by the PSNR (in dB, higher is better): $10\log_{10}(1/\mathrm{MSE})$, computed per image and then averaged. Gaining 3 dB corresponds to halving the error.

Because $A$ is an orthogonal projection, the squared error splits exactly into an error on the measured frequencies and an error inside the cone (`split_mse` in `metrics.py`). For a reconstruction that is consistent with the measurement, all of the error is in the cone, and this is the only part the network has to invent.

## Four methods

| Method | Loss | Needs complete images? |
|---|---|---|
| Baseline | none, $\hat{x} = y$ | no |
| Supervised | $\lVert f(y) - x \rVert^2$ | yes |
| Consistency only | $\lVert A f(y) - y \rVert^2$ | no |
| Equivariant | consistency + $\lVert f(y_2) - x_2 \rVert^2$ | no |

- **Baseline.** Does nothing. All of its error is in the cone. On the MNIST test set (10,000 images, 30-degree half-angle) it reaches **16.25 dB** (mean of per-image PSNR). The raw values are in `results/baseline.json`.
- **Supervised.** Learns from pairs $(y, x)$. This is not possible in the real setting, and it serves as an upper reference.
- **Consistency only.** A good reconstruction should give back $y$ when measured again. This is necessary but not sufficient: the identity map $\hat{x} = y$ already satisfies it, so the constraint says nothing about the cone.
- **Equivariant.** Assumes that the set of images is stable under 90-degree rotations. The network proposes $x_1 = f(y)$, which is rotated into $x_2$ and measured again with the same operator, giving $y_2 = A x_2$. The network must then recover $x_2$ from $y_2$. The cone is fixed but the image is rotated, so $y_2$ contains waves of $x_1$ that were inside the cone for $y$. To stay consistent on both $y$ and $y_2$, the network is pushed to fill the cone plausibly.

Digits in MNIST are all upright, so the rotation-stability assumption is false for the raw dataset. `data.py` therefore applies to each image a random number of quarter turns (0 to 3, fixed seed), which makes the dataset exactly stable under 90-degree rotation.

## Research question

Without complete images, does equivariance bring anything over consistency alone, and how far is it from the supervised upper reference?

## Scope and limits

This is a 2D, noiseless toy on digits. It illustrates a mechanism and measures its empirical behaviour. It does not show that the method would work on real cryo-ET data, and it does not establish theoretical guarantees for exact recovery.