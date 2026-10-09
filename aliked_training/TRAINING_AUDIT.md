# Training audit follow-up

Reviewed base: `maxb` at `d06291a956e2d98653f95a2bf7cfe6f43530bfe9`.

## Implemented

- Treat missing depth as unknown, retain known out-of-view/occlusion errors,
  check both reprojection directions, report evaluable/correct/putative counts.
- Use a seeded scene-balanced holdout (50 pairs/scene, cap 200/source), persist
  selection and manifest hashes, and select by mean correct matches by default.
  Report precision, MS, repeatability and GT coverage alongside selection.
- Match depth resizing to the RGB pixel-centre convention with nearest-exact;
  expose absolute depth tolerance without silently changing its value.
- Evaluate with the public DKD by default, including its threshold fallback.
  Keep training random probes and second NMS separate from deployment detection.
- Replace per-candidate CUDA scalar decisions and quadratic NMS storage with
  stable greedy CPU selection and O(N) scratch. Preserve differentiable gathers.
- Distinguish repeated homography draws while retaining deterministic resume.
  Isolate validation RNG and use nonblocking transfers of pinned tensors.
- Add optional source losses, reliability distributions, score-gradient norms,
  point/visibility statistics, outside-SDDH ratio and sample/update counters.
- Ship working config examples, regression tests and corrected setup instructions.
- Version full checkpoints and reject old evaluation state or changed manifests
  instead of comparing incompatible best scores after resume.

## Deliberately experimental

The four loss formulas and baseline temperature are preserved. `t_rel=.1` is
an explicit ablation against the paper's `1`, not an author-confirmed bug fix.
At temperature 1 and 800 normalized target descriptors, softmax target
probability is bounded by `exp(2)/(exp(2)+799) ≈ .00916`; the new diagnostics
help measure the score-learning signal before selecting a temperature.

The ALIKE archive's reliability uses unnormalized exp, unlike sparse ALIKED.
Its LR is 3e-4; this is the origin of this reconstruction's setting. The ALIKED
paper does not publish LR/warmup. Peak convention, L1/L2 reprojection, source
mixture, rotation range and the meaning of 100K steps remain explicit choices.

## Validation limits and next experiments

These are matching proxies, not official HPatches/IMW reproductions. Mean correct
matches avoids the single-perfect-match precision failure but still depends on
the keypoint budget, coverage and data; compare fixed protocols. Downstream
homography/pose evaluation, real-data overfit, temperature/augmentation sweeps,
CUDA profiling and AMP require the actual datasets and target hardware.

Canonical-path and scene overlap checks do not detect renamed/copied image
content. Manifest hashing does not detect changes to image/depth bytes in place.
Keep the dataset immutable and audit scene/image identity across all sources.

CPU regression coverage includes actual ALIKED forward/backward and exact
model/optimizer resume across an accumulation/epoch boundary. CUDA speed,
800x800 GPU memory and benchmark quality are not claimed by these tests.

## Sources

- [ALIKED paper, sections V–VI](https://arxiv.org/pdf/2304.03608)
- [ALIKE paper](https://arxiv.org/pdf/2112.02906)
- [Official ALIKE training archive](https://github.com/Shiaoming/ALIKE/blob/4e61fc77e75fb1cb3e7447d0a21f854cc61a0d32/assets/ALIKE_code.zip)
- [Author's reliability clarification](https://github.com/Shiaoming/ALIKED/issues/11#issuecomment-2094546617)
- [Temperature discussion (unconfirmed user observation)](https://github.com/Shiaoming/ALIKED/issues/17)
- [PyTorch resize conventions](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)
