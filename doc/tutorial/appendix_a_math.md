# Appendix A · Math Reference

This appendix collects the derivations used in the tutorial. It is not a
substitute for the papers, but it is enough to check signs, indices and
normalisation conventions while implementing.

---

## A.1 Notation

| Symbol | Meaning |
|---|---|
| `s ∈ S`, `a ∈ A` | state, action |
| `π(a|s)` | policy |
| `P(s'|s,a)` | transition kernel |
| `r(s,a)` | reward |
| `γ ∈ [0,1)` | discount |
| `τ = (s_0,a_0,s_1,...)` | trajectory |
| `ρ_π(s,a)` | discounted state-action occupancy |
| `φ(s)` | feature vector |
| `μ(π)` | expected discounted feature sum |
| `θ` | reward / model parameters |
| `T_o`, `T_p` | observation and action horizons |
| `x_0`, `x_t`, `ε` | clean action chunk, noisy chunk, noise |
| `ᾱ_t` | cumulative noise schedule |

---

## A.2 Behaviour cloning and compounding error

For a deterministic policy `π_θ` and expert `π*`, suppose the per-step action
error is bounded by `ε` and the policy is stable enough that state deviations grow
at most linearly. A standard telescoping argument gives a cost bound of order
`εT²` over horizon `T`. The intuition:

```text
step 1: action error ε → state error O(ε)
step 2: policy sees an off-distribution state → action error O(ε) + O(ε)
...
step T: accumulated state error O(εT), action error O(εT)
sum over T steps → O(εT²)
```

This is why BC with 1% per-step error can fail over 60 steps. DAgger changes the
state distribution to the policy's own by querying the expert on visited states,
turning the bound into `O(εT)` under no-regret assumptions.

The bound is not a guarantee for neural policies; it is a diagnostic. If BC fails
much worse than the bound suggests, look for normalization bugs and multimodal
data before blaming compounding error.

---

## A.3 Maximum-entropy IRL

### Trajectory distribution

```text
p_θ(τ) = exp(r_θ(τ)) / Z_θ,
Z_θ = ∫ exp(r_θ(τ)) dτ,
r_θ(τ) = Σ_t r_θ(s_t,a_t).
```

### Likelihood gradient

```text
∇_θ log p_θ(τ) = ∇_θ r_θ(τ) - ∇_θ log Z_θ
              = ∇_θ r_θ(τ) - E_{p_θ}[∇_θ r_θ(τ)].
```

Averaging over demonstrations:

```text
∇_θ L = E_D[∇_θ r_θ(τ)] - E_{p_θ}[∇_θ r_θ(τ)].
```

For a linear reward `r_θ(s,a) = θ·φ(s,a)`:

```text
∇_θ L = E_D[φ(s,a)] - E_{π_θ}[φ(s,a)].
```

### Soft Bellman equations

For discrete actions:

```text
Q(s,a) = r(s,a) + γ Σ_{s'} P(s'|s,a) V(s')
V(s)   = log Σ_a exp Q(s,a)
π(a|s) = exp(Q(s,a) - V(s)).
```

The log-sum-exp is a soft maximum. At `β → ∞` in a temperature-scaled version,
the soft policy converges to the hard optimal policy.

### Why the partition is intractable

For a continuous action space, `V(s) = log ∫ exp Q(s,a) da`, which has no closed
form for a neural Q. For a long horizon, `Z_θ` is a high-dimensional integral
over trajectories. AIRL/GAIL replace it with a discriminator.

---

## A.4 AIRL

### Reward decomposition

AIRL parameterises:

```text
f(s,a,s') = g(s) + γ h(s') - h(s).
```

The term `γh(s') - h(s)` is a potential-based shaping term. Potential shaping
does not change the set of optimal policies (Ng et al., 1999), so `g` is the
part that carries task reward. `h` absorbs the state-only bias of the
discriminator.

### Discriminator objective

```text
max_D  E_{ρ_E}[log D(s,a,s')] + E_{ρ_π}[log(1 - D(s,a,s'))].
```

For a fixed policy, the optimal discriminator is:

```text
D*(s,a,s') = ρ_E(s,a,s') / (ρ_E(s,a,s') + ρ_π(s,a,s')).
```

At equilibrium, `f = log(ρ_E/ρ_π) + const`, which is the MaxEnt reward up to a
state-only constant. Training the policy to maximise `f` is therefore a
tractable approximation to the MaxEnt IRL gradient.

### Practical caveats

- The discriminator sees only samples, not the true densities; the equilibrium is
  approximate.
- The shaping term can absorb reward if `g` and `h` are not regularised.
- Reward is identifiable only on the support of the expert/policy distributions.

---

## A.5 GAIL

### Occupancy measure

```text
ρ_π(s,a) = π(a|s) Σ_t γ^t p_t(s).
```

### Objective

```text
min_π max_D  E_{ρ_π}[log D(s,a)] + E_{ρ_E}[log(1 - D(s,a))].
```

For fixed `π`, the inner optimum is:

```text
D*(s,a) = ρ_E(s,a) / (ρ_E(s,a) + ρ_π(s,a)).
```

Substituting:

```text
E_{ρ_π}[log D*] + E_{ρ_E}[log(1-D*)]
= -2 log 2 + 2 JS(ρ_E || ρ_π),
```

so the outer minimisation is JS-divergence minimisation.

### Reward

```text
r(s,a) = -log(1 - D(s,a)) = softplus(f(s,a)).
```

At the optimum `D=1/2`, `r = log 2` everywhere; the gradient vanishes. Early
stopping and regularisation keep the discriminator from saturating.

---

## A.6 Apprenticeship learning

### Feature expectation

```text
μ(π) = E_{τ~π}[Σ_t γ^t φ(s_t)].
```

For a stochastic policy, this is a discounted state visitation weighted by the
feature map. For a linear reward `r = w·φ`, the value is `w·μ(π)` up to a
constant.

### Max-margin problem

```text
max_{w,t} t
s.t. w·μ_E ≥ w·μ_j + t  for all j,
     ||w||₂ ≤ 1.
```

### Projection algorithm

Given `μ_E` and `{μ_1,...,μ_i}`, solve:

```text
min_λ || μ_E - Σ_j λ_j μ_j ||²
s.t. λ_j ≥ 0, Σ λ_j = 1.
```

Let `μ_bar = Σ λ_j μ_j` and `w = μ_E - μ_bar`. Then:

- `w` is orthogonal to the affine hull of the current policies;
- `||w||` is the distance from the expert to the convex hull;
- if `||w||` is small, no policy in the hull is far from the expert.

Training a new policy on `w·φ` adds a vertex in the direction of the residual.
The process terminates when the residual is below a tolerance or the budget is
exhausted.

### QP form

With `G_ij = μ_i·μ_j` and `b_i = μ_E·μ_i`:

```text
min_λ 0.5 λ^T G λ - b^T λ
s.t. λ ≥ 0, 1^T λ = 1.
```

Projected gradient descent uses `λ ← Proj_simplex(λ - η(Gλ - b))` with
`η = 1/||G||₂`. The projection is the Euclidean projection onto the probability
simplex.

---

## A.7 CVAE for ACT

### Model

```text
q_φ(z | s, a_{1:T})   encoder
p_θ(a_{1:T} | s, z)   decoder
p(z) = N(0, I)
```

### ELBO

```text
log p_θ(a|s) ≥ E_{q_φ(z|s,a)}[log p_θ(a|s,z)]
                - KL(q_φ(z|s,a) || p(z)).
```

ACT maximises the ELBO with an L1 reconstruction term and a weight `β` on the KL:

```text
L = E_q[ ||a - a_hat||₁ ] + β KL(q || p).
```

At inference, `z` is set to the prior mean (zero), or sampled for diversity. ACT
uses zero for deterministic evaluation.

### Why KL annealing

Early in training, the decoder is random. A large KL term pushes `q` to the prior
before the decoder can use `z`, causing posterior collapse. Annealing `β` from 0
to its final value lets the decoder learn first.

---

## A.8 Diffusion models

### Forward process

```text
q(x_t | x_{t-1}) = N(√(1-β_t) x_{t-1}, β_t I)
q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I)
x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε.
```

### Training objective

A variational bound simplifies to noise prediction:

```text
L = E_{t,x_0,ε}[ ||ε - ε_θ(x_t,t,c)||² ].
```

The constant and weighting terms are dropped; this is the standard “simple” loss.

### DDPM reverse step

```text
x_{t-1} = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) ε_θ) + σ_t z,
σ_t² = β_t (1-ᾱ_{t-1})/(1-ᾱ_t).
```

### DDIM reverse step

```text
x_0^pred = (x_t - √(1-ᾱ_t) ε_θ)/√ᾱ_t
x_{t-1} = √(ᾱ_{t-1}) x_0^pred + √(1-ᾱ_{t-1}) ε_θ.
```

With `η=0`, DDIM is deterministic. Fewer inference steps trade sample quality for
speed.

### Conditioning

The observation features `c` enter through FiLM:

```text
h ← h * (1 + γ(c,t)) + β(c,t),
```

where `γ,β` are linear functions of the time embedding and the observation
feature. This is cheaper than cross-attention and works well for low-dimensional
action sequences.

---

## A.9 PPO

### Clipped surrogate

```text
r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
L_clip = E_t[ min(r_t A_t, clip(r_t,1-ε,1+ε) A_t) ].
```

### GAE

```text
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
A_t = δ_t + γλ A_{t+1}
R_t = A_t + V(s_t).
```

For a terminated episode, `V(s_{t+1}) = 0`. For a truncated episode, bootstrap
with the value of the final observation. The GAE recursion stops at both
termination and truncation:

```text
A_t = δ_t + γλ (1 - done_t) A_{t+1}.
```

### Tanh-normal change of variables

```text
a = tanh(u), u ~ N(μ,σ²)
log π(a) = log N(u|μ,σ) - Σ_i log(1 - tanh²(u_i)).
```

The correction is `-log|da/du|`. Clamping `a` away from ±1 avoids `atanh(±1)`.

---

## A.10 PointNet

For a point set `{p_1,...,p_N}`, PointNet computes:

```text
f(P) = ρ( max_i h(p_i) ),
```

where `h` is a shared MLP and `ρ` is a global MLP. The max is permutation-
invariant, so `f(P) = f(ΠP)` for any permutation `Π`. This is the property tested
in chapter 12.

---

## A.11 Reward ambiguity

For any potential `Φ` and constant `c`:

```text
r'(s,a) = r(s,a) + γΦ(s') - Φ(s) + c
```

induces the same optimal policies as `r` (potential shaping) up to the constant.
IRL can therefore recover a reward only up to these transformations. Evaluate the
**policy**, not the reward heatmap, and state the ambiguity explicitly.

---

## A.12 References

- Ng, Harada, Russell, *Policy Invariance Under Reward Transformations* (1999).
- Abbeel & Ng, *Apprenticeship Learning via Inverse Reinforcement Learning*
  (2004).
- Ziebart et al., *Maximum Entropy Inverse Reinforcement Learning* (2008).
- Ross, Gordon, Bagnell, *A Reduction of Imitation Learning and Structured
  Prediction to No-Regret Online Learning* (2011).
- Ho & Ermon, *Generative Adversarial Imitation Learning* (2016).
- Fu, Luo, Levine, *Learning Robust Rewards with Adversarial Inverse
  Reinforcement Learning* (2018).
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017).
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost
  Hardware* (2023).
- Chi et al., *Diffusion Policy* (2023).
- Ze et al., *3D Diffusion Policy* (2024).
- Song et al., *Denoising Diffusion Implicit Models* (2021).
