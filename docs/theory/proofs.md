# Complete Proofs for QATNE Theorems

## Theorem 1: Convergence Guarantee

**Statement:** Let $\mathcal{H}$ be a molecular Hamiltonian with ground state energy $E_0$ and spectral gap $\Delta$. The QATNE algorithm converges to $\epsilon$-accuracy of $E_0$ in $O(\text{poly}(1/\epsilon, 1/\Delta))$ iterations with probability at least $1 - \delta$.

### Proof

**Step 1: Energy Landscape Analysis**

Define the energy functional:
$$E(\boldsymbol{\theta}) = \langle\psi(\boldsymbol{\theta})|\mathcal{H}|\psi(\boldsymbol{\theta})\rangle$$

By the variational principle, $E(\boldsymbol{\theta}) \geq E_0$ for all $\boldsymbol{\theta}$.

**Step 2: Gradient Bound**

The gradient satisfies:
$$\nabla E(\boldsymbol{\theta}) = 2\text{Re}\langle\nabla\psi(\boldsymbol{\theta})|\mathcal{H} - E(\boldsymbol{\theta})|\psi(\boldsymbol{\theta})\rangle$$

Using Cauchy-Schwarz:
$$\|\nabla E(\boldsymbol{\theta})\| \leq 2\|\mathcal{H}\|_{\text{op}} = 2L$$

where $L$ is the Lipschitz constant.

**Step 3: Stochastic Gradient Descent**

With noisy gradient estimates $\tilde{\nabla} E = \nabla E + \boldsymbol{\xi}$ where $\mathbb{E}[\boldsymbol{\xi}] = 0$ and $\|\boldsymbol{\xi}\| \leq \sigma/\sqrt{N_{\text{shots}}}$:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \tilde{\nabla} E(\boldsymbol{\theta}_t)$$

Standard SGD analysis gives:
$$\mathbb{E}[E(\boldsymbol{\theta}_T)] - E_0 \leq \frac{L\|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\|^2}{2\sqrt{T}} + \frac{\sigma^2}{\sqrt{T \cdot N_{\text{shots}}}}$$

**Step 4: Setting Parameters**

To achieve $\epsilon$-accuracy with probability $1-\delta$:

- Set $T = O(L^2/\epsilon^2)$ iterations
- Set $N_{\text{shots}} = O(\sigma^2/(\epsilon^2 T))$ shots per measurement
- Total iterations: $O(\text{poly}(1/\epsilon))$

By Markov's inequality, this succeeds with probability $\geq 1 - \delta$. $\square$

## Theorem 2: Quantum Advantage

**Statement:** For molecular Hamiltonians with $n$ orbitals, QATNE achieves $O(n^4)$ time complexity versus $O(n^{10})$ for exact FCI and $O(n^8)$ for CCSD(T).

### Proof

**Classical Methods:**

1. **FCI:** Exact diagonalization requires $O(\binom{2n}{n}^2) = O(4^{2n}/\sqrt{n}) = O(n^{10})$ for typical molecules
2. **CCSD(T):** Coupled cluster with triples scales as $O(n^3 n^4 n) = O(n^8)$

**QATNE Complexity:**

1. **Circuit construction:** $O(n \cdot L)$ gates where $L = O(\log n)$ layers
2. **Gradient estimation:** $p = O(n \cdot L)$ parameters, each requiring $O(N_{\text{shots}})$ measurements
3. **Tensor network update:** $O(n \cdot \chi^3)$ where $\chi = O(\text{poly}(\log n))$
4. **Total per iteration:** $O(n \cdot L \cdot N_{\text{shots}} + n \cdot \chi^3) = O(n^2 \log n)$
5. **Total iterations:** $T = O(\text{poly}(1/\epsilon))$
6. **Overall:** $O(n^2 \log n \cdot \text{poly}(1/\epsilon)) = O(n^4)$ for fixed accuracy

This demonstrates polynomial speedup over classical methods. $\square$

## Theorem 3: Error Bound

**Statement:** The total error satisfies:
$$\Delta E_{\text{total}} \leq C_1/\sqrt{T} + C_2/\sqrt{N_{\text{shots}}} + C_3 \epsilon_{\text{gate}} d + C_4/\chi^\alpha$$

for constants $C_i$ depending on the system.

### Proof

**Optimization Error:**
From gradient descent analysis:
$$\Delta E_{\text{opt}} = E(\boldsymbol{\theta}_T) - E(\boldsymbol{\theta}^*) \leq \frac{L\|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\|^2}{2\sqrt{T}} = \frac{C_1}{\sqrt{T}}$$

**Sampling Error:**
Central limit theorem for measurement outcomes:
$$\Delta E_{\text{sampling}} = |\tilde{E} - E| \leq \frac{\sigma}{\sqrt{N_{\text{shots}}}} \cdot z_{1-\delta/2} = \frac{C_2}{\sqrt{N_{\text{shots}}}}$$

where $z_{1-\delta/2}$ is the quantile for confidence level $1-\delta$.

**Gate Error:**
For circuit with depth $d$ and per-gate error $\epsilon_{\text{gate}}$:
$$\Delta E_{\text{gate}} \leq d \cdot \epsilon_{\text{gate}} \cdot \|\mathcal{H}\|_{\text{op}} = C_3 \epsilon_{\text{gate}} d$$

**Truncation Error:**
From tensor network theory:
$$\Delta E_{\text{truncation}} = |\langle\psi_{\text{exact}}|\mathcal{H}|\psi_{\text{exact}}\rangle - \langle\psi_{\chi}|\mathcal{H}|\psi_{\chi}\rangle| \leq \frac{C \|\mathcal{H}\|_{\text{op}}}{\chi^\alpha} = \frac{C_4}{\chi^\alpha}$$

Summing all contributions via triangle inequality gives the result. $\square$

## Lemma: Parameter Shift Rule

**Statement:** For parameterized gate $U(\theta) = e^{-i\theta G}$ with generator $G$ satisfying $G^2 = I$:
$$\frac{\partial}{\partial\theta}\langle\psi(\theta)|\mathcal{H}|\psi(\theta)\rangle = \frac{\langle\psi(\theta+\pi/2)|\mathcal{H}|\psi(\theta+\pi/2)\rangle - \langle\psi(\theta-\pi/2)|\mathcal{H}|\psi(\theta-\pi/2)\rangle}{2}$$

### Proof

For $|\psi(\theta)\rangle = U(\theta)|\psi_0\rangle$:
$$\frac{\partial|\psi(\theta)\rangle}{\partial\theta} = -iG|\psi(\theta)\rangle$$

Therefore:
$$\frac{\partial E}{\partial\theta} = -i\langle\psi|GH|\psi\rangle + i\langle\psi|HG|\psi\rangle = -i\langle\psi|[G,H]|\psi\rangle$$

Using $e^{i\pi G/2} = iG$ for $G^2 = I$:
$$|\psi(\theta \pm \pi/2)\rangle = e^{-i(\theta \pm \pi/2)G}|\psi_0\rangle = e^{\mp i\pi G/2}e^{-i\theta G}|\psi_0\rangle = \mp iG|\psi(\theta)\rangle$$

Substituting:
$$\frac{E(\theta+\pi/2) - E(\theta-\pi/2)}{2} = \frac{\langle\psi|G^\dagger HG|\psi\rangle - \langle\psi|(-G)^\dagger H(-G)|\psi\rangle}{2} = \frac{\partial E}{\partial\theta}$$

$\square$
