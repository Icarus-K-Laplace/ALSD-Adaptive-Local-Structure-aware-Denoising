### 2. 📝docs/mathematical_model.md

```markdown
# Mathematical Derivation of ALSD

## 1. Noise Modeling
We assume an additive white Gaussian noise model:
$$ Y = X + N, \quad N \sim \mathcal{N}(0, \sigma_n^2) $$
Where $Y$ is the observed image, $X$ is the clean signal, and $N$ is noise.

## 2. Robust Noise Estimation (MAD)
Standard deviation is sensitive to edges (which act as outliers in the gradient domain). I utilize **Median Absolute Deviation (MAD)** in the wavelet/gradient domain for robust estimation:
$$ \hat{\sigma}_n = \frac{\text{Median}(|Y - \text{Median}(Y)|)}{0.6745} $$
The factor $0.6745$ calibrates the MAD to the standard deviation of a normal distribution.

## 3. Local Wiener Filtering
Our goal is to minimize the Mean Square Error (MSE): $E[|X - \hat{X}|^2]$.
In the local spatial domain, the optimal weight $w$ for linear fusion is:
$$ w = \frac{\sigma_x^2}{\sigma_x^2 + \sigma_n^2} $$

However, we only observe $Y$. Since noise and signal are independent:
$$ \sigma_y^2 = \sigma_x^2 + \sigma_n^2 \implies \sigma_x^2 = \max(0, \sigma_y^2 - \sigma_n^2) $$

Substituting this back, we get the implementation formula used in `src/denoiser.py`:
$$ w \approx \frac{\max(0, \text{Var}(Y) - \hat{\sigma}_n^2)}{\text{Var}(Y)} $$

This ensures that in flat regions where $\text{Var}(Y) \approx \sigma_n^2$, the weight $w$ drops to 0, fully suppressing noise.
