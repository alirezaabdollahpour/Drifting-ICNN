"""NPF (Vesseron & Cuturi, 2024) ICNN potential + drift field.


--------------
NPFInputConvexPotential
    The convex potential ψ_ω(z): PSD outer base (diag + low-rank) + deep
    cascade of per-layer convex quadratic injections + non-negative
    hidden-to-hidden weights (parameterized as exp(raw)) + convex
    non-decreasing activation (softplus by default — see init_as_identity
    docstring for why ELU is unsafe here).

NPFDriftField
    V(x) = ∇ψ_ω(x) - x with an Adam inner loop that fits ∇ψ to a Sinkhorn
    barycentric target. Two methods:
        * compute_V(x_gen, y_pos)            -> V
        * compute_V_with_stats(x_gen, y_pos) -> (V, stats)
    Supports two init modes: "identity" (default) and "gaussian" (closed-form
    affine Brenier map between Gaussian approximations of x_gen and y_pos,
    optionally blended with identity via init_blend).

Init contract (NOT a choice — applied jointly):
    (a) Principled LogNormal init on the non-negative cascade weights
        (Hoedt & Klambauer-style positive-weight moments) — applied
        automatically inside NPFNonNegativeDense.reset_parameters.
    (b) Identity init that scales b_linears.weight and b_out.weight to
        O(init_eps) (NOT zero — see init_as_identity docstring), zeros
        the corresponding biases and the outer linear term, and shrinks
        the per-layer / output quadratics so ∇ψ(z) ≈ z to O(init_eps).

Sinkhorn helpers (also exported)
    sinkhorn_simple, sinkhorn_log_domain
    barycentric_target_simple, barycentric_target_log

Architecture helpers
    make_tapered_hidden_dims, count_parameters

Gaussian-OT helpers
    gaussian_ot_affine_map, sample_mean_cov, psd_matrix_power
"""


from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _npf_softplus_inverse(y: float) -> float:
    """Numerically stable inverse of softplus: log(exp(y) - 1) for y > 0."""
    if y <= 0:
        return -1e3
    return float(math.log(math.expm1(y)))


def _npf_principled_moments(fan_in: int) -> Tuple[float, float, float, float, float]:
    """Hoedt & Klambauer-style LogNormal moments for positive-weight init."""
    if fan_in <= 0:
        raise ValueError(f"NPF fan-in must be positive; got {fan_in}.")
    denom = 6.0 * (math.pi - 1.0) + (fan_in - 1.0) * (3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0)
    mu_w = math.sqrt((6.0 * math.pi) / (fan_in * denom))
    sigma_w2 = 1.0 / float(fan_in)
    mu_b = math.sqrt((3.0 * fan_in) / denom)
    log_var_plus_mean_sq = math.log(sigma_w2 + mu_w * mu_w)
    log_mean_sq = math.log(mu_w * mu_w)
    tilde_mu = log_mean_sq - 0.5 * log_var_plus_mean_sq
    tilde_sigma = math.sqrt(max(log_var_plus_mean_sq - log_mean_sq, 1e-12))
    return mu_w, sigma_w2, mu_b, tilde_mu, tilde_sigma


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def make_tapered_hidden_dims(width: int = 512, depth: int = 4,
                             taper: int = 2, min_width: int = 16) -> List[int]:
    """[width, width/taper, width/taper², ...] clipped to min_width."""
    return [max(int(min_width), int(round(width / (taper ** i)))) for i in range(int(depth))]


# ---------------------------------------------------------------------------
# Gaussian-OT helpers (closed-form affine Brenier map)
# ---------------------------------------------------------------------------

def _symmetrize_matrix(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.T)


def psd_matrix_power(A: torch.Tensor, power: float, eps: float = 1e-5) -> torch.Tensor:
    A = _symmetrize_matrix(A)
    evals, evecs = torch.linalg.eigh(A)
    evals = evals.clamp_min(eps)
    return (evecs * evals.pow(power).unsqueeze(0)) @ evecs.T


def sample_mean_cov(x: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Empirical mean/covariance with diagonal jitter."""
    x = x.detach()
    n, d = x.shape
    mean = x.mean(dim=0)
    xc = x - mean
    denom = max(n - 1, 1)
    cov = (xc.T @ xc) / denom + eps * torch.eye(d, device=x.device, dtype=x.dtype)
    return mean, cov


def gaussian_ot_affine_map(x_source: torch.Tensor, y_target: torch.Tensor,
                           eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form affine Brenier map between Gaussian approximations.

    Returns (A, b) such that T0(x) = A x + b transports the Gaussian
    approximation of x_source to the Gaussian approximation of y_target.
    A is symmetric PSD by construction.
    """
    m1, S1 = sample_mean_cov(x_source, eps=eps)
    m2, S2 = sample_mean_cov(y_target, eps=eps)
    S1_sqrt = psd_matrix_power(S1, 0.5, eps=eps)
    S1_inv_sqrt = psd_matrix_power(S1, -0.5, eps=eps)
    middle = S1_sqrt @ S2 @ S1_sqrt
    middle_sqrt = psd_matrix_power(middle, 0.5, eps=eps)
    A = _symmetrize_matrix(S1_inv_sqrt @ middle_sqrt @ S1_inv_sqrt)
    b = m2 - A @ m1
    return A, b


# ---------------------------------------------------------------------------
# Sinkhorn helpers (so notebooks can use a consistent default)
# ---------------------------------------------------------------------------

def sinkhorn_simple(C: torch.Tensor, reg: float = 0.05, num_iters: int = 100) -> torch.Tensor:
    """Direct-domain Sinkhorn — fine for 2D toy distributions, unstable in
    high dimension or with very small reg. Use sinkhorn_log_domain for
    latent-space MNIST and similar."""
    K = torch.exp(-C / (reg + 1e-8))
    u = torch.ones(C.shape[0], device=C.device, dtype=C.dtype)
    v = torch.ones(C.shape[1], device=C.device, dtype=C.dtype)
    for _ in range(num_iters):
        u = 1.0 / (K @ v + 1e-8)
        v = 1.0 / (K.T @ u + 1e-8)
    return u.unsqueeze(1) * K * v.unsqueeze(0)


def sinkhorn_log_domain(C: torch.Tensor, reg: float = 0.2, num_iters: int = 80) -> torch.Tensor:
    """Stable log-domain Sinkhorn with uniform marginals."""
    if reg <= 0:
        raise ValueError("Sinkhorn `reg` must be positive.")
    n, m = C.shape
    log_K = -C / reg
    log_u = torch.zeros(n, device=C.device, dtype=C.dtype)
    log_v = torch.zeros(m, device=C.device, dtype=C.dtype)
    log_a = torch.full((n,), -math.log(n), device=C.device, dtype=C.dtype)
    log_b = torch.full((m,), -math.log(m), device=C.device, dtype=C.dtype)
    for _ in range(num_iters):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K.transpose(0, 1) + log_u.unsqueeze(0), dim=1)
    log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    return torch.exp(log_P)


def barycentric_target_simple(x: torch.Tensor, y: torch.Tensor,
                              reg: float = 0.05, num_iters: int = 100) -> torch.Tensor:
    """Barycentric Sinkhorn target ȳ_i = Σ_j P_ij y_j / Σ_j P_ij (direct domain)."""
    C = torch.cdist(x, y, p=2) ** 2
    P = sinkhorn_simple(C, reg=reg, num_iters=num_iters)
    P_row = P / (P.sum(1, keepdim=True) + 1e-8)
    return P_row @ y


def barycentric_target_log(x: torch.Tensor, y: torch.Tensor,
                           reg: float = 0.2, num_iters: int = 80,
                           normalize_cost: bool = True) -> torch.Tensor:
    """Barycentric Sinkhorn target — log-domain, with optional median-cost
    normalization that makes `reg` dimensionless."""
    C = torch.cdist(x, y, p=2) ** 2
    if normalize_cost:
        scale = C.detach().median().clamp_min(1e-6)
        C = C / scale
    P = sinkhorn_log_domain(C, reg=reg, num_iters=num_iters)
    P_row = P / (P.sum(1, keepdim=True).clamp_min(1e-8))
    return P_row @ y


# ---------------------------------------------------------------------------
# NPF building blocks
# ---------------------------------------------------------------------------

class NPFNonNegativeDense(nn.Module):
    """Non-negative linear (W = exp(raw)) with principled LogNormal init."""

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weight_param = nn.Parameter(torch.empty(in_features, out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        _, _, mu_b, tilde_mu, tilde_sigma = _npf_principled_moments(self.in_features)
        with torch.no_grad():
            if tilde_sigma == 0.0:
                self.weight_param.fill_(tilde_mu)
            else:
                self.weight_param.normal_(mean=tilde_mu, std=tilde_sigma)
            if self.bias is not None:
                self.bias.fill_(-mu_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.exp(self.weight_param)
        y = x.matmul(weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class NPFQuadraticForm(nn.Module):
    """Stack of `num_forms` convex quadratics Q(z) = ||δ⊙z||² + ||A z||²."""

    def __init__(self, input_dim: int, num_forms: int, rank: int = 1, init_eps: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_forms = int(num_forms)
        self.rank = int(rank)
        self.init_eps = float(init_eps)
        delta_raw_init = _npf_softplus_inverse(self.init_eps if self.init_eps > 0 else 0.0)
        self.delta_raw = nn.Parameter(torch.full((self.num_forms, self.input_dim), delta_raw_init))
        if self.rank > 0:
            if self.init_eps > 0.0:
                std = self.init_eps / math.sqrt(max(self.rank * self.input_dim, 1))
                self.A = nn.Parameter(std * torch.randn(self.num_forms, self.rank, self.input_dim))
            else:
                self.A = nn.Parameter(torch.zeros(self.num_forms, self.rank, self.input_dim))
        else:
            self.register_parameter("A", None)

    @property
    def delta(self) -> torch.Tensor:
        return F.softplus(self.delta_raw)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.delta
        q_diag = ((z.unsqueeze(1) * delta.unsqueeze(0)) ** 2).sum(dim=-1)
        if self.A is not None:
            Az = torch.einsum("ord,bd->bor", self.A, z)
            q_lr = (Az ** 2).sum(dim=-1)
            return q_diag + q_lr
        return q_diag


class NPFInputConvexPotential(nn.Module):
    """NPF ICNN potential ψ_ω(z): PSD outer base + deep convex cascade.

    ψ_ω(z) = ½ ||δ ⊙ z||² + ½ ||outer_A z||² + outer_a · z
             + cascade(z) + q_out(z) + b_out(z)

    where cascade is the NPF non-negative cascade with per-layer convex
    quadratic injections (q_blocks) and input-to-hidden affine injections
    (b_linears). Convexity is preserved by construction:
      * outer quadratics δ²·z² and ||outer_A z||² are convex,
      * outer_a · z and b_out·z are affine,
      * cascade activations are convex non-decreasing,
      * hidden-to-hidden weights w_linears / w_out are W = exp(raw) ≥ 0,
      * per-layer quadratics q_blocks are convex,
    so no post-step weight projection is ever required.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        outer_rank: int = 4,
        inner_rank: int = 1,
        activation: str = "softplus",
        elu_alpha: float = 1.0,
        softplus_beta: float = 1.0,
        init_eps: float = 1e-2,
        outer_delta_init: float = 1.0,
    ):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("NPFInputConvexPotential needs at least 1 hidden layer.")
        if outer_delta_init <= 0.0:
            raise ValueError(f"outer_delta_init must be > 0; got {outer_delta_init}.")
        self.input_dim = int(input_dim)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.outer_rank = int(outer_rank)
        self.inner_rank = int(inner_rank)
        self.activation = activation.lower()
        self.elu_alpha = float(elu_alpha)
        self.softplus_beta = float(softplus_beta)
        self.init_eps = float(init_eps)
        self.outer_delta_init = float(outer_delta_init)

        self.outer_delta_raw = nn.Parameter(
            torch.full((self.input_dim,), _npf_softplus_inverse(self.outer_delta_init))
        )
        if self.outer_rank > 0:
            if self.init_eps > 0.0:
                std = self.init_eps / math.sqrt(max(self.outer_rank * self.input_dim, 1))
                self.outer_A = nn.Parameter(std * torch.randn(self.outer_rank, self.input_dim))
            else:
                self.outer_A = nn.Parameter(torch.zeros(self.outer_rank, self.input_dim))
        else:
            self.register_parameter("outer_A", None)

        self.outer_a = nn.Parameter(torch.zeros(self.input_dim))

        self.q_blocks = nn.ModuleList()
        self.b_linears = nn.ModuleList()
        self.w_linears: nn.ModuleList = nn.ModuleList()

        for l, width in enumerate(self.hidden_sizes):
            self.q_blocks.append(
                NPFQuadraticForm(
                    input_dim=self.input_dim,
                    num_forms=width,
                    rank=self.inner_rank,
                    init_eps=self.init_eps,
                )
            )
            self.b_linears.append(nn.Linear(self.input_dim, width, bias=True))
            if l == 0:
                self.w_linears.append(None)  # type: ignore[arg-type]
            else:
                self.w_linears.append(
                    NPFNonNegativeDense(
                        in_features=self.hidden_sizes[l - 1],
                        out_features=width,
                        use_bias=False,
                    )
                )

        self.w_out = NPFNonNegativeDense(
            in_features=self.hidden_sizes[-1], out_features=1, use_bias=False
        )
        self.q_out = NPFQuadraticForm(
            input_dim=self.input_dim,
            num_forms=1,
            rank=self.inner_rank,
            init_eps=self.init_eps,
        )
        self.b_out = nn.Linear(self.input_dim, 1, bias=True)

    # ------------------------------------------------------------------
    # Initializers
    # ------------------------------------------------------------------

    def init_as_identity(self):
        """Force ∇ψ(z) ≈ z to O(init_eps) at t=0.

        Joint with the principled LogNormal cascade init that runs inside
        NPFNonNegativeDense.reset_parameters. The two are complementary —
        identity init kills ψ-gradient paths that would otherwise corrupt
        T(x)≈x while the cascade keeps its LogNormal draws so it is "live
        at init".

        Why softplus is the safe activation here: at init the cascade
        input is small in magnitude, where ELU(u)=u for u≥0 has zero
        second derivative and the cascade contributes no curvature to ψ;
        the inner loop would then degenerate to fitting an affine map.
        Softplus has f''>0 everywhere, so the cascade can carry curvature
        from the first inner step.

        Why b_linears.weight and b_out.weight are scaled to O(eps) and
        NOT zeroed: zero'd input-to-hidden weights mean every neuron in
        a layer sees the same q_blocks-only signal, ∂L/∂(cascade params)
        lies in a near-rank-1 subspace, and Adam stalls in the affine
        basin. The O(eps) Gaussian preserves T(x)≈x to O(eps) but breaks
        the symmetry the inner loop needs.

        Cascade-output shrink (w_linears, w_out): without this, the
        LogNormal init makes ψ_cascade an O(1) constant offset on top of
        ψ_outer at t=0. The gradient w.r.t. z is still O(eps) (because
        b_l and q_l are O(eps)), but Adam spends its first hundreds of
        outer steps cancelling that constant before the cascade can
        carry useful curvature. Shrinking the non-negative weights
        log-scale to log(eps/√fan_in) makes ψ_cascade itself O(eps) at
        init while leaving the cascade fully connected (gradients flow
        on the first inner step).
        """
        eps = self.init_eps
        delta_raw_init = _npf_softplus_inverse(eps) if eps > 0.0 else -1e3
        d = self.input_dim
        with torch.no_grad():
            self.outer_delta_raw.fill_(_npf_softplus_inverse(self.outer_delta_init))
            if self.outer_A is not None:
                if eps > 0.0:
                    std = eps / math.sqrt(max(self.outer_rank * self.input_dim, 1))
                    self.outer_A.normal_(0.0, std)
                else:
                    self.outer_A.zero_()
            self.outer_a.zero_()
            for bl in self.b_linears:
                if eps > 0.0:
                    bl.weight.normal_(0.0, eps / math.sqrt(d))
                else:
                    bl.weight.zero_()
                if bl.bias is not None:
                    bl.bias.zero_()
            for q in self.q_blocks:
                q.delta_raw.fill_(delta_raw_init)
                if q.A is not None:
                    if eps > 0.0:
                        std = eps / math.sqrt(max(q.rank * q.input_dim, 1))
                        q.A.normal_(0.0, std)
                    else:
                        q.A.zero_()
            # Shrink hidden-to-hidden cascade weights so each layer's
            # contribution to ψ-value is O(eps), not O(1) (Bug 4 fix).
            # The non-negative parameterisation is W = exp(weight_param),
            # so weight_param = log(target_W) gives W = target_W exactly.
            for wl in self.w_linears:
                if wl is None:
                    continue
                fan_in = wl.in_features
                if eps > 0.0:
                    target_w = eps / math.sqrt(max(fan_in, 1))
                    wl.weight_param.fill_(math.log(target_w))
                else:
                    wl.weight_param.fill_(-1e3)
                if wl.bias is not None:
                    wl.bias.zero_()
            self.q_out.delta_raw.fill_(delta_raw_init)
            if self.q_out.A is not None:
                if eps > 0.0:
                    std = eps / math.sqrt(max(self.q_out.rank * self.q_out.input_dim, 1))
                    self.q_out.A.normal_(0.0, std)
                else:
                    self.q_out.A.zero_()
            # Same shrink for the output non-negative projection.
            fan_in_out = self.w_out.in_features
            if eps > 0.0:
                target_w_out = eps / math.sqrt(max(fan_in_out, 1))
                self.w_out.weight_param.fill_(math.log(target_w_out))
            else:
                self.w_out.weight_param.fill_(-1e3)
            if self.w_out.bias is not None:
                self.w_out.bias.zero_()
            if eps > 0.0:
                self.b_out.weight.normal_(0.0, eps / math.sqrt(d))
            else:
                self.b_out.weight.zero_()
            if self.b_out.bias is not None:
                self.b_out.bias.zero_()

    def set_gaussian_affine_init(self, A: torch.Tensor, b: torch.Tensor,
                                 blend: float = 1.0, eps: float = 1e-5):
        """Set ∇ψ(z) ≈ ((1-blend)I + blend·A) z + blend·b at t=0.

        Realised by reshaping the outer (diag + low-rank) PSD term:
            ∇ψ_outer(z) = (diag(δ²) + outer_A^T outer_A) z + outer_a
        We pick (δ², outer_A, outer_a) so that
            diag(δ²) + outer_A^T outer_A = A_eff   and   outer_a = b_eff.

        If outer_rank ≥ input_dim the eigendecomposition of A_eff is
        captured exactly inside outer_A and δ is set to a small floor for
        strong convexity. Otherwise the top-`outer_rank` eigen-pairs go
        into outer_A and the diagonal residual lands on δ — this is
        approximate but PSD-safe (off-diagonal residual energy is
        dropped, never made negative).

        The cascade and per-layer quadratics are reset to identity-init
        floor so they contribute O(eps) to T(z).
        """
        d = self.input_dim
        if A.shape != (d, d):
            raise ValueError(f"A must be ({d},{d}); got {tuple(A.shape)}.")
        if b.shape != (d,):
            raise ValueError(f"b must be ({d},); got {tuple(b.shape)}.")

        device = self.outer_delta_raw.device
        dtype = self.outer_delta_raw.dtype
        A = A.detach().to(device=device, dtype=dtype)
        b = b.detach().to(device=device, dtype=dtype)
        I = torch.eye(d, device=device, dtype=dtype)
        blend = float(blend)
        A_eff = _symmetrize_matrix((1.0 - blend) * I + blend * A)
        b_eff = blend * b

        evals, evecs = torch.linalg.eigh(A_eff)
        evals = evals.clamp_min(eps)
        # Sort descending
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx]
        evecs = evecs[:, idx]

        k = max(0, self.outer_rank)
        delta_raw_floor = _npf_softplus_inverse(math.sqrt(eps))

        with torch.no_grad():
            if self.outer_A is not None:
                self.outer_A.zero_()
                k_eff = min(k, d)
                for i in range(k_eff):
                    self.outer_A[i].copy_(evals[i].sqrt() * evecs[:, i])
                if k_eff >= d:
                    # Full capture in outer_A; δ to a tiny floor.
                    self.outer_delta_raw.fill_(delta_raw_floor)
                else:
                    # Diagonal residual on δ.
                    outer_part = self.outer_A.t() @ self.outer_A  # (d, d)
                    diag_res = torch.diagonal(A_eff - outer_part).clamp_min(eps)
                    delta_vals = diag_res.sqrt()
                    delta_raw_vals = torch.tensor(
                        [_npf_softplus_inverse(float(v)) for v in delta_vals.tolist()],
                        device=device, dtype=dtype,
                    )
                    self.outer_delta_raw.copy_(delta_raw_vals)
            else:
                # outer_rank=0: only diagonal δ available; exact only when
                # A_eff is diagonal, otherwise approximate.
                diag_A = torch.diagonal(A_eff).clamp_min(eps)
                delta_vals = diag_A.sqrt()
                delta_raw_vals = torch.tensor(
                    [_npf_softplus_inverse(float(v)) for v in delta_vals.tolist()],
                    device=device, dtype=dtype,
                )
                self.outer_delta_raw.copy_(delta_raw_vals)

            self.outer_a.copy_(b_eff)

            # Reset cascade and per-layer quadratics to identity-init floor
            # so their contribution to ∇ψ is O(eps).
            for bl in self.b_linears:
                bl.weight.zero_()
                if bl.bias is not None:
                    bl.bias.zero_()
            for q in self.q_blocks:
                q.delta_raw.fill_(delta_raw_floor)
                if q.A is not None:
                    q.A.zero_()
            self.q_out.delta_raw.fill_(delta_raw_floor)
            if self.q_out.A is not None:
                self.q_out.A.zero_()
            self.b_out.weight.zero_()
            if self.b_out.bias is not None:
                self.b_out.bias.zero_()

    def set_gaussian_init_from_samples(self, x_source: torch.Tensor,
                                       y_target: torch.Tensor,
                                       blend: float = 1.0, eps: float = 1e-4
                                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: compute (A,b) from sample stats then call
        set_gaussian_affine_init. Returns the (A,b) used."""
        A, b = gaussian_ot_affine_map(x_source, y_target, eps=eps)
        self.set_gaussian_affine_init(A, b, blend=blend, eps=eps)
        return A, b

    # ------------------------------------------------------------------
    # Forward / gradient
    # ------------------------------------------------------------------

    def _act(self, u: torch.Tensor) -> torch.Tensor:
        if self.activation == "elu":
            return F.elu(u, alpha=self.elu_alpha)
        if self.activation == "softplus":
            beta = self.softplus_beta
            return F.softplus(beta * u) / beta
        if self.activation == "relu":
            return F.relu(u)
        raise ValueError(f"Unsupported NPF activation '{self.activation}'")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_flat = z.view(z.size(0), -1)

        delta_out = F.softplus(self.outer_delta_raw)
        q_diag = 0.5 * (delta_out.pow(2) * z_flat.pow(2)).sum(dim=-1)
        if self.outer_A is not None:
            Az = z_flat @ self.outer_A.t()
            q_lr = 0.5 * (Az ** 2).sum(dim=-1)
        else:
            q_lr = torch.zeros(z_flat.size(0), dtype=z_flat.dtype, device=z_flat.device)
        linear = z_flat @ self.outer_a

        q0 = self.q_blocks[0](z_flat)
        b0 = self.b_linears[0](z_flat)
        h = self._act(q0 + b0)
        for l in range(1, len(self.hidden_sizes)):
            ql = self.q_blocks[l](z_flat)
            bl = self.b_linears[l](z_flat)
            wl = self.w_linears[l](h)
            h = self._act(wl + ql + bl)

        phi = (
            self.w_out(h).squeeze(-1)
            + self.q_out(z_flat).squeeze(-1)
            + self.b_out(z_flat).squeeze(-1)
        )
        return q_diag + q_lr + linear + phi

    @torch.enable_grad()
    def gradient(self, z: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        """T_ω(z) = ∇_z ψ_ω(z)."""
        z_in = z.detach().clone().requires_grad_(True)
        psi = self.forward(z_in)
        return torch.autograd.grad(psi.sum(), z_in, create_graph=create_graph)[0]


# ---------------------------------------------------------------------------
# Drift field
# ---------------------------------------------------------------------------

class NPFDriftField:
    """V(x) = ∇ψ_ω(x) - x using the NPF input convex potential.

    Inner-loop objective (selected via `inner_objective`):

      * "regression" (legacy default) — fit T_ω(x) by Adam against a
        Sinkhorn barycentric target ȳ. Cheap but inherits the high-reg
        barycentric collapse pathology: when reg is large, ȳ ≈ E[y]
        for every x and ψ learns the trivial "push everything to the
        mean" map.

      * "semi_dual" — Makkuva-Taghvaei-Lee-Oh (2020) min-max objective
        with an auxiliary ICNN `φ` parameterising the conjugate ψ*.
        Loss V(ψ, φ) = E_y[<y, ∇φ(y)> - ψ(∇φ(y))] - E_x[ψ(x)],
        minimised over ψ and maximised over φ. This is the standard
        W2GN setup (Korotin et al., 2019) and avoids any Sinkhorn
        smoothing. T = ∇ψ recovers the Brenier map μ → ν.

    Two init modes (independent of inner_objective):
        * "identity"  — ∇ψ(z) ≈ z at t=0 (default).
        * "gaussian"  — on the first compute_V call, sets ∇ψ(z) to the
                        closed-form Brenier map between Gaussian
                        approximations of (x, y), blended with identity
                        via init_blend.

    Adam state across outer batches:
        * `reset_inner_optimizer=True` (default for semi_dual; safer for
           regression too) — fresh Adam state at the top of every inner
           loop. Avoids carrying batch-t's momentum into batch-t+1's
           regression problem (the inner objective changes every batch,
           so persisted momentum points the wrong way).

    Convexity is preserved automatically because non-negative weights are
    parameterized as exp(·); no post-step projection is required.
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128, 64),
        outer_rank: int = 4,
        inner_rank: int = 1,
        activation: str = "softplus",
        elu_alpha: float = 1.0,
        softplus_beta: float = 1.0,
        init_eps: float = 1e-2,
        outer_delta_init: float = 1.0,
        inner_steps: int = 30,
        inner_lr: float = 1e-2,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip: float = 5.0,
        # Sinkhorn target (only used by inner_objective="regression")
        sinkhorn_reg: float = 0.2,
        sinkhorn_iters: int = 80,
        sinkhorn_target_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        # Init mode
        init_mode: str = "identity",
        init_blend: float = 1.0,
        gaussian_init_eps: float = 1e-4,
        # Inner objective (regression vs Makkuva-style semi-dual)
        inner_objective: str = "regression",
        semi_dual_phi_steps: int = 1,
        # W2GN cycle-consistency stabiliser for the semi-dual objective.
        # Adds γ·(‖∇φ(∇ψ(x)) − x‖² + ‖∇ψ(∇φ(y)) − y‖²) which (i) is bounded
        # below by 0 so ψ cannot run away to ±∞ to maximise the dual, and
        # (ii) enforces ψ ≈ φ⁻¹ which is exactly the geometric requirement
        # for the optimum of the dual. Set to 0.0 to recover the bare
        # Makkuva min–max (unstable for the architecture in this repo).
        cycle_weight: float = 0.0,
        reset_inner_optimizer: bool = True,
        # Strong convexity is implicit in the outer δ; kept here for
        # API parity with the ICNN baseline. Currently unused.
        strong_convexity: float = 1.0,
    ):
        self.dim = int(dim)
        self.hidden_dims = list(hidden_dims)
        self.outer_rank = int(outer_rank)
        self.inner_rank = int(inner_rank)
        self.activation = activation
        self.elu_alpha = float(elu_alpha)
        self.softplus_beta = float(softplus_beta)
        self.init_eps = float(init_eps)
        self.outer_delta_init = float(outer_delta_init)

        self.psi = NPFInputConvexPotential(
            input_dim=dim,
            hidden_sizes=self.hidden_dims,
            outer_rank=outer_rank,
            inner_rank=inner_rank,
            activation=activation,
            elu_alpha=elu_alpha,
            softplus_beta=softplus_beta,
            init_eps=init_eps,
            outer_delta_init=outer_delta_init,
        )
        # Joint init: principled LogNormal draws come from the layer
        # constructors; identity init is applied on top. Gaussian init,
        # if requested, is a one-shot override on the first batch (since
        # it depends on sample statistics).
        self.psi.init_as_identity()

        self.inner_lr = float(inner_lr)
        self.adam_betas = adam_betas
        self.adam_eps = float(adam_eps)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.inner_steps = int(inner_steps)

        self.sinkhorn_reg = float(sinkhorn_reg)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self._user_sinkhorn_target_fn = sinkhorn_target_fn

        if init_mode not in {"identity", "gaussian"}:
            raise ValueError(f"init_mode must be 'identity' or 'gaussian'; got {init_mode!r}")
        self.init_mode = init_mode
        self.init_blend = float(init_blend)
        self.gaussian_init_eps = float(gaussian_init_eps)
        self._did_gaussian_init = False
        self._last_gaussian_init_v_norm: Optional[float] = None

        if inner_objective not in {"regression", "semi_dual"}:
            raise ValueError(
                f"inner_objective must be 'regression' or 'semi_dual'; got {inner_objective!r}"
            )
        self.inner_objective = inner_objective
        self.semi_dual_phi_steps = max(1, int(semi_dual_phi_steps))
        self.cycle_weight = float(cycle_weight)
        self.reset_inner_optimizer = bool(reset_inner_optimizer)

        self.strong_convexity = float(strong_convexity)

        # Auxiliary conjugate-side ICNN φ for the semi-dual objective.
        # Same architecture as ψ. Gradient ∇φ(y) approximates T⁻¹(y),
        # i.e. the inverse Brenier map; at the optimum it equals
        # (∇ψ)⁻¹ ∘ identity, and ψ ∘ ∇φ ≈ ψ ∘ ψ*-grad ≈ id.
        if self.inner_objective == "semi_dual":
            self.phi = NPFInputConvexPotential(
                input_dim=dim,
                hidden_sizes=self.hidden_dims,
                outer_rank=outer_rank,
                inner_rank=inner_rank,
                activation=activation,
                elu_alpha=elu_alpha,
                softplus_beta=softplus_beta,
                init_eps=init_eps,
                outer_delta_init=outer_delta_init,
            )
            self.phi.init_as_identity()
        else:
            self.phi = None

        self.optimizer = self._make_optimizer(self.psi)
        self.phi_optimizer: Optional[optim.Optimizer] = (
            self._make_optimizer(self.phi) if self.phi is not None else None
        )

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _make_optimizer(self, module: nn.Module) -> optim.Optimizer:
        return optim.Adam(
            module.parameters(),
            lr=self.inner_lr,
            betas=self.adam_betas,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )

    def to(self, device) -> "NPFDriftField":
        self.psi = self.psi.to(device)
        if self.phi is not None:
            self.phi = self.phi.to(device)
        # Rebind optimizers to device-resident parameters.
        self.optimizer = self._make_optimizer(self.psi)
        if self.phi is not None:
            self.phi_optimizer = self._make_optimizer(self.phi)
        return self

    def parameters(self):
        # Both ψ and φ are part of the inner problem; expose both so
        # external callers (e.g. checkpointers) see all trainable state.
        if self.phi is None:
            return self.psi.parameters()
        return list(self.psi.parameters()) + list(self.phi.parameters())

    def _sinkhorn_target(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._user_sinkhorn_target_fn is not None:
            return self._user_sinkhorn_target_fn(x, y)
        return barycentric_target_log(
            x, y, reg=self.sinkhorn_reg, num_iters=self.sinkhorn_iters,
            normalize_cost=True,
        )

    def _maybe_gaussian_init(self, x: torch.Tensor, y: torch.Tensor):
        if self.init_mode != "gaussian" or self._did_gaussian_init:
            return None, None
        A, b = self.psi.set_gaussian_init_from_samples(
            x, y, blend=self.init_blend, eps=self.gaussian_init_eps
        )
        # Reset Adam state after manually changing parameters.
        self.optimizer = self._make_optimizer(self.psi)
        if self.phi is not None:
            # φ also gets the inverse affine warm-start so that ∇φ ∘ ∇ψ ≈ id
            # at t=0; cheapest version is to set φ to the same affine — V
            # at init is then 0, and the semi-dual gradients come from
            # batch-level fluctuations on the first inner step.
            self.phi.init_as_identity()
            self.phi_optimizer = self._make_optimizer(self.phi)
        self._did_gaussian_init = True
        with torch.no_grad():
            eye = torch.eye(self.dim, device=x.device, dtype=x.dtype)
            A_eff = (1.0 - self.init_blend) * eye + self.init_blend * A.to(
                device=x.device, dtype=x.dtype
            )
            b_eff = self.init_blend * b.to(device=x.device, dtype=x.dtype)
            T0 = x @ A_eff.T + b_eff
            self._last_gaussian_init_v_norm = float(((T0 - x) ** 2).mean().item())
        return A, b

    # ------------------------------------------------------------------
    # Inner loop / drift
    # ------------------------------------------------------------------

    def _inner_loop_regression(self, x: torch.Tensor, y_target: torch.Tensor
                               ) -> Tuple[torch.Tensor, float]:
        """Run `inner_steps` Adam steps fitting T_ω(x) ≈ y_target."""
        if self.reset_inner_optimizer:
            self.optimizer = self._make_optimizer(self.psi)
        last_inner_loss = float("nan")
        for _ in range(self.inner_steps):
            self.optimizer.zero_grad(set_to_none=True)
            T_x = self.psi.gradient(x, create_graph=True)
            loss = ((T_x - y_target) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.psi.parameters(), self.grad_clip)
            self.optimizer.step()
            last_inner_loss = float(loss.item())

        T_x = self.psi.gradient(x, create_graph=False)
        V = (T_x.detach() - x).detach()
        return V, last_inner_loss

    def _inner_loop_semi_dual(self, x: torch.Tensor, y: torch.Tensor
                              ) -> Tuple[torch.Tensor, float]:
        """Makkuva semi-dual + W2GN cycle-consistency: alternate φ-max / ψ-min.

        V(ψ, φ) = E_y[<y, ∇φ(y)> - ψ(∇φ(y))] - E_x[ψ(x)]

        With ``cycle_weight > 0`` the loss adds the W2GN regulariser

            L_cycle = γ · (E_x‖∇φ(∇ψ(x)) − x‖² + E_y‖∇ψ(∇φ(y)) − y‖²),

        which keeps ψ and φ as proper inverses of each other and makes the
        bare semi-dual stable. The φ-step picks up the x-cycle (because φ
        owns the inverse-of-ψ side); the ψ-step picks up the y-cycle.
        Without this term, ψ can run away to ±∞ to maximise the dual
        unboundedly — see the warning in the notebook for empirical
        evidence of that failure mode.

        ψ and φ are *separate* ICNNs, so each step has its own optimizer
        and its own grad clip. Convexity of both is preserved by their
        non-negative cascade parameterisation.
        """
        assert self.phi is not None and self.phi_optimizer is not None
        if self.reset_inner_optimizer:
            self.optimizer = self._make_optimizer(self.psi)
            self.phi_optimizer = self._make_optimizer(self.phi)
        last_psi_loss = float("nan")
        cycle_weight = self.cycle_weight

        for _ in range(self.inner_steps):
            # ---- φ-step: maximise V over φ-parameters with ψ fixed.
            #
            # Differentiating <y, ∇φ(y)> - ψ(∇φ(y)) w.r.t. φ-parameters
            # requires create_graph=True on the φ-gradient. We freeze
            # ψ-params for the duration so backward through ψ does not
            # build the parameter-side of ψ's graph (saves memory and
            # avoids touching ψ.grad with stale values).
            for p in self.psi.parameters():
                p.requires_grad_(False)
            try:
                for _ in range(self.semi_dual_phi_steps):
                    self.phi_optimizer.zero_grad(set_to_none=True)
                    T_phi_y = self.phi.gradient(y, create_graph=True)
                    inner_prod = (T_phi_y * y).sum(dim=-1)
                    psi_at_T = self.psi(T_phi_y)
                    # max V_φ = <y, ∇φ(y)> - ψ(∇φ(y))  →  min negation.
                    phi_loss = -(inner_prod - psi_at_T).mean()

                    if cycle_weight > 0.0:
                        # x-cycle: φ should invert ψ on x ∼ μ.
                        # ψ is frozen so ∇ψ(x) is a pure tensor (no
                        # parameter-side autograd dependency); detach is
                        # technically redundant but kept for clarity.
                        T_psi_x = self.psi.gradient(x, create_graph=False).detach()
                        T_phi_T_psi_x = self.phi.gradient(T_psi_x, create_graph=True)
                        cycle_x = (T_phi_T_psi_x - x).pow(2).mean()
                        phi_loss = phi_loss + cycle_weight * cycle_x

                    phi_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.phi.parameters(), self.grad_clip)
                    self.phi_optimizer.step()
            finally:
                for p in self.psi.parameters():
                    p.requires_grad_(True)

            # ---- ψ-step: minimise V over ψ-parameters with φ fixed.
            #
            # E_y[ψ(∇φ(y))] - E_x[ψ(x)]  is linear in ψ-params (ψ is
            # affine in its own parameters along any fixed input).
            # Gradient flow into φ is killed by detaching ∇φ(y).
            self.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                T_phi_y_det = self.phi.gradient(y, create_graph=False).detach()
            psi_loss = self.psi(T_phi_y_det).mean() - self.psi(x).mean()

            if cycle_weight > 0.0:
                # y-cycle: ψ should invert φ on y ∼ ν. T_phi_y_det is
                # already detached so no φ-graph leakage; create_graph
                # on the ψ-side gives us the second-derivative pathway
                # we need to pull this loss into ψ-params.
                T_psi_T_phi_y = self.psi.gradient(T_phi_y_det, create_graph=True)
                cycle_y = (T_psi_T_phi_y - y).pow(2).mean()
                psi_loss = psi_loss + cycle_weight * cycle_y

            psi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.psi.parameters(), self.grad_clip)
            self.optimizer.step()
            last_psi_loss = float(psi_loss.item())

        T_x = self.psi.gradient(x, create_graph=False)
        V = (T_x.detach() - x).detach()
        return V, last_psi_loss

    def compute_V(self, x_gen: torch.Tensor, y_pos: torch.Tensor) -> torch.Tensor:
        """Drift V = ∇ψ(x) - x after `inner_steps` updates on ψ (and φ)."""
        x = x_gen.detach()
        y = y_pos.detach()
        self._maybe_gaussian_init(x, y)
        if self.inner_objective == "semi_dual":
            V, _ = self._inner_loop_semi_dual(x, y)
        else:
            with torch.no_grad():
                y_target = self._sinkhorn_target(x, y)
            V, _ = self._inner_loop_regression(x, y_target)
        return V

    def compute_V_with_stats(self, x_gen: torch.Tensor, y_pos: torch.Tensor
                             ) -> Tuple[torch.Tensor, dict]:
        """Same as compute_V but returns (V, stats) for ablations.

        stats keys (always present, may be NaN if not applicable):
            inner_loss             — last inner-step loss (regression L²
                                      for "regression"; ψ-loss V for
                                      "semi_dual", which can be < 0).
            target_v_norm          — ||ȳ - x||² (Sinkhorn ref). Computed
                                      for both objectives as a diagnostic.
            fit_ratio              — ||V||² / target_v_norm. With
                                      "semi_dual", this is a *comparison*
                                      to the Sinkhorn reference, not the
                                      actual training target.
            gaussian_init_v_norm   — affine init drift norm (or None).
        """
        x = x_gen.detach()
        y = y_pos.detach()
        self._maybe_gaussian_init(x, y)
        with torch.no_grad():
            y_target = self._sinkhorn_target(x, y)
            target_v_norm = float(((y_target - x) ** 2).mean().item())
        if self.inner_objective == "semi_dual":
            V, last_inner_loss = self._inner_loop_semi_dual(x, y)
        else:
            V, last_inner_loss = self._inner_loop_regression(x, y_target)
        v_norm = float((V ** 2).mean().item())
        stats = {
            "inner_loss": last_inner_loss,
            "target_v_norm": target_v_norm,
            "fit_ratio": v_norm / (target_v_norm + 1e-8),
            "gaussian_init_v_norm": self._last_gaussian_init_v_norm,
            "objective": self.inner_objective,
        }
        return V, stats


__all__ = [
    # Building blocks
    "NPFNonNegativeDense",
    "NPFQuadraticForm",
    "NPFInputConvexPotential",
    "NPFDriftField",
    # Helpers
    "make_tapered_hidden_dims",
    "count_parameters",
    "gaussian_ot_affine_map",
    "psd_matrix_power",
    "sample_mean_cov",
    "sinkhorn_simple",
    "sinkhorn_log_domain",
    "barycentric_target_simple",
    "barycentric_target_log",
]
