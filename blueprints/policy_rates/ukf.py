import fannypack.utils as fp
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Iterable
import torch


@dataclass
class UnscentedKalmanFilter:
    fx: Callable[..., torch.Tensor]
    hx: Callable[..., torch.Tensor]
    x: torch.Tensor      # (n,)
    P: torch.Tensor      # (n,n)
    Q: torch.Tensor      # (n,n)
    R: torch.Tensor      # (m,m)
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    jitter: float = 0.0  # set >0 if you want a fixed, differentiable jitter

    @staticmethod
    def _sigma_weights(n: int, alpha: float, beta: float, kappa: float, dtype, device):
        lam = alpha**2 * (n + kappa) - n
        c = n + lam
        Wm = torch.full((2*n + 1,), 1.0/(2.0*c), dtype=dtype, device=device)
        Wm[0] = lam / c
        Wc = Wm.clone()
        Wc[0] = Wm[0] + (1.0 - alpha**2 + beta)
        gamma = torch.sqrt(torch.as_tensor(c, dtype=dtype, device=device))
        return Wm, Wc, gamma

    @staticmethod
    def _sigma_points(x: torch.Tensor, P: torch.Tensor, alpha: float, beta: float, kappa: float):
        n = x.shape[0]
        Wm, Wc, gamma = UnscentedKalmanFilter._sigma_weights(
            n, alpha, beta, kappa, x.dtype, x.device)
        # Differentiable Cholesky; assumes P is SPD (add tiny fixed jitter outside if needed)
        S = torch.linalg.cholesky(P)
        U = gamma * S
        sigmas = [x]
        for i in range(n):
            col = U[:, i]
            sigmas.append(x + col)
            sigmas.append(x - col)
        X = torch.stack(sigmas, dim=0)  # (2n+1, n)
        return X, Wm, Wc

    def _add_jitter(self, M: torch.Tensor) -> torch.Tensor:
        if self.jitter > 0.0:
            I = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
            return M + self.jitter * I
        return M

    def predict(self, **fx_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        X, Wm, Wc = UnscentedKalmanFilter._sigma_points(
            self.x, self._add_jitter(self.P), self.alpha, self.beta, self.kappa)
        Xp = torch.stack([self.fx(xi, **fx_kwargs)
                         for xi in X], dim=0)  # (2n+1, n)

        # (n,)
        x_pred = (Wm[:, None] * Xp).sum(dim=0)
        # (2n+1, n)
        dX = Xp - x_pred

        P_pred = torch.zeros_like(self.P)
        for i in range(X.shape[0]):
            vi = dX[i][:, None]
            P_pred = P_pred + Wc[i] * (vi @ vi.T)
        P_pred = self._add_jitter(P_pred) + self.Q

        self.x, self.P = x_pred, P_pred
        return self.x, self.P

    def update(self, z: torch.Tensor, **hx_kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.x.shape[0]
        X, Wm, Wc = UnscentedKalmanFilter._sigma_points(
            self.x, self._add_jitter(self.P), self.alpha, self.beta, self.kappa)

        Zsig = torch.stack([self.hx(xi, **hx_kwargs)
                           for xi in X], dim=0)  # (2n+1, m)
        m = Zsig.shape[1]
        R = self._add_jitter(self.R)
        assert R.shape == (m, m)

        # (m,)
        z_pred = (Wm[:, None] * Zsig).sum(dim=0)
        # (2n+1, m)
        dZ = Zsig - z_pred
        # (2n+1, n)
        dX = X - self.x

        S = torch.zeros((m, m), dtype=self.x.dtype, device=self.x.device)
        Pxz = torch.zeros((n, m), dtype=self.x.dtype, device=self.x.device)
        for i in range(2*n + 1):
            dz = dZ[i][:, None]
            dx = dX[i][:, None]
            S = S + Wc[i] * (dz @ dz.T)
            Pxz = Pxz + Wc[i] * (dx @ dz.T)
        S = self._add_jitter(S) + R

        # Solve is preferable to inv for stability and gradients
        Sinv = torch.linalg.inv(S)
        # (n,m)
        K = Pxz @ Sinv

        # (m,)
        y = z - z_pred
        # (n,)
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T
        # symmetrize (still differentiable)
        self.P = 0.5 * (self.P + self.P.T)

        # Log-likelihood tensor (no .item())
        # ll = -0.5 * (yᵀ S⁻¹ y + log|S| + m log(2π))
        maha = (y[None, :] @ Sinv @ y[:, None]
                ).squeeze()                  # scalar
        # stable log|S|
        sign, logabsdet = torch.slogdet(S)
        # sign should be +1 for SPD; if not, gradients will reflect that
        ll = -0.5 * (maha + logabsdet + m * torch.log(torch.tensor(2.0 *
                     torch.pi, dtype=S.dtype, device=S.device)))
        return self.x, self.P, ll

    def step(self, z: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        px = kwargs.pop("predict_kwargs", {})
        ux = kwargs.pop("update_kwargs", {})
        self.predict(**px)
        return self.update(z, **ux)


def accumulate_loglik(
    ukf: UnscentedKalmanFilter,
    zs: Iterable[torch.Tensor],
    predict_kwargs_seq: Optional[Iterable[dict]] = None,
    update_kwargs_seq: Optional[Iterable[dict]] = None,
) -> torch.Tensor:
    """
    Run a sequence and return the SUM log-likelihood Tensor suitable for backprop.
    """
    if predict_kwargs_seq is None:
        predict_kwargs_seq = [{} for _ in zs]
    if update_kwargs_seq is None:
        update_kwargs_seq = [{} for _ in zs]

    total_ll = 0.0
    for z, pk, uk in zip(zs, predict_kwargs_seq, update_kwargs_seq):
        _, _, ll = ukf.step(z, predict_kwargs=pk, update_kwargs=uk)
        total_ll = total_ll + ll  # keep as Tensor
    return total_ll


class SquareRootUnscentedKalmanFilter:
    """
    Square-root Unscented Kalman Filter (SR-UKF).
    Uses fannypack.utils.cholupdate for numerically stable covariance updates.
    """

    def __init__(
        self,
        fx: Callable[[torch.Tensor], torch.Tensor],
        hx: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        P: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.fx = fx
        self.hx = hx

        # Ensure x is 1D and get state dimension
        if x.dim() > 1:
            x = x.flatten()
        self.x = x.clone()
        self.n = self.x.shape[0]

        # Store noise covariances
        self.Q = Q.clone()
        self.R = R.clone()

        # Initialize square root of covariance (lower triangular)
        if P.dim() != 2 or P.shape[0] != self.n or P.shape[1] != self.n:
            raise ValueError(f"P must be ({self.n}, {self.n}), got {P.shape}")

        self.S = torch.linalg.cholesky(
            P + 1e-6 * torch.eye(self.n, dtype=P.dtype, device=P.device)
        )

        # Sigma-point parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha**2 * (self.n + kappa) - self.n
        self.c = self.n + self.lam

        # Weights for mean and covariance
        self.Wm = torch.full(
            (2 * self.n + 1,),
            1.0 / (2.0 * self.c),
            dtype=x.dtype,
            device=x.device
        )
        self.Wc = self.Wm.clone()
        self.Wm[0] = self.lam / self.c
        self.Wc[0] = self.Wm[0] + (1.0 - alpha**2 + beta)

        self.gamma = torch.sqrt(
            torch.tensor(self.c, dtype=x.dtype, device=x.device)
        )

    def _sigma_points(self, x: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        Compute sigma points from mean and lower-triangular covariance root.

        Returns:
            Sigma points with shape (2*n+1, n)
        """
        n = self.n

        # Validate inputs
        assert x.shape == (n,), f"x shape mismatch: {x.shape} vs ({n},)"
        assert S.shape == (n, n), f"S shape mismatch: {S.shape} vs ({n}, {n})"

        sigmas = [x]
        for i in range(n):
            col = self.gamma * S[:, i]
            sigmas.append(x + col)
            sigmas.append(x - col)

        result = torch.stack(sigmas, dim=0)
        assert result.shape == (
            2*n+1, n), f"Sigma points shape error: {result.shape}"
        return result

    def predict(self, **fx_kwargs):
        """
        Square-root UKF prediction step using QR decomposition.
        Handles negative Wc[0] properly.
        """
        n = self.n

        # Generate sigma points
        X = self._sigma_points(self.x, self.S)  # (2n+1, n)

        # Propagate sigma points through process model
        X_pred_list = []
        for xi in X:
            pred_i = self.fx(xi, **fx_kwargs)
            # Ensure output is 1D with correct size
            if pred_i.dim() > 1:
                pred_i = pred_i.flatten()
            if pred_i.shape[0] != n:
                raise ValueError(
                    f"fx must return state of size {n}, got {pred_i.shape}"
                )
            X_pred_list.append(pred_i)

        X_pred = torch.stack(X_pred_list, dim=0)  # (2n+1, n)

        # Predicted mean
        x_pred = (self.Wm[:, None] * X_pred).sum(dim=0)  # (n,)

        # Deviations from predicted mean
        dX = X_pred - x_pred  # (2n+1, n)

        # Process noise square root
        Q2 = self.Q
        if Q2.dim() > 2:
            Q2 = Q2.squeeze()
        if Q2.dim() == 1:
            Q2 = torch.diag(Q2)
        if Q2.shape != (n, n):
            raise ValueError(f"Q must be ({n}, {n}), got {Q2.shape}")

        Sq = torch.linalg.cholesky(
            Q2 + 1e-9 * torch.eye(n, dtype=Q2.dtype, device=Q2.device)
        )

        # Build square-root covariance using QR decomposition
        # Start with sigma points 1 through 2n (skip index 0)
        W_sqrt = torch.sqrt(torch.abs(self.Wc[1:]))  # (2n,)
        weighted_dX = W_sqrt[:, None] * dX[1:]  # (2n, n)

        # Augment with process noise
        A = torch.cat([weighted_dX.T, Sq.T], dim=1)  # (n, 2n+n)
        # R is (n, n) upper triangular
        _, R = torch.linalg.qr(A.T, mode='reduced')
        S = R.T  # Convert to lower triangular (n, n)

        # Handle first sigma point (index 0) separately
        # This weight can be negative, requiring a downdate
        w0 = self.Wc[0]
        if abs(w0) > 1e-12:
            v = torch.sqrt(torch.abs(w0)) * dX[0]  # (n,)

            # cholupdate expects upper triangular
            S_upper = S.T
            weight_sign = 1.0 if w0 > 0 else -1.0
            S_upper = fp.cholupdate(
                S_upper,
                v,
                weight=torch.tensor(
                    weight_sign, dtype=S.dtype, device=S.device)
            )
            S = S_upper.T  # Back to lower triangular

        # Update state
        self.x = x_pred
        self.S = S

        return self.x, self.S

    def update(self, z: torch.Tensor, **hx_kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SR-UKF update step.
        """
        n = self.n

        # Ensure z is 1D
        if z.dim() > 1:
            z = z.flatten()

        # Generate sigma points
        X = self._sigma_points(self.x, self.S)  # (2n+1, n)

        # Propagate through measurement model
        Zsig_list = []
        for xi in X:
            zi = self.hx(xi, **hx_kwargs)
            if zi.dim() > 1:
                zi = zi.flatten()
            Zsig_list.append(zi)

        Zsig = torch.stack(Zsig_list, dim=0)  # (2n+1, m)
        m = Zsig.shape[1]

        if z.shape[0] != m:
            raise ValueError(
                f"Observation size mismatch: z has {z.shape[0]}, expected {m}")

        # Predicted measurement
        z_pred = (self.Wm[:, None] * Zsig).sum(dim=0)  # (m,)

        # Deviations
        dZ = Zsig - z_pred  # (2n+1, m)
        dX = X - self.x  # (2n+1, n)

        # Cross-covariance Pxy
        Pxy = torch.zeros((n, m), dtype=self.x.dtype, device=self.x.device)
        for i in range(2*n + 1):
            Pxy += self.Wc[i] * torch.outer(dX[i], dZ[i])

        # Innovation covariance square root using QR
        W_sqrt = torch.sqrt(torch.abs(self.Wc))  # (2n+1,)
        weighted_dZ = W_sqrt[:, None] * dZ  # (2n+1, m)

        # Measurement noise square root
        R2 = self.R
        if R2.dim() > 2:
            R2 = R2.squeeze()
        if R2.dim() == 1:
            R2 = torch.diag(R2)
        if R2.shape != (m, m):
            raise ValueError(f"R must be ({m}, {m}), got {R2.shape}")

        Sr = torch.linalg.cholesky(
            R2 + 1e-6 * torch.eye(m, dtype=R2.dtype,
                                  device=R2.device)  # Increased jitter
        )

        # QR decomposition for innovation covariance
        DZ_R = torch.cat([weighted_dZ.T, Sr.T], dim=1)  # (m, 2n+1+m)
        _, Szz_upper = torch.linalg.qr(DZ_R.T, mode='reduced')  # (m, m)
        Szz = Szz_upper.T  # Lower triangular (m, m)

        # Kalman gain via Cholesky solve
        temp = torch.linalg.solve_triangular(Szz, Pxy.T, upper=False)
        K_T = torch.linalg.solve_triangular(Szz.T, temp, upper=True)
        K = K_T.T  # (n, m)

        # Innovation
        y = z - z_pred  # (m,)

        # Update state
        x_new = self.x + K @ y

        # # CLIP STATE UPDATE
        # x_new = torch.clamp(x_new, min=-10.0, max=10.0)
        self.x = x_new

        # Update covariance square root via cholupdate (Joseph form)
        S_upper = self.S.T  # Convert to upper triangular for cholupdate
        for i in range(m):
            ki = K[:, i]  # (n,)
            try:
                S_upper = fp.cholupdate(
                    S_upper,
                    ki,
                    weight=torch.tensor(-1.0, dtype=S_upper.dtype,
                                        device=S_upper.device)
                )
            except:
                # If cholupdate fails, just add small jitter to maintain PD
                S_upper = S_upper + 1e-6 * \
                    torch.eye(n, dtype=S_upper.dtype, device=S_upper.device)

        self.S = S_upper.T  # Back to lower triangular

        # Ensure S is valid
        if torch.isnan(self.S).any() or torch.isinf(self.S).any():
            # Reset to safe covariance
            self.S = torch.eye(n, dtype=self.S.dtype,
                               device=self.S.device) * 0.1

        # Compute log-likelihood
        S_innov = Szz @ Szz.T  # Innovation covariance (m, m)
        S_inv = torch.cholesky_inverse(Szz)
        maha = (y @ S_inv @ y).squeeze()
        sign, logabsdet = torch.slogdet(S_innov)

        ll = -0.5 * (
            maha +
            logabsdet +
            m * torch.log(torch.tensor(2.0 * torch.pi,
                          dtype=S_innov.dtype, device=S_innov.device))
        )

        return self.x, self.S, ll, z_pred

    def step(self, z: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combined predict and update step.

        Args:
            z: Observation vector
            **kwargs: Keyword arguments passed to predict and update

        Returns:
            Updated state, square-root covariance, log-likelihood
        """
        # Separate kwargs for predict and update
        predict_kwargs = kwargs.pop('predict_kwargs', {})
        update_kwargs = kwargs.pop('update_kwargs', {})

        # If no separation provided, pass all kwargs to both
        if not predict_kwargs and not update_kwargs:
            predict_kwargs = kwargs
            update_kwargs = kwargs

        self.predict(**predict_kwargs)
        return self.update(z, **update_kwargs)
