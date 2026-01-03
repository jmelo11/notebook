from __future__ import annotations
from typing import Optional, List
from torch.optim.optimizer import Optimizer
from torch.optim import Optimizer
from copy import deepcopy
import torch
from torch.func import vmap
from functools import reduce
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from typing import Optional, Union, TypeAlias, Iterable, Dict, Any, Tuple

ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[Dict[str, Any]
                                     ], Iterable[Tuple[str, torch.Tensor]]
]


def _armijo(f, x, gx, dx, t, alpha=0.1, beta=0.5):
    """Line search to find a step size that satisfies the Armijo condition."""
    f0 = f(x, 0, dx)
    f1 = f(x, t, dx)
    while f1 > f0 + alpha * t * gx.dot(dx):
        t *= beta
        f1 = f(x, t, dx)
    return t


def _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, x):
    """Applies the inverse of the Nystrom approximation of the Hessian to a vector."""
    z = U.T @ x
    z = (lambd_r + mu) * (U @ (S_mu_inv * z)) + (x - U @ z)
    return z


def _nystrom_pcg(hess, b, x, mu, U, S, r, tol, max_iters):
    """Solves a positive-definite linear system using NyströmPCG.

    `Frangella et al. Randomized Nyström Preconditioning.
    SIAM Journal on Matrix Analysis and Applications, 2023.
    <https://epubs.siam.org/doi/10.1137/21M1466244>`"""
    lambd_r = S[r - 1]
    S_mu_inv = (S + mu) ** (-1)

    resid = b - (hess(x) + mu * x)
    with torch.no_grad():
        z = _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, resid)
        p = z.clone()

    i = 0

    while torch.norm(resid) > tol and i < max_iters:
        v = hess(p) + mu * p
        with torch.no_grad():
            alpha = torch.dot(resid, z) / torch.dot(p, v)
            x += alpha * p

            rTz = torch.dot(resid, z)
            resid -= alpha * v
            z = _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, resid)
            beta = torch.dot(resid, z) / rTz

            p = z + beta * p

        i += 1

    if torch.norm(resid) > tol:
        print(
            f"Warning: PCG did not converge to tolerance. Tolerance was {tol} but norm of residual is {torch.norm(resid)}")

    return x


class NysNewtonCG(Optimizer):
    """Implementation of NysNewtonCG, a damped Newton-CG method that uses Nyström preconditioning.

    `Rathore et al. Challenges in Training PINNs: A Loss Landscape Perspective.
    Preprint, 2024. <https://arxiv.org/abs/2402.01868>`

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    NOTE: This optimizer is currently a beta version.

    Our implementation is inspired by the PyTorch implementation of `L-BFGS
    <https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS>`.

    The parameters rank and mu will probably need to be tuned for your specific problem.
    If the optimizer is running very slowly, you can try one of the following:
    - Increase the rank (this should increase the accuracy of the Nyström approximation in PCG)
    - Reduce cg_tol (this will allow PCG to terminate with a less accurate solution)
    - Reduce cg_max_iters (this will allow PCG to terminate after fewer iterations)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1.0)
        rank (int, optional): rank of the Nyström approximation (default: 10)
        mu (float, optional): damping parameter (default: 1e-4)
        chunk_size (int, optional): number of Hessian-vector products to be computed in parallel (default: 1)
        cg_tol (float, optional): tolerance for PCG (default: 1e-16)
        cg_max_iters (int, optional): maximum number of PCG iterations (default: 1000)
        line_search_fn (str, optional): either 'armijo' or None (default: None)
        verbose (bool, optional): verbosity (default: False)

    """

    def __init__(self, params, lr=1.0, rank=10, mu=1e-4, chunk_size=1,
                 cg_tol=1e-16, cg_max_iters=1000, line_search_fn=None, verbose=False):
        defaults = dict(lr=lr, rank=rank, chunk_size=chunk_size, mu=mu, cg_tol=cg_tol,
                        cg_max_iters=cg_max_iters, line_search_fn=line_search_fn)
        self.rank = rank
        self.mu = mu
        self.chunk_size = chunk_size
        self.cg_tol = cg_tol
        self.cg_max_iters = cg_max_iters
        self.line_search_fn = line_search_fn
        self.verbose = verbose
        self.U = None
        self.S = None
        self.n_iters = 0
        super(NysNewtonCG, self).__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError(
                "NysNewtonCG doesn't currently support per-parameter options (parameter groups)")

        if self.line_search_fn is not None and self.line_search_fn != 'armijo':
            raise ValueError("NysNewtonCG only supports Armijo line search")

        self._params = self.param_groups[0]['params']
        self._params_list = list(self._params)
        self._numel_cache = None

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns (i) the loss and (ii) gradient w.r.t. the parameters.
            The closure can compute the gradient w.r.t. the parameters by calling torch.autograd.grad on the loss with create_graph=True.
        """
        if self.n_iters == 0:
            # Store the previous direction for warm starting PCG
            self.old_dir = torch.zeros(
                self._numel(), device=self._params[0].device)

        # NOTE: The closure must return both the loss and the gradient
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss, grad_tuple = closure()

        g = torch.cat([grad.view(-1)
                      for grad in grad_tuple if grad is not None])

        # One step update
        for group_idx, group in enumerate(self.param_groups):
            def hvp_temp(x):
                return self._hvp(g, self._params_list, x)

            # Calculate the Newton direction
            d = _nystrom_pcg(hvp_temp, g, self.old_dir,
                             self.mu, self.U, self.S, self.rank, self.cg_tol, self.cg_max_iters)

            # Store the previous direction for warm starting PCG
            self.old_dir = d

            # Check if d is a descent direction
            if torch.dot(d, g) <= 0:
                print("Warning: d is not a descent direction")

            if self.line_search_fn == 'armijo':
                x_init = self._clone_param()

                def obj_func(x, t, dx):
                    self._add_grad(t, dx)
                    loss = float(closure()[0])
                    self._set_param(x)
                    return loss

                # Use -d for convention
                t = _armijo(obj_func, x_init, g, -d, group['lr'])
            else:
                t = group['lr']

            self.state[group_idx]['t'] = t

            # update parameters
            ls = 0
            for p in group['params']:
                np = torch.numel(p)
                dp = d[ls:ls+np].view(p.shape)
                ls += np
                p.data.add_(-dp, alpha=t)

        self.n_iters += 1

        return loss, g

    def update_preconditioner(self, grad_tuple):
        """Update the Nystrom approximation of the Hessian.

        Args:
            grad_tuple (tuple): tuple of Tensors containing the gradients of the loss w.r.t. the parameters.
            This tuple can be obtained by calling torch.autograd.grad on the loss with create_graph=True.
        """

        # Flatten and concatenate the gradients
        gradsH = torch.cat([gradient.view(-1)
                           for gradient in grad_tuple if gradient is not None])

        # Generate test matrix (NOTE: This is transposed test matrix)
        p = gradsH.shape[0]
        Phi = torch.randn(
            (self.rank, p), device=gradsH.device) / (p ** 0.5)
        Phi = torch.linalg.qr(Phi.t(), mode='reduced')[0].t()

        Y = self._hvp_vmap(gradsH, self._params_list)(Phi)

        # Calculate shift
        shift = torch.finfo(Y.dtype).eps
        Y_shifted = Y + shift * Phi

        # Calculate Phi^T * H * Phi (w/ shift) for Cholesky
        choleskytarget = torch.mm(Y_shifted, Phi.t())

        # Perform Cholesky, if fails, do eigendecomposition
        # The new shift is the abs of smallest eigenvalue (negative) plus the original shift
        try:
            C = torch.linalg.cholesky(choleskytarget)
        except:
            # eigendecomposition, eigenvalues and eigenvector matrix
            eigs, eigvectors = torch.linalg.eigh(choleskytarget)
            shift = shift + torch.abs(torch.min(eigs))
            # add shift to eigenvalues
            eigs = eigs + shift
            # put back the matrix for Cholesky by eigenvector * eigenvalues after shift * eigenvector^T
            C = torch.linalg.cholesky(
                torch.mm(eigvectors, torch.mm(torch.diag(eigs), eigvectors.T)))

        try:
            B = torch.linalg.solve_triangular(
                C, Y_shifted, upper=False, left=True)
        # temporary fix for issue @ https://github.com/pytorch/pytorch/issues/97211
        except:
            B = torch.linalg.solve_triangular(C.to('cpu'), Y_shifted.to(
                'cpu'), upper=False, left=True).to(C.device)

        # B = V * S * U^T b/c we have been using transposed sketch
        _, S, UT = torch.linalg.svd(B, full_matrices=False)
        self.U = UT.t()
        self.S = torch.max(torch.square(S) - shift, torch.tensor(0.0))

        self.rho = self.S[-1]

        if self.verbose:
            print(f'Approximate eigenvalues = {self.S}')

    def _hvp_vmap(self, grad_params, params):
        return vmap(lambda v: self._hvp(grad_params, params, v), in_dims=0, chunk_size=self.chunk_size)

    def _hvp(self, grad_params, params, v):
        Hv = torch.autograd.grad(grad_params, params, grad_outputs=v,
                                 retain_graph=True)
        Hv = tuple(Hvi.detach() for Hvi in Hv)
        return torch.cat([Hvi.reshape(-1) for Hvi in Hv])

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # Avoid in-place operation by creating a new tensor
            p.data = p.data.add(
                update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            # Replace the .data attribute of the tensor
            p.data = pdata.data


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe(
    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(
                memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(
                memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(
                min_step, max_step)
        )

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    # type: ignore[possibly-undefined]
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        # type: ignore[possibly-undefined]
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],  # type: ignore[possibly-undefined]
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            # type: ignore[possibly-undefined]
            bracket_g[high_pos] = g_new.clone(
                memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (
                0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                # type: ignore[possibly-undefined]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            # type: ignore[possibly-undefined]
            bracket_g[low_pos] = g_new.clone(
                memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
    return f_new, g_new, t, ls_func_evals


class LBFGS(Optimizer):
    """ 
    Taken from Pytorch implementation of LBFGS. It has been modified to correctly update the progress bar.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1,
        max_iter: int = 20,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: Optional[str] = None,
    ):
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGS doesn't support per-parameter options " "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )

        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset: offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure(update_status=False))
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure):
        """Perform a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group["lr"]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        line_search_fn = group["line_search_fn"]
        history_size = group["history_size"]

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state["func_evals"] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get("d")
        t = state.get("t")
        old_dirs = state.get("old_dirs")
        old_stps = state.get("old_stps")
        ro = state.get("ro")
        H_diag = state.get("H_diag")
        prev_flat_grad = state.get("prev_flat_grad")
        prev_loss = state.get("prev_loss")

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state["n_iter"] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state["n_iter"] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if "al" not in state:
                    state["al"] = [None] * history_size
                al = state["al"]

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(
                    memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state["n_iter"] == 1:
                t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                closure()
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd
                    )
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure(update_status=False))
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state["func_evals"] += ls_func_evals

            # only to update the pbar
            closure()

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state["d"] = d
        state["t"] = t
        state["old_dirs"] = old_dirs
        state["old_stps"] = old_stps
        state["ro"] = ro
        state["H_diag"] = H_diag
        state["prev_flat_grad"] = prev_flat_grad
        state["prev_loss"] = prev_loss

        return orig_loss


class BFGS(Optimizer):
    """
    Implements BFGS algorithm. Chap.6 of `Numerical Optimization` by Nocedal and Wright.

    """

    def __init__(self, params: ParamsT,
                 max_iter: int = 20,
                 tolerance_grad: float = 1e-7,
                 tolerance_change: float = 1e-9,
                 lr: float = 1.0,
                 ):
        defaults = dict(max_iter=max_iter,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, lr=lr)
        super().__init__(params, defaults=defaults)

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset: offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure(False))
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "BFGS requires a closure"
        closure = torch.enable_grad()(closure)

        state = self.state[self._params[0]]
        state.setdefault("n_iter", 0)

        state["H"] = torch.eye(
            self._numel(), device=self._params[0].device, dtype=self._params[0].dtype)
        H = state["H"]

        max_iter = self.defaults['max_iter']
        tol_grad = self.defaults['tolerance_grad']
        tol_change = self.defaults['tolerance_change']
        lr = self.defaults['lr']

        # --- 1) initial eval at x_k
        loss = closure()                  # f = f(x_k)
        grad = self._gather_flat_grad()   # g = ∇f(x_k)
        if grad.norm() < tol_grad:
            return loss

        for _ in range(max_iter-1):
            state["n_iter"] += 1
            # --- 2) compute search direction p_k = -H g
            pk = -H @ grad

            # --- 3) define obj_func for the line-search
            def obj_func(x, t, d):
                return self._directional_evaluate(closure, x, t, d)

            # initial values for Wolfe
            f0 = loss
            g0 = grad
            gtd0 = grad.dot(pk)

            if state["n_iter"] == 1:
                t = min(1.0, 1.0 / grad.abs().sum()) * lr
            else:
                s_prev = state["s_prev"]
                y_prev = state["y_prev"]
                # curvature ratio
                t = s_prev.dot(y_prev) / y_prev.dot(y_prev)
                # optional safeguard: keep t ∈ [1e-3, 10]
                t = max(1e-3, min(t, 10.0))

            # --- 4) call your strong‑Wolfe routine
            x_init = self._clone_param()
            f_new, g_new, t, _ = _strong_wolfe(
                obj_func,
                x=x_init,          # unused in our obj_func
                t=t,
                d=pk,
                f=f0,
                g=g0,
                gtd=gtd0
            )

            # --- 5) perform the accepted step permanently
            s = t * pk
            self._add_grad(t, pk)     # x_{k+1} = x_k + α p_k

            # --- 6) new loss & gradient

            # check convergence
            if g_new.norm() < tol_grad:
                break

            if abs(f_new - loss) < tol_change:
                break

            grad_new = g_new
            loss = f_new

            # --- 7) BFGS update of H
            y = grad_new - grad
            ys = y.dot(s)

            state["s_prev"] = s.clone()
            state["y_prev"] = y.clone()
            if ys <= 0:
                # safeguard: reset H if not positive‑definite
                H = torch.eye(H.size(0), device=H.device, dtype=H.dtype)
                print(
                    f"Warning: H is not positive-definite. Resetting H to identity matrix.")
            else:
                rho = 1.0 / ys
                I = torch.eye(H.size(0), device=H.device, dtype=H.dtype)
                V = I - rho * s.unsqueeze(1) @ y.unsqueeze(0)
                H = V @ H @ V.t() + rho * s.unsqueeze(1) @ s.unsqueeze(0)

            # prepare for next iter
            grad = grad_new

            # only for updating the pbar
            closure()

        # store H back into the state and return final loss
        state["H"] = H
        return loss


class SSBroyden(Optimizer):
    """
    Implements Self-Scaled Broyden. Chap. 6 of `Numerical Optimization` by Nocedal and Wright.
    """

    def __init__(self,
                 params,
                 max_iter: int = 20,
                 tolerance_grad: float = 1e-7,
                 tolerance_change: float = 1e-9,
                 lr: float = 1.0,
                 method: str = "SSBroyden1",
                 initial_scale: bool = False     # only used for SSBroyden1
                 ):
        defaults = dict(
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            lr=lr,
            method=method,
            initial_scale=initial_scale,
        )
        super().__init__(params, defaults)
        # flatten list of parameters we optimize
        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset: offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure(update_status=False))
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SSBroyden requires a closure"
        closure = torch.enable_grad()(closure)

        state = self.state[self._params[0]]
        state.setdefault("n_iter", 0)
        # initialize H if first call
        state.setdefault(
            "H",
            torch.eye(self._numel(),
                      device=self._params[0].device,
                      dtype=self._params[0].dtype)
        )
        H = state["H"]

        max_iter = self.defaults["max_iter"]
        tol_grad = self.defaults["tolerance_grad"]
        tol_change = self.defaults["tolerance_change"]
        lr = self.defaults["lr"]
        method = self.defaults["method"]
        initial_scale = self.defaults["initial_scale"]

        # --- 1) initial evaluation
        loss = closure()
        grad = self._gather_flat_grad()
        if grad.norm() < tol_grad:
            print("Gradient is below tolerance, returning loss")
            state["H"] = H
            return loss

        # --- 2) main loop
        for _ in range(max_iter):
            state["n_iter"] += 1

            # search direction
            pk = -H @ grad

            # prepare strong‐Wolfe call
            def obj_func(x, t, d):
                return self._directional_evaluate(closure, x, t, d)

            x_init = self._clone_param()
            f0, g0 = loss, grad
            gtd0 = grad.dot(pk)

            # initial step-length guess
            if state["n_iter"] == 1:
                t = min(1.0, 1.0 / grad.abs().sum()) * lr
            else:
                s_prev = state["s_prev"]
                y_prev = state["y_prev"]
                t = s_prev.dot(y_prev) / y_prev.dot(y_prev)
                t = max(1e-3, min(t, 10.0))

            # strong‐Wolfe line search
            f_new, g_new, t, _ = _strong_wolfe(
                obj_func,
                x=x_init,
                t=t,
                d=pk,
                f=f0,
                g=g0,
                gtd=gtd0,
            )

            # apply step
            s = t * pk
            self._add_grad(t, pk)

            # new grad & loss
            if g_new.norm() < tol_grad:
                loss = f_new
                break
            if abs(f_new - loss) < tol_change:
                print(
                    f"Loss change is below tolerance, returning loss: {f_new}")
                loss = f_new
                break

            grad_new = g_new
            loss = f_new

            # stash for next iteration
            y = grad_new - grad
            ys = y.dot(s)
            state["s_prev"] = s.clone()
            state["y_prev"] = y.clone()

            # --- 3) Hessian‐approx update
            if method == "SSBroyden1":
                n = H.size(0)
                inv_ys = 1.0 / ys
                H_y = H @ y
                yHy = y.dot(H_y)
                h = yHy * inv_ys
                b = -t * inv_ys * s.dot(grad)
                a = h * b - 1.0

                # build a torch scalar 1.0 on correct device/dtype
                one = torch.tensor(1.0,
                                   device=H.device, dtype=H.dtype)

                # compute θ and τ
                if initial_scale and torch.allclose(H, torch.eye(n, device=H.device, dtype=H.dtype)):
                    rho_m = torch.min(
                        one, h * (one - torch.sqrt(torch.abs(a)/(one+a))))
                    theta_m = (rho_m - one) / a
                    theta_p = one / rho_m
                    theta_sr1 = one / (one - b)
                    if b == 1.0:
                        theta = theta_m
                    elif b > 1.0:
                        theta = torch.max(theta_m, theta_sr1)
                    else:
                        theta = torch.min(theta_p, theta_sr1)
                    tau = h / (one + a * theta)
                else:
                    rho_m = torch.min(
                        one, h * (one - torch.sqrt(torch.abs(a)/(one+a))))
                    theta_m = (rho_m - one) / a
                    theta_p = torch.max(
                        one, b * (one + torch.sqrt(torch.abs(a)/(one+a))))
                    theta_sr1 = one / (one - b)
                    if b == 1.0:
                        theta = theta_m
                    elif b > 1.0:
                        theta = torch.max(theta_m, theta_sr1)
                    else:
                        theta = torch.min(theta_p, theta_sr1)

                    rho_k = torch.min(one, one / b)
                    sigma = one + theta * a
                    sigma_nm1 = torch.abs(sigma) ** (one / (one - n))
                    if theta <= 0.0:
                        tau = torch.min(rho_k * sigma_nm1, sigma)
                    else:
                        tau = rho_k * torch.min(sigma_nm1, one / theta)

                # rank‑2 update
                v = s * inv_ys - H_y / yHy
                phi = (one - theta) / (one + a * theta)
                outer_Hy = H_y.unsqueeze(1) * H_y.unsqueeze(0)
                outer_v = v.unsqueeze(1) * v.unsqueeze(0)
                outer_s = s.unsqueeze(1) * s.unsqueeze(0)

                H = (
                    (H - outer_Hy / yHy + phi * yHy * outer_v) / tau
                    + inv_ys * outer_s
                )
            else:
                raise ValueError(
                    f"Unknown method {method}. Supported methods are: SSBroyden1")
            # prepare next iter
            grad = grad_new

            closure()

        # save H back
        state["H"] = H
        return loss
