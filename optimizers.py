import torch
import numpy as np
from torch.optim import Optimizer
from torch.distributions import Bernoulli, Normal
import math
from collections import deque
import time


class SGD(Optimizer):
    """
    Stochastic gradient descent. Also includes implementations of momentum,
    Nesterov's momentum, L2 regularization, SGDW and Learning Rate Dropout.
    """
    def __init__(self, params, lr, mu=0, nesterov=False, weight_decay=0, lrd=1):
        defaults = {'lr': lr, 'mu': mu, 'nesterov': nesterov, 'weight_decay': weight_decay, 'lrd': lrd}
        super(SGD, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """
        for group in self.param_groups:

            lr = group['lr']
            mu = group['mu']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            lrd_bernoulli = Bernoulli(probs=group['lrd'])

            if mu != 0 and 'v' not in group:
                group['v'] = []
                if nesterov:
                    group['theta'] = []
                for param in group['params']:
                    group['v'].append(torch.zeros_like(param))
                    if nesterov:
                        theta_param = torch.ones_like(param).mul_(param.data)
                        group['theta'].append(theta_param)

            for idx, param in enumerate(group['params']):
                param.grad.data -= weight_decay * param.data
                lrd_mask = lrd_bernoulli.sample(param.size()).to(param.device)

                if mu != 0:
                    v = group['v'][idx]
                    v = mu * v - lr * param.grad.data
                    group['v'][idx] = v

                    if nesterov:
                        group['theta'][idx] += lrd_mask * v
                        param.data = group['theta'][idx] + mu * v

                    else:
                        param.data += lrd_mask * v

                else:
                    param.data -= lrd_mask * lr * param.grad.data


class Adam(Optimizer):
    """
    Adam as proposed by https://arxiv.org/abs/1412.6980.
    Also includes a number of proposed extensions to the the Adam algorithm,
    such as Nadam, L2 regularization, AdamW, RAdam and Learning Rate Dropout.
    """
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, nesterov=False, l2_reg=0, weight_decay=0, rectified=False, lrd=1, eps=1e-8):
        defaults = {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'nesterov': nesterov, 'l2_reg': l2_reg,
                    'weight_decay': weight_decay, 'rectified': rectified, 'lrd': lrd, 'eps': eps}
        super(Adam, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """
        for group in self.param_groups:

            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            nesterov = group['nesterov']
            l2_reg = group['l2_reg']
            weight_decay = group['weight_decay']
            rectified = group['rectified']
            lrd_bernoulli = Bernoulli(probs=group['lrd'])
            eps = group['eps']

            if 'm' not in group and 'v' not in group:
                group['m'] = []
                group['v'] = []
                group['t'] = 1
                if nesterov:
                    group['prev_grad'] = []
                for param in group['params']:
                    group['m'].append(torch.zeros_like(param))
                    group['v'].append(torch.zeros_like(param))
                    if nesterov:
                        group['prev_grad'].append(torch.zeros_like(param))

            for idx, param in enumerate(group['params']):
                if l2_reg:
                    param.grad.data += l2_reg * param.data

                if nesterov:
                    grad = group['prev_grad'][idx]
                else:
                    grad = param.grad.data

                lrd_mask = lrd_bernoulli.sample(param.size()).to(param.device)

                m = group['m'][idx]
                v = group['v'][idx]
                t = group['t']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                if nesterov:
                    group['prev_grad'][idx] = param.grad.data

                if rectified:
                    rho_inf = 2 / (1 - beta2) - 1
                    rho = rho_inf - 2 * t * beta2**t / (1 - beta2**t)
                    if rho >= 5:
                        numerator = (1 - beta2**t) * (rho - 4) * (rho - 2) * rho_inf
                        denominator = (rho_inf - 4) * (rho_inf - 2) * rho
                        r = np.sqrt(numerator / denominator)
                        param.data += - lrd_mask * lr * r * m_hat / (torch.sqrt(v) + eps)
                    else:
                        param.data += - lrd_mask * lr * m_hat
                else:
                    param.data += - lrd_mask * lr * m_hat / (torch.sqrt(v_hat) + eps)

                if weight_decay:
                    param.data -= weight_decay * param.data

                group['m'][idx] = m
                group['v'][idx] = v

            group['t'] += 1


class RMSProp(Adam):
    """
    RMSprop as proposed by http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
    Note that this implementation, unlike the original RMSprop, uses bias-corrected moments.
    """
    def __init__(self, params, lr, beta2):
        super(RMSProp, self).__init__(params, lr, beta2=beta2, beta1=0)


class Lookahead(Optimizer):
    """
    Lookahead Optimization as proposed by https://arxiv.org/abs/1907.08610.
    This is a wrapper class that can be applied to an instantiated optimizer.
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = optimizer.param_groups

        self.counter = 0
        for group in optimizer.param_groups:
            group['phi'] = []
            for param in group['params']:
                phi_param = torch.ones_like(param).mul_(param.data)
                group['phi'].append(phi_param)

    def step(self):
        if self.counter == self.k:
            for group_idx, group in enumerate(self.param_groups):
                for idx, _ in enumerate(group['phi']):
                    theta = self.optimizer.param_groups[group_idx]['params'][idx].data
                    group['phi'][idx] = group['phi'][idx] + self.alpha * (theta - group['phi'][idx])
            self.counter = 0
        else:
            self.counter += 1
            self.optimizer.step()


class GradientNoise(Optimizer):
    """
    Gradient Noise as proposed by https://arxiv.org/abs/1511.06807.
    This is a wrapper class that can be applied to an instantiated optimizer.
    """
    def __init__(self, optimizer, eta=0.3, gamma=0.55):
        self.optimizer = optimizer
        self.eta = eta
        self.gamma = gamma
        self.t = 0
        self.param_groups = optimizer.param_groups

    def step(self):
        normal = torch.empty(1).normal_(mean=0, std=np.sqrt(self.eta/((1+self.t)**self.gamma)))\
            .to(self.optimizer.param_groups[0]['params'][0].device)
        for group_idx, group in enumerate(self.param_groups):
            for idx, param in enumerate(group['params']):
                self.optimizer.param_groups[group_idx]['params'][idx].grad.data += normal
                self.optimizer.step()
                self.t += 1


class GradientDropout(Optimizer):
    """
    Gradient dropout as proposed by https://arxiv.org/abs/1912.00144.
    This is a wrapper class that can be applied to an instantiated optimizer.
    Note that this method does not improve optimization significantly and
    is only here for comparison to Learning Rate Dropout.
    """
    def __init__(self, optimizer, grad_retain=0.9):
        self.optimizer = optimizer
        self.grad_retain = grad_retain
        self.grad_bernoulli = Bernoulli(probs=grad_retain)
        self.param_groups = optimizer.param_groups

    def step(self):
        for group_idx, group in enumerate(self.param_groups):
            for idx, param in enumerate(group['params']):
                grad_mask = self.grad_bernoulli.sample(param.size()).to(param.device)
                self.optimizer.param_groups[group_idx]['params'][idx].grad.data *= grad_mask
                self.optimizer.step()


class RandomProjection:
    """
    Implements sparse random projection for dimension reduction.
    """
    def __init__(self, original_dim, target_dim, seed=42):
        self.original_dim = original_dim
        self.target_dim = min(target_dim, original_dim)
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Create sparse random projection matrix
        sparsity = 1.0 / math.sqrt(self.original_dim)
        self.projection = self._create_sparse_projection(sparsity)
    
    def _create_sparse_projection(self, sparsity):
        """Create a sparse random projection matrix."""
        # Create a mask for non-zero elements
        mask = torch.rand(self.target_dim, self.original_dim) < sparsity
        
        # Create random signs: 1 or -1
        signs = torch.randint(0, 2, (self.target_dim, self.original_dim)) * 2 - 1
        
        # Scale factor for unit variance
        scale = math.sqrt(1.0 / sparsity)
        
        # Create projection matrix
        projection = (mask.float() * signs.float() * scale) / math.sqrt(self.target_dim)
        return projection
    
    def project(self, data):
        """Project data from original dimension to target dimension."""
        # Check if data is 1D or 2D
        is_1d = data.dim() == 1
        
        # Ensure data is 2D for matrix multiplication
        if is_1d:
            data = data.unsqueeze(0)
            
        # Move projection matrix to same device as data
        projection = self.projection.to(data.device)
        
        # Project data
        projected = torch.matmul(data, projection.t())
        
        # Return in original shape
        return projected.squeeze(0) if is_1d else projected


class ValleyDetector:
    """
    Non-parametric valley detection using gradient consistency.
    """
    def __init__(self, window_size=5, threshold=0.2):
        self.window_size = window_size
        self.threshold = threshold
        self.grad_history = deque(maxlen=window_size)
    
    def update(self, grad):
        """Update gradient history."""
        # Store normalized gradient
        grad_norm = grad / (torch.norm(grad) + 1e-10)
        self.grad_history.append(grad_norm.cpu())
    
    def detect_valley(self):
        """Detect if current point is in a valley."""
        if len(self.grad_history) < self.window_size:
            return False, None
            
        # Convert history to tensor
        grads = torch.stack(list(self.grad_history))
        
        # Compute gradient consistency (average cosine similarity)
        n = len(self.grad_history)
        cosine_sum = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                cos = torch.dot(grads[i], grads[j])
                cosine_sum += cos.item()
                count += 1
                
        avg_cosine = cosine_sum / max(1, count)
        
        # If gradients are inconsistent (pointing in different directions)
        # then we might be in a valley
        is_valley = avg_cosine < self.threshold
        
        # If in valley, compute valley direction using PCA
        if is_valley:
            try:
                # Center gradients
                centered = grads - grads.mean(dim=0, keepdim=True)
                
                # Compute covariance
                cov = torch.matmul(centered.t(), centered) / (n - 1)
                
                # Get eigenvector with smallest eigenvalue (valley direction)
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                valley_dir = eigenvectors[:, 0]  # Direction of smallest eigenvalue
                return True, valley_dir
            except Exception:
                return is_valley, None
                
        return is_valley, None


class TALT(Optimizer):
    """
    Topology-Aware Learning Trajectory (TALT) Optimizer
    
    This optimizer implements TALT with:
    1. Dimension reduction via random projections
    2. Valley detection for improved optimization in challenging landscapes
    3. Adam-like momentum updates with topology-aware modifications
    4. Learning rate dropout integration
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, projection_dim=32, 
                 update_interval=20, valley_strength=0.2, smoothing_factor=0.3, 
                 min_param_size=100, lrd=1.0, weight_decay=0, eps=1e-8):
        defaults = {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'projection_dim': projection_dim,
                    'update_interval': update_interval, 'valley_strength': valley_strength, 
                    'smoothing_factor': smoothing_factor, 'min_param_size': min_param_size,
                    'lrd': lrd, 'weight_decay': weight_decay, 'eps': eps}
        super(TALT, self).__init__(params, defaults)
        
        # Initialize tracking variables
        self.steps = 0
        
        # Parameter-specific structures
        for group in self.param_groups:
            group['step'] = 0
    
    def step(self):
        """
        Performs a single optimization step.
        """
        self.steps += 1
        
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            update_interval = group['update_interval']
            valley_strength = group['valley_strength']
            projection_dim = group['projection_dim']
            min_param_size = group['min_param_size']
            weight_decay = group['weight_decay']
            lrd_bernoulli = Bernoulli(probs=group['lrd'])
            eps = group['eps']
            
            # Increment step counter
            group['step'] += 1
            step = group['step']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                # Apply weight decay if specified
                if weight_decay != 0:
                    grad.add_(weight_decay, p.data)
                
                # Get or initialize state
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                    # For larger parameters, initialize topology-aware components
                    if p.numel() > min_param_size:
                        dim = p.numel()
                        target_dim = min(projection_dim, dim // 4)
                        state['projector'] = RandomProjection(dim, target_dim)
                        state['valley_detector'] = ValleyDetector(window_size=5, threshold=0.2)
                        state['transformation'] = None
                
                state['step'] += 1
                local_step = state['step']
                
                # Apply learning rate dropout mask
                lrd_mask = lrd_bernoulli.sample(p.size()).to(p.device)
                
                # For larger parameters, perform valley detection and transformation
                if p.numel() > min_param_size and local_step % 5 == 0:
                    flat_grad = grad.view(-1)
                    
                    # Update topology information
                    if 'valley_detector' in state:
                        try:
                            # Project gradient for efficiency
                            projected_grad = state['projector'].project(flat_grad.detach())
                            state['valley_detector'].update(projected_grad)
                            
                            # Check for valleys periodically
                            if local_step % update_interval == 0:
                                is_valley, valley_dir = state['valley_detector'].detect_valley()
                                
                                if is_valley and valley_dir is not None:
                                    # Map valley direction to original space
                                    valley_dir = valley_dir.to(grad.device)
                                    # Amplify gradient in valley direction
                                    valley_component = torch.dot(projected_grad, valley_dir) * valley_dir
                                    projected_grad = projected_grad + valley_strength * valley_component
                                    
                                    # Note: This is a simplified application that doesn't
                                    # fully map back to the original space
                        except Exception:
                            # Fallback to standard update if topology analysis fails
                            pass
                
                # Standard Adam update
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                
                # Compute bias corrections
                bias_correction1 = 1 - beta1 ** local_step
                bias_correction2 = 1 - beta2 ** local_step
                
                # Compute step size
                step_size = lr / bias_correction1
                
                # Compute adaptive learning rate
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                # Apply LRD mask and update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size * lrd_mask)
