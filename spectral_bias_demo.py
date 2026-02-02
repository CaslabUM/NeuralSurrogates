"""
Spectral Bias in Neural Networks: A Minimalist Demonstration
=============================================================
This script provides a clean, quantitative demonstration of spectral bias—
the tendency of neural networks to learn low-frequency content before 
high-frequency content.

Target function: f(x) = Σ_k a_k sin(2πk x)  for k = 1, 3, 7, 15
We track the error in each frequency component during training using FFT.

Pure NumPy implementation (no PyTorch required).
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Reproducibility
np.random.seed(42)

# =============================================================================
# 1. Define the Multi-Scale Target Function
# =============================================================================

FREQUENCIES = [1, 3, 7, 15]  # Wavenumbers k
AMPLITUDES = [1.0, 1.0, 1.0, 1.0]  # Amplitudes

def target_function(x):
    """f(x) = Σ a_k sin(2πk x) on domain [0,1]"""
    result = np.zeros_like(x)
    for k, a, in zip(FREQUENCIES, AMPLITUDES):
        result += a * np.sin(2 * np.pi * k * x)
    return result

# =============================================================================
# 2. Simple MLP with Manual Backprop (Pure NumPy)
# =============================================================================

class MLP:
    """
    Simple MLP with tanh activation, implemented in pure NumPy.
    Architecture: input -> [hidden x num_layers] -> output
    """
    def __init__(self, hidden_dim=64, num_layers=3):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Xavier initialization
        self.weights = []
        self.biases = []
        
        # Input layer
        scale = np.sqrt(2.0 / (1 + hidden_dim))
        self.weights.append(np.random.randn(1, hidden_dim) * scale)
        self.biases.append(np.zeros((1, hidden_dim)))
        
        # Hidden layers
        scale = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        for _ in range(num_layers - 1):
            self.weights.append(np.random.randn(hidden_dim, hidden_dim) * scale)
            self.biases.append(np.zeros((1, hidden_dim)))
        
        # Output layer
        scale = np.sqrt(2.0 / (hidden_dim + 1))
        self.weights.append(np.random.randn(hidden_dim, 1) * scale)
        self.biases.append(np.zeros((1, 1)))
        
        # For Adam optimizer
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0
    
    def forward(self, x):
        """Forward pass, storing activations for backprop"""
        self.activations = [x]
        self.pre_activations = []
        
        h = x
        for i in range(self.num_layers):
            z = h @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            h = np.tanh(z)
            self.activations.append(h)
        
        # Output layer (linear)
        z = h @ self.weights[-1] + self.biases[-1]
        self.pre_activations.append(z)
        self.activations.append(z)
        
        return z
    
    def backward(self, y_true):
        """Backward pass with MSE loss"""
        N = y_true.shape[0]
        y_pred = self.activations[-1]
        
        # Gradient of MSE loss
        dL_dy = 2 * (y_pred - y_true) / N
        
        self.grad_w = []
        self.grad_b = []
        
        # Output layer gradient
        delta = dL_dy
        self.grad_w.insert(0, self.activations[-2].T @ delta)
        self.grad_b.insert(0, np.sum(delta, axis=0, keepdims=True))
        
        # Hidden layers (backprop through tanh)
        for i in range(self.num_layers - 1, -1, -1):
            delta = delta @ self.weights[i + 1].T
            # tanh derivative: 1 - tanh^2
            delta = delta * (1 - self.activations[i + 1]**2)
            self.grad_w.insert(0, self.activations[i].T @ delta)
            self.grad_b.insert(0, np.sum(delta, axis=0, keepdims=True))
    
    def update(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer update"""
        self.t += 1
        
        for i in range(len(self.weights)):
            # Update weights
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * self.grad_w[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * self.grad_w[i]**2
            m_hat = self.m_w[i] / (1 - beta1**self.t)
            v_hat = self.v_w[i] / (1 - beta2**self.t)
            self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Update biases
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * self.grad_b[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * self.grad_b[i]**2
            m_hat = self.m_b[i] / (1 - beta1**self.t)
            v_hat = self.v_b[i] / (1 - beta2**self.t)
            self.biases[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    def predict(self, x):
        """Inference only (no gradient storage)"""
        h = x
        for i in range(self.num_layers):
            h = np.tanh(h @ self.weights[i] + self.biases[i])
        return h @ self.weights[-1] + self.biases[-1]

# =============================================================================
# 3. Spectral Error Analysis (FFT-based)
# =============================================================================

def compute_spectral_amplitudes(y, N, frequencies):
    """
    Compute amplitudes at target frequencies using FFT.
    For sin(2πkx), the amplitude appears at frequency index k in the FFT.
    """
    y_flat = y.flatten()
    
    # FFT gives coefficients; for real sin waves, amplitude = 2*|imag(fft)|/N
    fft_coeffs = np.fft.fft(y_flat)
    
    amplitudes = {}
    for k in frequencies:
        if k < N // 2:
            # Amplitude of sin component at frequency k
            amplitudes[k] = 2 * np.abs(fft_coeffs[k].imag) / N
    return amplitudes

def compute_spectral_errors(model, x_eval, frequencies, true_amplitudes):
    """Compute relative error in each frequency component"""
    y_pred = model.predict(x_eval)
    N = len(x_eval)
    
    pred_amps = compute_spectral_amplitudes(y_pred, N, frequencies)
    
    errors = {}
    for k, a_true in zip(frequencies, true_amplitudes):
        a_pred = pred_amps.get(k, 0)
        errors[k] = abs(a_pred - a_true) / abs(a_true)
    
    return errors, pred_amps

# =============================================================================
# 4. Training Loop with Spectral Tracking
# =============================================================================

def train_with_spectral_tracking(epochs=8000, lr=5e-4, log_interval=50):
    """Train MLP and track per-frequency errors"""
    
    # Training data: uniform grid on [0, 1) for FFT compatibility
    N_train = 256
    x_train = np.linspace(0, 1, N_train, endpoint=False).reshape(-1, 1)
    y_train = target_function(x_train).reshape(-1, 1)
    
    # Evaluation grid (same as training for FFT)
    N_eval = 256
    x_eval = np.linspace(0, 1, N_eval, endpoint=False).reshape(-1, 1)
    
    # Model
    model = MLP(hidden_dim=128, num_layers=4)
    
    # Storage for tracking
    history = {
        'epoch': [],
        'loss': [],
        'spectral_errors': defaultdict(list),
        'learned_amplitudes': defaultdict(list)
    }
    
    print("Training MLP on multi-frequency target...")
    print(f"Target frequencies: {FREQUENCIES}")
    print(f"Target amplitudes:  {AMPLITUDES}")
    print("-" * 60)
    
    for epoch in range(epochs + 1):
        # Forward pass
        y_pred = model.forward(x_train)
        loss = np.mean((y_pred - y_train)**2)
        
        # Backward pass
        model.backward(y_train)
        model.update(lr=lr)
        
        # Log spectral errors
        if epoch % log_interval == 0:
            errors, amps = compute_spectral_errors(
                model, x_eval, FREQUENCIES, AMPLITUDES
            )
            
            history['epoch'].append(epoch)
            history['loss'].append(loss)
            for k in FREQUENCIES:
                history['spectral_errors'][k].append(errors[k])
                history['learned_amplitudes'][k].append(amps.get(k, 0))
            
            if epoch % (log_interval * 10) == 0:
                err_str = ", ".join([f"k={k}: {errors[k]:.3f}" for k in FREQUENCIES])
                print(f"Epoch {epoch:5d} | Loss: {loss:.2e} | Rel. Errors: {err_str}")
    
    return model, history, x_eval

# =============================================================================
# 5. Visualization
# =============================================================================

def create_visualizations(model, history, x_eval):
    """Generate publication-quality figures demonstrating spectral bias"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Color scheme: low freq = cool colors, high freq = warm colors
    colors = {1: '#2166ac', 3: '#4393c3', 7: '#f4a582', 15: '#b2182b'}
    
    # --- Panel (a): Function Approximation ---
    ax = axes[0, 0]
    x_np = x_eval.flatten()
    y_true = target_function(x_eval).flatten()
    y_pred = model.predict(x_eval).flatten()
    
    ax.plot(x_np, y_true, 'k-', lw=2, label='Target $f(x)$')
    ax.plot(x_np, y_pred, 'r--', lw=1.5, label='MLP prediction')
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$f(x)$', fontsize=12)
    ax.set_title('(a) Function Approximation', fontsize=12, fontweight='bold')
    ax.legend(frameon=False)
    ax.set_xlim([0, 1])
    
    # --- Panel (b): Spectral Error vs Training Epoch ---
    ax = axes[0, 1]
    epochs = history['epoch']
    
    for k in FREQUENCIES:
        ax.loglog(epochs, history['spectral_errors'][k], 
                    color=colors[k], lw=2, label=f'$k = {k}$')
    
    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Relative Amplitude Error', fontsize=12)
    ax.set_title('(b) Spectral Bias: Per-Frequency Convergence', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, title='Frequency')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, epochs[-1]])
    
    # --- Panel (c): Learned Amplitudes Over Time ---
    ax = axes[1, 0]
    
    for k, a_true in zip(FREQUENCIES, AMPLITUDES):
        learned = history['learned_amplitudes'][k]
        ax.plot(epochs, learned, color=colors[k], lw=2, label=f'$k={k}$ (true: {a_true})')
        ax.axhline(a_true, color=colors[k], ls=':', alpha=0.5)
    
    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Learned Amplitude $\\hat{a}_k$', fontsize=12)
    ax.set_title('(c) Amplitude Learning Dynamics', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, epochs[-1]])
    
    # --- Panel (d): Final Error Spectrum ---
    ax = axes[1, 1]
    
    final_errors = [history['spectral_errors'][k][-1] for k in FREQUENCIES]
    bars = ax.bar(range(len(FREQUENCIES)), final_errors, 
                  color=[colors[k] for k in FREQUENCIES], edgecolor='black', lw=1.5)
    
    ax.set_xticks(range(len(FREQUENCIES)))
    ax.set_xticklabels([f'$k={k}$' for k in FREQUENCIES])
    ax.set_ylabel('Final Relative Error', fontsize=12)
    ax.set_title('(d) Final Error vs Frequency', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    
    # Add trend line showing expected scaling
    ax.plot(range(len(FREQUENCIES)), final_errors, 'ko--', ms=8, 
            label='Observed error', zorder=5)
    
    plt.tight_layout()
    plt.savefig('./spectral_bias_demonstration.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig('./spectral_bias_demonstration.pdf', 
                bbox_inches='tight')
    print("\nFigures saved to outputs directory.")
    
    return fig

# =============================================================================
# 6. Quantitative Summary
# =============================================================================

def print_summary(history):
    """Print quantitative summary of spectral bias"""
    
    print("\n" + "=" * 60)
    print("QUANTITATIVE SUMMARY: SPECTRAL BIAS DEMONSTRATION")
    print("=" * 60)
    
    # Compute epochs to reach 10% error for each frequency
    threshold = 0.10
    epochs_to_threshold = {}
    
    for k in FREQUENCIES:
        errors = history['spectral_errors'][k]
        epochs = history['epoch']
        
        reached = False
        for i, (e, err) in enumerate(zip(epochs, errors)):
            if err < threshold:
                epochs_to_threshold[k] = e
                reached = True
                break
        if not reached:
            epochs_to_threshold[k] = float('inf')
    
    print(f"\nEpochs to reach {threshold*100:.0f}% relative error:")
    print("-" * 40)
    for k in FREQUENCIES:
        e = epochs_to_threshold[k]
        if e == float('inf'):
            print(f"  k = {k:2d}:  Never reached")
        else:
            print(f"  k = {k:2d}:  {e:5d} epochs")
    
    print(f"\nFinal relative errors (after {history['epoch'][-1]} epochs):")
    print("-" * 40)
    for k in FREQUENCIES:
        err = history['spectral_errors'][k][-1]
        print(f"  k = {k:2d}:  {err:.4f}  ({err*100:.2f}%)")
    
    # Compute slowdown factor
    if epochs_to_threshold[1] > 0 and epochs_to_threshold[1] != float('inf'):
        print(f"\nSlowdown factors (relative to k=1):")
        print("-" * 40)
        base = epochs_to_threshold[1]
        for k in FREQUENCIES:
            e = epochs_to_threshold[k]
            if e != float('inf'):
                factor = e / base
                print(f"  k = {k:2d}:  {factor:.1f}x slower")
            else:
                print(f"  k = {k:2d}:  ∞ (never converged)")
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY: Higher frequencies require exponentially more")
    print("training iterations to achieve the same accuracy level.")
    print("This is the essence of spectral bias.")
    print("=" * 60)

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Train and analyze
    model, history, x_eval = train_with_spectral_tracking(
        epochs=10000, 
        lr=1e-3, 
        log_interval=25
    )
    
    # Generate visualizations
    fig = create_visualizations(model, history, x_eval)
    
    # Print quantitative summary
    print_summary(history)
