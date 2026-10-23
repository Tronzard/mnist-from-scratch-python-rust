#Implementation from scratch without ML libraries
import numpy as np


class MLP:
    def __init__(self, input_dim: int = 784, hidden_dim: int = 64, output_dim: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)

        #Layer1
        # He uniform initialization to keep scale of singals stable when passing through ReLU
        limit = np.sqrt(6.0 / input_dim)

        #Random initialization of weights and biases
        self.w1 = rng.uniform(-limit, limit, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        
        #Layer 2
        #Xavier uniform initialization
        bound = np.sqrt(6.0 / (hidden_dim + output_dim))

        #Random initialization of weights and biases
        self.w2 = rng.uniform(-bound, bound, (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #Layer1, weighted sum 
        z1 = x @ self.w1 + self.b1

        #relu
        a1 = np.maximum(z1, 0)

        #Layer2
        logits = a1 @ self.w2 + self.b2

        #Subtracting for removing enumerical overflow during power or e during softmax
        shifted = logits - np.max(logits, axis=1, keepdims=True)

        #Softmax
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return z1, a1, probs

    def loss_and_gradients(self, x: np.ndarray, y: np.ndarray):
        z1, a1, probs = self.forward(x)
        batch = x.shape[0]
        res = np.clip(probs[np.arange(batch), y])

        #Preventing log0, calculating negative cross entropy
        loss = -np.log(res, 0.000000000001, 1).mean()

        # dL/dlogits = softmax(logits) - one_hot(y), scaled by batch size
        dlogits = probs.copy()
        dlogits[np.arange(batch), y] -= 1.0
        dlogits /= batch
        dw2 = a1.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dz1 = (dlogits @ self.w2.T) * (z1 > 0)
        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        return float(loss), (dw1, db1, dw2, db2)

    def train_batch(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        loss, grads = self.loss_and_gradients(x, y)
        for param, grad in zip((self.w1, self.b1, self.w2, self.b2), grads):
            param -= lr * grad
        return loss

    def accuracy(self, x: np.ndarray, y: np.ndarray, batch_size: int = 512) -> float:
        correct = 0
        for start in range(0, len(x), batch_size):
            _, _, probs = self.forward(x[start:start + batch_size])
            correct += np.count_nonzero(np.argmax(probs, axis=1) == y[start:start + batch_size])
        return correct / len(x)
