import unittest
import numpy as np
from mnist_numpy.network import MLP


class TestMLP(unittest.TestCase):
    def test_softmax_probabilities_sum_to_one(self):
        model = MLP(input_dim=3, hidden_dim=4, output_dim=2)
        _, _, p = model.forward(np.array([[0.2, 0.4, 0.6], [1, 0, 0]], dtype=np.float32))
        np.testing.assert_allclose(p.sum(axis=1), np.ones(2), atol=1e-6)

    def test_finite_difference_gradient(self):
        model = MLP(input_dim=3, hidden_dim=4, output_dim=2, seed=7)
        x = np.array([[0.3, 0.5, 0.1], [0.9, 0.2, 0.8]], dtype=np.float32)
        y = np.array([0, 1], dtype=np.uint8)
        _, gradients = model.loss_and_gradients(x, y)
        eps = 1e-3
        for param, analytic in zip((model.w1, model.b1, model.w2, model.b2), gradients):
            # A handful of analytical vs. finite-difference coordinates.
            for index in list(np.ndindex(param.shape))[:4]:
                original = param[index]
                param[index] = original + eps
                loss_plus, _ = model.loss_and_gradients(x, y)
                param[index] = original - eps
                loss_minus, _ = model.loss_and_gradients(x, y)
                param[index] = original
                numeric = (loss_plus - loss_minus) / (2 * eps)
                self.assertAlmostEqual(float(analytic[index]), numeric, delta=2e-3)

    def test_training_reduces_loss(self):
        rng = np.random.default_rng(123)
        x = rng.random((16, 3), dtype=np.float32)
        y = (x[:, 0] > x[:, 1]).astype(np.uint8)
        model = MLP(input_dim=3, hidden_dim=8, output_dim=2, seed=7)
        initial, _ = model.loss_and_gradients(x, y)
        for _ in range(200):
            model.train_batch(x, y, 0.2)
        final, _ = model.loss_and_gradients(x, y)
        self.assertLess(final, initial)


if __name__ == "__main__":
    unittest.main()
