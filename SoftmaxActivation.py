import numpy as np

class SoftmaxActivation:
    def __init__(self):
        pass
    
    def forward(self, z):
        """Compute the softmax of vector z in a numerically stable way."""
        exp_z = np.exp(z - np.max(z))  # Subtract max to prevent overflow
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward(self, dout, cache):
        """
        Backward pass for the softmax function.
        Arguments:
        dout -- derivative of the loss with respect to the softmax output
        cache -- cached values from the forward pass
        """
        z = cache
        dz = self.forward(z)
        dz[range(len(dz)), np.argmax(dz, axis=1)] -= 1  # Adjust gradient
        return dz * dout

# Example usage:
softmax = SoftmaxActivation()

# Example forward pass
logits = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
probs = softmax.forward(logits)
print("Softmax probabilities:", probs)

# Example backward pass (with dummy gradient)
dummy_dout = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
grad = softmax.backward(dummy_dout, logits)
print("Gradient:", grad)
