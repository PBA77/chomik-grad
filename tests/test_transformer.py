import unittest

import numpy as np

from chomikgrad import SGD, Tensor, TransformerEncoderBlock


class TransformerTests(unittest.TestCase):
    def test_encoder_forward_backward_and_update(self) -> None:
        rng = np.random.default_rng(11)
        block = TransformerEncoderBlock(8, 2, 16, rng=rng)
        inputs = Tensor(rng.normal(size=(2, 4, 8)).astype(np.float32))

        output = block(inputs)
        self.assertEqual(output.shape, (2, 4, 8))
        loss = (output * output).mean()
        loss.backward()

        parameters = block.parameters()
        self.assertEqual(len(parameters), 16)
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.shape, parameter.shape)

        before = parameters[0].numpy().copy()
        SGD(parameters, lr=0.01).step()
        self.assertFalse(np.array_equal(before, parameters[0].numpy()))


if __name__ == "__main__":
    unittest.main()
