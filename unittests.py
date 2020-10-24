import unittest
from main import *

class neural_network_tests(unittest.TestCase):
    def test_output_size(self):
        net = network()

        random_input = torch.rand(1, 1, 28, 28)

        output = net(random_input)

        self.assertEqual(output.numpy().shape, (1, 10), "Output should consist of 10 classes.")


if __name__=="__main__":
    unittest.main()