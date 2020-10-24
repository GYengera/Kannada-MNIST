import unittest
from main import *

class neural_network_tests(unittest.TestCase):
    def test_output_size(self):
        net = network()

        random_input = torch.rand(5, 1, 28, 28)

        output = net(random_input)

        self.assertEqual(output.detach().numpy().shape, (5, 10), "Output is of incorrect dimensions.")


if __name__=="__main__":
    unittest.main()