import argparse
from convnet import *
from training import *
from testing import *


parser = argparse.ArgumentParser()
parser.add_argument('--train_csv', type=str, default="./train.csv")
parser.add_argument('--val_csv', type=str, default="./Dig-MNIST.csv")
parser.add_argument('--test_csv', type=str, default="./test.csv")
parser.add_argument('--sample_output', type=str, default="./sample_submission.csv")
parser.add_argument('--output_file', type=str, default="./test_output.csv")
parser.add_argument('--model_path', type=str, default="./nn_model/")
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--test', type=bool, default=True)
args = parser.parse_args()


def main():
    #selects gpu if available, otherwise runs on the cpu.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = network().to(device)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    if (args.train):
        print ("Begin Training.")
        train_network(net, device, args.train_csv, args.val_csv, args.model_path)
        print ("Training Complete.")

    if (args.test):
        test_network(net, device, args.test_csv, args.model_path, args.sample_output, args.output_file)
        print ("Results written to file.")

    return


if __name__=="__main__":
    main()
