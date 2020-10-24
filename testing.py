from dataset import *
from torch.utils.data import DataLoader
import os


def test_network(net, device, test_csv, model_path, sample_output, output_file):
    """
    @brief: Load the model and write test results to output csv file.
    """
    try:
        net.load_state_dict(torch.load(os.path.join(model_path,"model.ckpt")))
    except:
        print ("Trained model does not exist.")
        raise

    #Reading test data
    test_images = read_data("test", test_csv)
    test_set = KannadaDataset(images=test_images, transforms=test_transforms())
    test_data = DataLoader(test_set, batch_size=1, shuffle=False)

    net.eval()
    predictions = torch.LongTensor().to(device) #Tensor to store model predictions.
    with torch.no_grad():
        for images in test_data:
            images = images.to(device)

            outputs = net(images)
            predictions = torch.cat((predictions, outputs.argmax(dim=1)), dim=0)

    #Write test results to output file.
    submission = pd.read_csv(sample_output)
    submission['label'] = predictions.cpu().numpy()
    submission.to_csv(output_file, index=False)

    return
