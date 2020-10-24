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


    test_images = read_data("test", test_csv)
    test_set = KannadaDataset(images=test_images, transforms=test_transforms())
    test_data = DataLoader(test_set, batch_size=1, shuffle=False)

    net.eval()
    predictions = torch.LongTensor().to(device)
    with torch.no_grad():
        for images in test_data:
            images = images.to(device)

            outputs = net(images)
            predictions = torch.cat((predictions, outputs.argmax(dim=1)), dim=0)

    submission = pd.read_csv(sample_output)
    submission['label'] = predictions.cpu().numpy()
    submission.to_csv(output_file, index=False)

    return
