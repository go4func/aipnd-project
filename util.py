
import argparse
import torch
from torch import optim
import torch.utils.data as D
from torchvision import datasets, transforms, models
from PIL import Image
import json
from fc_classifier import Classifier


def train_args():
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Train the network and use it to classify the flowers',
                    epilog='Text at the bottom of help')

    parser.add_argument('--data_dir', default='flowers',
                        help='location of train, valid and test data')
    parser.add_argument('--checkpoint', default='./',
                        help='location for saving trained network checkpoint')
    parser.add_argument(
        '--arch', choices=['vgg16_bn', 'densenet121'], default='vgg16_bn', help='choose architecture')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='learning rate of network')
    parser.add_argument('--hidden_units', default=4096, type=int,
                        help='units of hidden layer')
    parser.add_argument('--gpu', action='store_true',
                        help='user gpu to train network')
    parser.add_argument('--epochs', default=1, type=int,
                        help='training epochs')

    args = parser.parse_args()
    return args.data_dir, args.checkpoint, args.arch, args.learning_rate, args.hidden_units, args.gpu, args.epochs


def predict_args():
    parser = argparse.ArgumentParser(
        prog='predict.py',
        description='Predict the image',
                    epilog='Text at the bottom of help')
    parser.add_argument('--image',
                        help='path to the image to be predicted')
    parser.add_argument('--checkpoint',
                        help='the model checkpoint file, result from training script.')
    parser.add_argument('--class_map',
                        help='json file that has the map of classes to flower names.')
    parser.add_argument('--topk', default=5, type=int,
                        help='top k classes.')
    parser.add_argument('--gpu', action='store_true',
                        help='user gpu to predict image')

    args = parser.parse_args()
    return args.image, args.checkpoint, args.class_map, args.topk, args.gpu


def load_datasets(data_dir):
    """
    Args:
        data_dir: contains train, validation and test folder.
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # train
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_datasets = datasets.ImageFolder(
        train_dir, transform=train_transforms)
    train_loader = D.DataLoader(dataset=train_datasets,
                                shuffle=True, batch_size=64)

    # validation
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_datasets = datasets.ImageFolder(
        valid_dir, transform=valid_transforms)
    valid_loader = D.DataLoader(
        dataset=valid_datasets, batch_size=64, shuffle=True)

    # test
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = D.DataLoader(
        dataset=test_datasets, batch_size=64, shuffle=True)

    return train_loader, valid_loader, test_loader, train_datasets.class_to_idx


def create_classifier(arch, hidden_layers, drop_p=0.2):
    input_size = get_arch_input_size(arch)
    output_size = 102
    hidden_layers = hidden_layers
    return Classifier(input_size, output_size, hidden_layers, drop_p)


def create_model(arch):
    if arch == 'vgg16_bn':
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
    elif arch == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    return model


def save_check_point(fp, arch, hidden_layers, drop_p, model, optimizer, class_to_idx):
    checkpoint = {
        'arch': arch,
        'input_size': get_arch_input_size(arch),
        'output_size': 102,
        'hidden_layers': hidden_layers,
        'drop_p': drop_p,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx
    }
    torch.save(checkpoint, fp)
    print("model was saved to {}".format(fp))


def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    arch = checkpoint['arch']

    model = create_model(arch)
    model.classifier = create_classifier(
        arch, checkpoint['hidden_layers'], checkpoint['drop_p'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    optimizer = optim.Adam(params=model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(
        224), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    img = Image.open(image)
    return transform(img)


def cat_to_name(fp):
    with open(fp, 'r') as f:
        return json.load(f)


def get_arch_input_size(arch):
    if arch == 'vgg16_bn':
        return 25088
    elif arch == 'densenet121':
        return 1024
