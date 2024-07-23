import torch
from torch import nn, optim
from util import create_classifier, create_model, load_datasets, train_args, save_check_point


def train(data_dir, checkpoint, arch, hidden_units, gpu, epochs, learn_rate=0.01):
    if gpu:
        if not torch.cuda.is_available():
            print('System does not support gpu, cuda is not available.')
            return
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_loader, valid_loader, test_loader, class_to_idx = load_datasets(
        data_dir)

    model = create_model(arch)
    model.classifier = create_classifier(arch, [hidden_units])
    optimizer = optim.Adam(params=model.classifier.parameters(), lr=learn_rate)
    criterion = nn.NLLLoss()
    model.to(device=device)

    steps = 0
    running_loss = 0
    print_every = 5
    for e in range(epochs):
        model.train()
        for images, labels in train_loader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            log_ps = model.forward(images)
            loss = criterion.forward(input=log_ps, target=labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in valid_loader:
                        images, labels = images.to(device), labels.to(device)

                        log_ps = model.forward(images)
                        loss = criterion.forward(input=log_ps, target=labels)
                        test_loss += loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(valid_loader):.3f}.. "
                      f"Test accuracy: {accuracy*100/len(valid_loader):.2f}")
                model.train()
                running_loss = 0

    save_check_point(checkpoint, arch, [hidden_units],
                     0.2, model, optimizer, class_to_idx)
    return model


def main():
    data_dir, checkpoint, arch, learning_rate, hidden_units, gpu, epochs = train_args()
    print("Start training the network with parameters:")
    print(f"Data directory:\t {data_dir}")
    print(f"Checkpoint:\t {checkpoint}")
    print(f"Architecture:\t {arch}")
    print(f"Learning rate:\t {learning_rate}")
    print(f"Hidden units:\t {hidden_units}")
    print(f"Enable GPU:\t {gpu}")
    print(f"Epochs:\t\t {epochs}")

    model = train(data_dir, checkpoint, arch, hidden_units,
                  gpu, epochs, learning_rate)

    print("Finish training model with classifier:")
    print(model.classifier)


if __name__ == "__main__":
    main()
