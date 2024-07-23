import torch
from util import predict_args, load_checkpoint, process_image, cat_to_name


def predict(image_path, checkpoint, class_map, topk, gpu):
    if gpu:
        if not torch.cuda.is_available():
            print('System does not support gpu, cuda is not available.')
            return
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model, optimizer = load_checkpoint(checkpoint, device)
    idx_mapping = dict(map(reversed, model.class_to_idx.items()))

    with torch.no_grad():
        model.to(device)
        model.eval()
        image = torch.unsqueeze(process_image(image_path), 0)
        image = image.to(device)
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(topk, dim=1)

        probs = top_p.view(-1).tolist()
        classes = top_class.view(-1).tolist()
        names = [cat_to_name(class_map)[idx_mapping[each]]
                 for each in classes]

        print(probs)
        print(classes)
        print(names)

        print(f'The most likely image class is {
              names[0]} with probability {probs[0]*100:.2f}')
        print(f'Top {topk} classes:')
        for i in range(len(names)):
            print(f'{i+1}. {names[i]} with probability {probs[i]*100:.2f}')


def main():
    image, checkpoint, class_map, topk, gpu = predict_args()
    print("Start predicting the image with parameters:")
    print(f"Image:\t\t {image}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Class map:\t {class_map}")
    print(f"Top K:\t\t {topk}")
    print(f"Enable GPU:\t {gpu}")

    predict(image, checkpoint, class_map, topk, gpu)


if __name__ == "__main__":
    main()
