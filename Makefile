


train:
	python train.py --data_dir=flowers --checkpoint='out/vgg16_bn_checkpoint.pth' --arch=vgg16_bn --learning_rate=0.001 --hidden_units=1028 --gpu --epochs=7

predict:
	python predict.py --image='flowers/test/11/image_03130.jpg' --checkpoint='out/vgg16_bn_checkpoint.pth' --class_map=cat_to_name.json --topk=5 --gpu

predict-no-gpu:
	python predict.py --image='flowers/test/11/image_03130.jpg' --checkpoint='out/vgg16_bn_checkpoint.pth' --class_map=cat_to_name.json --topk=5