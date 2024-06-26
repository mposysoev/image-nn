train:
	python3.12 compress_img_tf.py example/panelki.jpg --epochs 50 --batch_size 64

train_torch:
	python3.12 compress_img_torch.py example/sakura.jpg --epochs 99 --batch_size 64

restore_final:
	python3.12 decompress_img_tf.py example/panelki.jpg_model_final.keras 1280 1280

restore_final_torch:
	python3.12 decompress_img_torch.py example/oranges.jpg_model_final.pth 1280 1280

train_torch_continue:
	python3.12 compress_img_torch.py example/oranges.jpg --epochs 50 --batch_size 128 --model_path example/orange_modeles/oranges.jpg_model_final.pth
