from utilities import *

def get_base_info(logfile, adv_file):
	file = open(logfile)
	lines = file.readlines()
	for line in lines:
		words = line.split(',')
		if words[1].split('.')[0] == adv_file.split('.')[0]:
			return words
	return None

def stitch(adv_folder, logfile):
	if "stitched_images" not in os.listdir():
		os.mkdir("stitched_images")
	adv_files = os.listdir(adv_folder)
	for adv_file in adv_files:
		print("stitching " + adv_file)
		adv_tensor = read_tiff_image(adv_folder+"/"+adv_file)
		# show_tiff_image(adv_folder+"/"+adv_file)
		info = get_base_info(logfile, adv_file)
		if info == None:
			continue
		base_img = Image.open("obj-dec/PyTorch-YOLOv3/"+info[0])
		# print(target_shape)
		trans = torchvision.transforms.ToPILImage()
		adv_img = trans(adv_tensor)
		adv_img = adv_img.resize((int(info[4])-int(info[2]),int(info[5])-int(info[3])))
		base_img.paste(adv_img, (int(info[2]), int(info[3])))
		base_img.save("stitched_images/" + adv_file.split('.')[0] + '.jpg')

