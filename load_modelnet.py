import torch
import csv
from tqdm import tqdm

class FileDataset(torch.utils.data.Dataset):
	def __init__(self, folder, modelnet, split, filenames, name2id, max_count=-1):
		self.folder = folder
		self.modelnet = modelnet
		self.split = split
		self.filenames = filenames
		self.name2id = name2id

		if max_count != -1:
			self.filenames = self.filenames[:max_count]

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, i):
		folder = self.folder
		modelnet = self.modelnet
		split = self.split
		filenames = self.filenames
		name2id = self.name2id

		filename = filenames[i]
		label = '_'.join(filename.split('_')[:-1])

		with open(f"{folder}/{label}/{filename}.txt") as file:
			cloud = file.read().split()
			cloud = list(map(lambda x : list(map(float, x.split(','))), cloud))

		label = name2id[label]

		return torch.tensor(cloud), torch.tensor(label)




def load_modelnet(folder='./datasets/ModelNet', modelnet='modelnet40', split='train', max_count=-1, num_workers=16):
	name2id = dict()
	
	with open(f"{folder}/{modelnet}_shape_names.txt") as file:
		for i, name in enumerate(file.read().split()):
			name2id[name] = i

	
	with open(f"{folder}/{modelnet}_{split}.txt") as file:
		filenames = file.read().split()

	

	dataset = FileDataset(folder, modelnet, split, filenames, name2id, max_count=max_count)
	loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, 
		num_workers=num_workers, collate_fn=lambda x : x[0], drop_last=False, prefetch_factor=8)

	clouds = []
	labels = []
	for cloud, label in tqdm(loader):
		clouds.append(cloud)
		labels.append(label)



	return torch.stack(clouds, dim=0), torch.stack(labels, dim=0)

# a, b = load_modelnet()
# print(a.shape, b.shape)