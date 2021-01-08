from torchvision import transforms, datasets

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])
        ])
    out_dir = '{}/dataset'.format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

DATA_FOLDER = r'C:\Users\denis\Documents\mnist'

# Load data
data = mnist_data()

