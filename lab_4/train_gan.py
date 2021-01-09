import pandas as pd

import torch
from torch import nn, optim
from IPython import display

from lab_4.model import *


discriminator = DiscriminatorNet()
generator = GeneratorNet()


if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()


# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Loss function
loss = nn.BCELoss()

# Number of steps to apply to the discriminator
d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1
# Number of epochs
num_epochs = 200

num_test_samples = 16
test_noise = noise(num_test_samples)

logger = Logger(model_name='GAN', data_name='audio')

df = pd.read_csv(r'C:\Users\denis\PycharmProjects\TTs_for_STC\data\features_lab4.csv')

data = My_Dataset(pd.get_dummies(df.drop(columns=['word', 'phoneme', 'allophone'])).values.astype(np.float32))

# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

for epoch in range(num_epochs):
    for n_batch, real_batch in enumerate(data_loader):

        # 1. Train Discriminator
        real_data = Variable(real_batch)
        if torch.cuda.is_available(): real_data = real_data.cuda()
        # Generate fake data
        fake_data = generator(noise(real_data.size(0))).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
                                                                real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(real_batch.size(0)))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        # Log error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # Display Progress
        if (n_batch) % 100 == 0:
        # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
        # Model Checkpoints
        logger.save_models(generator, discriminator, epoch)











