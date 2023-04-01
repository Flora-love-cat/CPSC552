"""
conditional GAN on MNIST dataset
"""
import torch
from torch import optim, nn
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
import math
import numpy as np



class Generator(nn.Module):
  def __init__(self, latent_dim=100, data_dim=784, num_classes=10): 

    super(Generator,self).__init__() 

    self.label_embedding = nn.Embedding(num_classes, 10)

    self.hidden_layer1 = nn.Sequential(
        nn.Linear(latent_dim + num_classes, 256),
        nn.LeakyReLU(0.2)
    )
    self.hidden_layer2 = nn.Sequential(
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2)
    )
    self.hidden_layer3 = nn.Sequential(
        nn.Linear(512, 1024),
        nn.LeakyReLU(0.2)
    )
    self.hidden_layer4 = nn.Sequential(
        nn.Linear(1024, data_dim),
        nn.Tanh()
    )
  def forward(self, x, labels):
      """
      x: random noise (batch_size, latent_dim)
      labels: random labels (batch_size, )

      return:
      fake data: (batch_size, data_dim)
      """
      
      # concat noise and random labels along last dimension
      x = torch.cat((x, self.label_embedding(labels).squeeze()), dim=-1)
      output = self.hidden_layer1(x)
      output = self.hidden_layer2(output)
      output = self.hidden_layer3(output)
      output = self.hidden_layer4(output)
      return output 


class Discriminator(nn.Module):
    def __init__(self, data_dim=784, num_classes=10):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, 10)

        self.hidden_layer1 = nn.Sequential(
            nn.Linear(data_dim+num_classes, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer4 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        """
        x: fake data (batch_size, data_dim) or real data (batch_size, 1, img_size, img_size)
        labels: real labels or random labels (batch_size, )
        """
        # concat data and real labels along last dimension
        x = torch.cat((x.view(x.size(0), -1), self.label_embedding(labels).squeeze()), -1)
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.hidden_layer4(output)
        return output 




def train_generator(device, g_optimizer, loss_fn, generator, discriminator, batch_size, **kwargs):
    """
    training generator for a batch by
        1. Generating fake images from random noise.
        2. let discriminator classify fake images as real (label 1)
        3. Computing loss on the result.
    
    Returns batch generator loss 
    """
    g_optimizer.zero_grad()
    # generate a batch of random noise
    noise = torch.randn(batch_size, kwargs['latent_dim']).to(device) 
    # generate a batch of random labels
    random_labels = torch.randint(low = 0, high = kwargs['num_classes'], size = (batch_size,)).to(device)
    
    # generate fake data by feed noise and random labels into generator
    fake_data = generator(noise, random_labels)

    # let discriminator classify fake data 
    pred_labels = discriminator(fake_data, random_labels)

    # maximize loss by let discriminator classify fake data as real (label 1)
    true_labels = torch.ones((batch_size, ), device=device).reshape(pred_labels.shape)
   
    # compute generator loss 
    loss = 0.5* loss_fn(pred_labels, true_labels)

    # Backpropagate the loss
    loss.backward()

    # perform a step of optimization
    g_optimizer.step()

    return loss.item()


def train_discriminator(device, d_optimizer, loss_fn, generator, discriminator, 
                        real_data, real_labels, batch_size=64, **kwargs): 
    """
    Train discriminator for a batch by
        1. let generator generate fake images from random noise.
        2. discriminator catches generator by classifying fake images as fake (all labels are 0) 
        3. discriminator classify real images as real (all labels are 1)
        3. Computing loss on the results.

    @params
    real_data: (batch_size, 1, 28, 28)
    
    Returns batch discriminator loss
    """
    d_optimizer.zero_grad()

    # generate a batch of random noise 
    noise = torch.randn(batch_size, kwargs['latent_dim']).to(device) 

    # generate a batch of random labels
    random_labels = torch.randint(low = 0, high = kwargs['num_classes'], size = (batch_size,)).to(device)

    fake_data = generator(noise, random_labels)

    # feed real data and real labels into discriminator to get output
    pred_label_real = discriminator(real_data, real_labels) 

    # feed fake data and real labels into discriminator to get output
    pred_label_fake = discriminator(fake_data, real_labels) 

    # generate true labels of real data (1)
    true_label_real = torch.ones((batch_size, ), device=device).reshape(pred_label_real.shape)

    # generate true labels of fake data (0)
    true_label_fake = torch.zeros((batch_size, ), device=device).reshape(pred_label_fake.shape)

    # compute discriminator loss
    loss = 0.5 * (loss_fn(pred_label_real, true_label_real) + loss_fn(pred_label_fake, true_label_fake))

    # Backpropogate the loss
    loss.backward()

    # perform a step of optimization
    d_optimizer.step()

    return loss.item()


def train_epoch(device, generator, discriminator, g_optimizer, 
                d_optimizer, loss_fn, train_loader, label_maps, **kwargs):
    """
    Train generator and discriminator together
    """
    loss = np.zeros((kwargs['n_epochs'], 2)) 

    for epoch in range(kwargs['n_epochs']):
        G_loss, D_loss = np.zeros([len(train_loader.dataset)]), np.zeros([len(train_loader.dataset)])

        for batch_id, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            batch_size = data.shape[0]  # if the batch size doesn't evenly divide the dataset length, this may change on the last epoch.
            G_loss[batch_id] = train_generator(device, g_optimizer, loss_fn, generator, discriminator, 
                                                batch_size=batch_size, **kwargs)
            D_loss[batch_id] = train_discriminator(device, d_optimizer, loss_fn, generator, discriminator, 
                                                   data, labels, batch_size=batch_size, **kwargs)
        # calculate average batch loss
        loss[epoch] = [torch.mean(D_loss), torch.mean(G_loss)] 

        print(f"Epoch {epoch}: loss_d: {loss[epoch][0]:.3f}, loss_g: {loss[epoch][1]:.3f}")

        plot_fakeimages(device, generator, epoch, label_maps, **kwargs)

    with open(f'./loss.npy', 'wb') as f:
        np.save(f, loss)
    

    

def plot_fakeimages(device, generator, epoch, label_maps, **kwargs):
    """Plot a batch of images generated by generator at a certain epoch"""  
    with torch.no_grad():
        noise = torch.randn(kwargs['batch_size'], kwargs['latent_dim']).to(device)
        random_labels = torch.randint(0, kwargs['num_classes'], (kwargs['batch_size'],)).to(device)
        generated_data = generator(noise, random_labels).cpu().view(kwargs['batch_size'], kwargs['img_size'], kwargs['img_size'])

        batch_sqrt = int(kwargs['batch_size'] ** 0.5)
        fig, ax = plt.subplots(batch_sqrt, batch_sqrt, figsize=(15, 15))
        for i, x in enumerate(generated_data):
            ax[math.floor(i / batch_sqrt)][i % batch_sqrt].set_title(label_maps[int(random_labels[i].item())])  # add labels
            ax[math.floor(i / batch_sqrt)][i % batch_sqrt].imshow(x.detach().numpy(), interpolation='nearest', cmap='gray')
            ax[math.floor(i / batch_sqrt)][i % batch_sqrt].get_xaxis().set_visible(False)
            ax[math.floor(i / batch_sqrt)][i % batch_sqrt].get_yaxis().set_visible(False)
        
        plt.suptitle(f"Epoch {epoch}")
        fig.savefig(f"./cGAN_Epoch{epoch}.png")
        # plt.show()

def plot_loss():
    """
    plot loss vs epoch
    """
    with open(f'./loss.npy', 'rb') as f:
        loss = np.load(f, allow_pickle=True)
    plt.figure(figsize=(8, 6))
    plt.plot(loss[:, 0], 'g', label="discriminator loss")
    plt.plot(loss[:, 1], 'b', label="generator loss")
    plt.title('cGAN', size=20, y=1.1)
    plt.xlabel("Number of iterations")
    plt.ylabel("Training Loss")
    plt.legend(bbox_to_anchor=(1.02, 1))

    plt.savefig(f'./trainingloss.png', bbox_inches='tight')
    # plt.show()


def main():


    #================================
    # Hyperparameters
    #=================================
    train_params = {
        "img_size": 28,
        "n_epochs": 24,
        "batch_size": 64,
        "lr_g": 0.0002,
        "lr_d": 0.0002
        }

    model_params = {
        'latent_dim': 100, 
        'data_dim': 784,
        'num_classes': 10
    }

    kwargs = {**train_params, **model_params}

    #=================================================
    # Initialize models, optimizers, loss function
    #=================================================
    # use an NVIDIA GPU, if one is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # initialize both generator and discriminator, attach to device
    generator = Generator(input_dim=kwargs['latent_dim'], output_dim=kwargs['data_dim'], num_classes=kwargs['num_classes']).to(device)
    discriminator = Discriminator(input_dim=kwargs['data_dim'], num_classes=kwargs['num_classes']).to(device) 

    g_optimizer = optim.Adam(generator.parameters(), lr=kwargs['lr_g'])
    d_optimizer = optim.Adam(discriminator.parameters(), lr=kwargs['lr_d'])

    loss_fn = nn.BCELoss(reduction='mean')


    #================================
    # loading the dataset
    #=================================
    # define a transform to 1) scale the images and 2) convert them into tensors
    transform = transforms.Compose([
        transforms.Resize(kwargs['img_size']), # scales the smaller edge of the image to have this size
        transforms.ToTensor(),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            './data', # specifies the directory to download the datafiles to, relative to the location of the notebook.
            train = True,
            download = True,
            transform = transform),
        batch_size = kwargs["batch_size"],
        shuffle=True
        )
    
    # Fashion MNIST has 10 classes, just like MNIST. 
    label_maps = {
        0: 'T-shirt/top',
        1	: 'Trouser',
        2	: 'Pullover',
        3	: 'Dress',
        4	: 'Coat',
        5	: 'Sandal',
        6	: 'Shirt',
        7	: 'Sneaker',
        8	: 'Bag',
        9	: 'Ankle boot'
    }

    #================================
    # Training
    #=================================
    train_epoch(device, generator, discriminator, g_optimizer, 
                d_optimizer, loss_fn, train_loader, label_maps, **kwargs)

    #================================
    # Evaluation
    #=================================
    plot_loss()


if __name__ == '__main__':
    main()
