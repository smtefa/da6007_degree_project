import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np

classes = ('airplane', 'cat', 'dog')

# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_set = torchvision.datasets.CIFAR10(root='../../data/',
                                         train=True,
                                         download=True,
                                         transform=transform_train)

test_set = torchvision.datasets.CIFAR10(root='../../data/',
                                        train=False, 
                                        transform=transform_test)

#------------------------------------------------------------------------------

# Add label noise to a data set
def add_label_noise(dataset, noise_level):
    noisy_dataset = dataset
    num_samples = len(dataset.targets)
    num_noise_samples = int(num_samples*noise_level)

    # Randomly select samples to corrupt labels
    random_indices = np.random.choice(num_samples, num_noise_samples,
                                      replace=False)

    # Add label noise to selected samples
    for idx in random_indices:
        true_label = noisy_dataset.targets[idx]
        noisy_labels = [label for label in range(10) if label != true_label]
        noisy_label = np.random.choice(noisy_labels)
        noisy_dataset.targets[idx] = noisy_label

    return noisy_dataset

#------------------------------------------------------------------------------

# Create data loaders
def make_data(batch_size, noise_level):
    # Add label noise to the training set
    noisy_train_set = add_label_noise(train_set, noise_level=noise_level)
    
    train_loader = torch.utils.data.DataLoader(noisy_train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False)
    
    return train_loader, test_loader

#------------------------------------------------------------------------------

# Filter out a smaller data set of airplanes, dogs and cats.
def filter_data(batch_size, scatter_size):
    class_indices = [test_set.class_to_idx[c] for c in classes]
    
    # Filter out images of cats, dogs, and airplanes
    filtered_data = []
    for image, label in test_set:
        if label in class_indices:
            filtered_data.append((image, label))    
    
    subset_indices = torch.randperm(len(filtered_data))[:scatter_size]
    subset_dataset = data.Subset(filtered_data, subset_indices)
    
    # Create a DataLoader for the subset dataset
    filtered_test_loader = torch.utils.data.DataLoader(subset_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False)
    
    return filtered_test_loader

    