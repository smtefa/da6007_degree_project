import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data import make_data, filter_data
from helper import plot_errors_and_losses, scatter_plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using ' + str(device)) # print if cpu or gpu is being used


#------------------------------------------------------------------------------
# A ResNet-18k model based on the pre-activation blocks. Source:
# https://gitlab.com/harvard-machine-learning/double-descent/tree/master
class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_channels=64):
        super(PreActResNet, self).__init__()
        self.in_planes = init_channels
        c = init_channels

        self.conv1 = nn.Conv2d(3, c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2)
        self.linear = nn.Linear(8*c*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def make_resnet18k(k=64, num_classes=10) -> PreActResNet:
    ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, init_channels=k)

#------------------------------------------------------------------------------

def train_resnet18k(k, params, noise=0.2, scat_params=[]):
    '''
    The main training and testing function for the ResNet-18 model. If
    scat_params is provided, this function will also plot the output geometry.

    Parameters
    ----------
    k : int
        ResNet-18 width parameter.
    params : list
        Model parameters [epochs, batch_size, minibatch_size, lr].
    noise : float, optional
        Noise level. The default is 0.2.
    scat_params : list, optional
        Plot parameters [scatter_size, snapshots, plot_raw]. The default is [].

    Returns
    -------
    train_losses : list
        Train losses at each epoch.
    test_losses : list
        Test losses at each epoch.
    train_errors : list
        Train errors at each epoch.
    test_errors : list
        Test errors at each epoch.

    '''
    [epochs, batch_size, minibatch_size, lr] = params
    train_loader, test_loader = make_data(batch_size, noise)
    
    if scat_params:
        [scatter_size, snapshots, plot_raw] = scat_params
        filt_test_loader = filter_data(batch_size, scatter_size)

    print("PARAMS:: EPOCHS: {}, MINIBATCH: {}, LR: {}, NOISE: {}".format(
        epochs, batch_size, lr, noise))
    
    print("Training ResNet-18 with width={}".format(k))
    model = make_resnet18k(k=k, num_classes=10).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    train_errors = []
    test_errors = []
    
    total_train_step = len(train_loader)
    total_test_step = len(test_loader)
    for epoch in range(epochs):
        # Train the model
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
    
            # Forward pass
            outputs = model(images)
            loss_f = criterion(outputs, labels)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss_f.backward()
            optimizer.step()
            
            running_train_loss += loss_f.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            if i+1 == total_train_step:
                train_losses.append(running_train_loss / minibatch_size)
                train_errors.append(1-train_correct/train_total)
                      
            # Uncomment the print statement to monitor the training process
            if i % minibatch_size == minibatch_size - 1:
                '''
                print("Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}".format(
                    epoch+1, max_epochs,
                    i+1, len(train_loader), running_train_loss / minibatch_size))
                '''
                running_train_loss = 0.0
        
        # Test the model
        model.eval()
        running_test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
        
                # Forward pass
                outputs = model(images)
                loss_f = criterion(outputs, labels)
                
                running_test_loss += loss_f.item()    
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                if i+1 == total_test_step:
                    test_losses.append(running_test_loss / minibatch_size)
                    test_errors.append(1-test_correct/test_total)
                    print('Epoch [{}/{}], Test Error: {:.4f}'.format(
                        epoch+1, epochs, 1-test_correct/test_total))
                
                # Uncomment the print statement to monitor the testing process
                if i % minibatch_size == minibatch_size - 1:
                    '''
                    print("Epoch [{}/{}], Step [{}/{}], Test Loss: {:.4f}".format(
                        epoch+1, max_epochs,
                        i+1, len(test_loader), running_test_loss / minibatch_size))
                    '''
                    running_test_loss = 0.0
              
        # Output layer geometry
        if scat_params:
            with torch.no_grad():
                if epoch+1 in snapshots:
                    raw_points = []
                    soft_points = []
                    class_labels = []
                    class_colors = {
                        0: 'r',     # red for airplane class
                        3: 'g',     # green for dog class
                        5: 'b'      # blue for cat class
                    }
                    
                    for images, labels in filt_test_loader:
                        images = images.to(device)
                        outputs = model(images)
                        soft_outputs = F.softmax(outputs, dim=1)
                        trunc_outputs = outputs[:, [0, 3, 5]]
                        trunc_soft_outputs = soft_outputs[:, [0, 3, 5]]
                        raw_points += trunc_outputs
                        soft_points += trunc_soft_outputs
                        class_labels += labels.tolist()
                    
                    raw_points = [tensor.cpu().tolist()
                                  for tensor in raw_points]
                    soft_points = [tensor.cpu().tolist()
                                    for tensor in soft_points]
                    
                    if plot_raw:
                        scatter_plot(epoch, noise, class_colors, class_labels,
                                     raw_points, softmax=False)
                    scatter_plot(epoch, noise, class_colors, class_labels,
                                 soft_points, softmax=True)
          
    errors_and_losses_params = [train_losses, test_losses,
                                train_errors, test_errors]
    plot_errors_and_losses(k, epochs, errors_and_losses_params)
    
    return train_losses, test_losses, train_errors, test_errors

#------------------------------------------------------------------------------
    
def compare_noises(params, scat_params):
    '''
    Function that recreates Figure 6.

    Parameters
    ----------
    params : list
        Model parameters [epochs, batch_size, minibatch_size, lr].
    scat_params : list
        Plot parameters [scatter_size, snapshots, plot_raw]. The default is [].

    Returns
    -------
    small_noise : list
        Train/test losses and errors for noise=0% at each epoch.
    inter_noise : list
        Train/test losses and errors for noise=10% at each epoch.
    large_noise : TYPE
        Train/test losses and errors for noise=20% at each epoch.

    '''
    x = list(range(1, params[0]+1))
    small_noise = train_resnet18k(64, params, 0.0, scat_params)
    inter_noise = train_resnet18k(64, params, 0.1, scat_params)
    large_noise = train_resnet18k(64, params, 0.2, scat_params)
        
    # Accuracy curves
    plt.plot(x, small_noise[3], label='noise = 0%')
    plt.plot(x, inter_noise[3], label='noise = 10%')
    plt.plot(x, large_noise[3], label='noise = 20%')
    plt.xlabel('Epoch')
    plt.ylabel('Test Error')
    plt.title('Test errors of ResNet-18 with various noise levels')
    plt.xscale('log')
    plt.legend()
    plt.savefig('test_errors.png')
    plt.show()
    
    plt.plot(x, small_noise[2], label='noise = 0%')
    plt.plot(x, inter_noise[2], label='noise = 10%')
    plt.plot(x, large_noise[2], label='noise = 20%')
    plt.xlabel('Epoch')
    plt.ylabel('Train Error')
    plt.title('Train errors of ResNet-18 with various noise levels')
    plt.xscale('log')
    plt.legend()
    plt.savefig('train_errors.png')
    plt.show()
    
    # Cross-entropy curves
    plt.plot(x, small_noise[1], label='noise = 0%')
    plt.plot(x, inter_noise[1], label='noise = 10%')
    plt.plot(x, large_noise[1], label='noise = 20%')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test losses of ResNet-18 with various noise levels')
    plt.xscale('log')
    plt.legend()
    plt.savefig('test_losses.png')
    plt.show()
    
    plt.plot(x, small_noise[0], label='noise = 0%')
    plt.plot(x, inter_noise[0], label='noise = 10%')
    plt.plot(x, large_noise[0], label='noise = 20%')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Train losses of ResNet-18 with various noise levels')
    plt.xscale('log')
    plt.legend()
    plt.savefig('train_losses.png')
    plt.show()
    
    return small_noise, inter_noise, large_noise

