import matplotlib.pyplot as plt

def plot_errors_and_losses(k, epochs, params):
    [train_losses, test_losses, train_errors, test_errors] = params
    x = list(range(1, epochs+1))
    
    # Cross-entropy curves
    plt.plot(x, train_losses, label='Train loss')
    plt.plot(x, test_losses, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses of ResNet-18 with width={}'.format(k))
    plt.xscale('log')
    plt.legend()
    plt.savefig(f'losses_{k}.png')
    plt.show()
    
    # Accuracy curves
    plt.plot(x, train_errors, label='Train error')
    plt.plot(x, test_errors, label='Test error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Errors of ResNet-18 with width={}'.format(k))
    plt.xscale('log')
    plt.legend()
    plt.savefig(f'errors_{k}.png')
    plt.show()

#------------------------------------------------------------------------------

def scatter_scale(c, ax):
    ax.set_xlim(-c, c)
    ax.set_ylim(-c, c)
    ax.set_zlim(-c, c)          
    
    # Add lines representing the axes passing through the origin
    ax.plot([-c, c], [0, 0], [0, 0], color='k', linestyle='-', linewidth=1)
    ax.plot([0, 0], [-c, c], [0, 0], color='k', linestyle='-', linewidth=1)
    ax.plot([0, 0], [0, 0], [-c, c], color='k', linestyle='-', linewidth=1)

#------------------------------------------------------------------------------

def scatter_plot(epoch, noise, colors, labels, points, softmax=False):   
    z = [point[0] for point in points]
    y = [point[1] for point in points]
    x = [point[2] for point in points]                
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Set scales
    if softmax == False:
        if  epoch+1 <= 50:
            scatter_scale(20, ax)       
    
        elif  51 <= epoch+1 <= 100:
            scatter_scale(40, ax)
    
        elif  101 <= epoch+1 <= 500:
            scatter_scale(60, ax)                               
            
        else:
            scatter_scale(80, ax)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)                 
    
    # Plot points with different colors based on classes
    for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], c=colors[labels[i]], s=1)
    
    # Set labels and title
    ax.set_xlabel('cat')
    ax.set_ylabel('dog')
    ax.set_zlabel('airplane')
    ax.set_title('noise={:.0%}, epoch={}'.format(noise, epoch+1))
    
    plt.gca().invert_yaxis()

    if softmax == False:
        plt.savefig('raw_scat_epoch_{:.0}_{}.png'.format(noise, epoch+1))  
    else:
        plt.savefig(f'soft_scat_epoch_{epoch+1}.png')  
    plt.show()

        