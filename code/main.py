from resnet18k import compare_noises

def main():
    '''
    The main code used to reproduce the figures. The default parameters are
    set to those which recreate the figures.
    
    Warning: The number of epochs should be set with respect to the available
    computational powers. If the GPU is not available, a very small number of
    epochs must be chosen.

    Returns
    -------
    None.

    '''
    # Hyper-parameters
    epochs = 100
    batch_size = 100 # batch size used by PyTorch's data loader
    minibatch_size = 100 # running losses' reset mark
    lr = 1e-4 # learning rate used by Adam

    scatter_size = 10000 # number of points in the scatter plot
    early_epochs = [1, 5, 10] # epochs for P1 of the double descent
    mid_epochs = [20, 40, 60, 80, 100] # epochs for P2 of the double descent
    late_epochs = [250, 500, 750, 1000] # epochs for P3 of the double descent
    snapshots = early_epochs + mid_epochs + late_epochs # P1 + P2 + P3
    
    plot_raw = False # change to True to disable softmax activation
    params = [epochs, batch_size, minibatch_size, lr]    
    scat_params = [scatter_size, snapshots, plot_raw]
    
    compare_noises(params, scat_params)
    
if __name__ == '__main__':
    main()
    
    
