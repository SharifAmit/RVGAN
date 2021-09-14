import matplotlib.pyplot as plt
import pandas as pd 
import os
from src.dataloader import *

def summarize_performance(step,g_global_model,g_local_model, dataset, n_samples=3,savedir='RVGAN'):
    # select a sample of input images
    n_patch = [1,1,1]
    [X_realA, X_realB, X_realC], _ = generate_real_data_random(dataset, n_samples, n_patch)
    #######################################
    # Resize to half
    out_shape = (int(X_realA.shape[1]/2),int(X_realA.shape[2]/2))
    [X_realA_half,X_realB_half,X_realC_half] = resize(X_realA,X_realB,X_realC,out_shape)
    # generate a batch of fake samples
    [X_fakeC_half, x_global], _ = generate_fake_data_coarse(g_global_model, X_realA_half, X_realB_half, n_patch)
    ##########################################
    # generate a batch of fake samples
    X_fakeC, _ = generate_fake_data_fine(g_local_model, X_realA, X_realB, x_global, n_patch)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realC = (X_realC + 1) / 2.0
    X_fakeC = (X_fakeC + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        twoD_img = X_fakeC[:,:,:,0]
        plt.imshow(twoD_img[i],cmap="gray")
    # plot real target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples*2 + i)
        plt.axis('off')
        twoD_img = X_realC[:,:,:,0]
        plt.imshow(twoD_img[i],cmap="gray")
    # save plot to file
    filename1 = savedir+'/local_plot_%06d.png' % (step+1)
    plt.savefig(filename1)
    plt.close()
    # save the generator model
    filename2 = savedir+'/local_model_%06d.h5' % (step+1)
    g_local_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

def summarize_performance_global(step, g_model, dataset, n_samples=3,savedir='RVGAN'):
    # select a sample of input images
    n_patch = [1,1,1]
    [X_realA, X_realB, X_realC], _ = generate_real_data_random(dataset, n_samples, n_patch)
    # Resize to half
    out_shape = (int(X_realA.shape[1]/2),int(X_realA.shape[2]/2))
    [X_realA_half,X_realB_half,X_realC_half] = resize(X_realA,X_realB, X_realC, out_shape)
    # generate a batch of fake samples
    [X_fakeC_half, x_global], _ = generate_fake_data_coarse(g_model, X_realA_half, X_realB_half, n_patch)
    # scale all pixels from [-1,1] to [0,1]
    X_realA_half = (X_realA_half + 1) / 2.0
    X_realC_half = (X_realC_half + 1) / 2.0
    X_fakeC_half = (X_fakeC_half + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_realA_half[i])
    # plot generated target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        twoD_img = X_fakeC_half[:,:,:,0]
        plt.imshow(twoD_img[i],cmap="gray")
    # plot real target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples*2 + i)
        plt.axis('off')
        twoD_img = X_realC_half[:,:,:,0]
        plt.imshow(twoD_img[i],cmap="gray")
    # save plot to file
    filename1 = savedir+'/global_plot_%06d.png' % (step+1)
    plt.savefig(filename1)
    plt.close()
    # save the generator model
    filename2 = savedir+'/global_model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

def plot_history(d1_hist, d2_hist, d3_hist, d4_hist, fm1_hist, fm2_hist, g_global_hist,g_local_hist, g_global_recon_hist, g_local_recon_hist, gan_hist,savedir='RVGAN'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.plot(d1_hist, label='dloss1')
    plt.plot(d2_hist, label='dloss2')
    plt.plot(d3_hist, label='dloss3')
    plt.plot(d4_hist, label='dloss4')
    plt.plot(fm1_hist, label='fm1')
    plt.plot(fm2_hist, label='fm2')
    plt.plot(g_global_hist, label='g_g_loss')
    plt.plot(g_local_hist, label='g_l_loss')
    plt.plot(g_global_recon_hist, label='g_g_rec')
    plt.plot(g_local_recon_hist, label='g_l_rec')
    plt.plot(gan_hist, label='gan_loss')
    plt.legend()
    filename = savedir+'/plot_line_plot_loss.png'
    plt.savefig(filename)
    plt.close()
    print('Saved %s' % (filename))

def to_csv(d1_hist, d2_hist, d3_hist, d4_hist,fm1_hist, fm2_hist, g_global_hist,g_local_hist,g_global_recon_hist, g_local_recon_hist, gan_hist ,savedir='RVGAN'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    d1 = np.array(d1_hist)
    d2 = np.array(d2_hist)
    d3 = np.array(d3_hist)
    d4 = np.array(d4_hist)
    fm1 = np.array(fm1_hist)
    fm2 = np.array(fm2_hist)
    g_global = np.array(g_global_hist)
    g_local = np.array(g_local_hist)
    g_g_rec = np.array(g_global_recon_hist)
    g_l_rec = np.array(g_local_recon_hist)
    gan = np.array(gan_hist)
    df = pd.DataFrame(data=(d1,d2,d3,d4,fm1,fm2,g_global,g_local,g_g_rec,g_l_rec,gan)).T
    df.columns=["d1","d2","d3","d4","fm1","fm2","g_global","g_local","g_g_rec","g_l_rec","gan"]
    filename = savedir+"/rv-loss.csv"
    df.to_csv(filename)
