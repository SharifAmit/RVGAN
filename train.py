from src.model import coarse_generator,fine_generator,RVgan,discriminator_ae
from src.visualization import summarize_performance, summarize_performance_global, plot_history, to_csv
from src.dataloader import resize, generate_fake_data_coarse, generate_fake_data_fine, generate_real_data, generate_real_data_random, load_real_data
import argparse
import time
import os
from numpy import load
import gc
import keras.backend as K



def train(d_model1, d_model2, g_global_model, g_local_model, 
          gan_model, dataset, n_epochs=20, n_batch=1, n_patch=[64,32],savedir='RVGAN'):
    
    if not os.path.exists(savedir):
      os.makedirs(savedir)
    # unpack dataset
    trainA, _, _ = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    
    # lists for storing loss, for plotting later
    d1_hist, d2_hist, d3_hist, d4_hist =  list(),list(), list(), list()
    fm1_hist,fm2_hist = list(),list()
    g_global_hist, g_local_hist, gan_hist =  list(), list(), list()
    g_global_recon_hist, g_local_recon_hist =list(),list()
    # manually enumerate epochs
    b = 0
    start_time = time.time()
    for k in range(n_epochs):
        for i in range(bat_per_epo):
          d_model1.trainable = True
          d_model2.trainable = True
          gan_model.trainable = False
          g_global_model.trainable = False
          g_local_model.trainable = False
          for j in range(2):
              # select a batch of real samples 
              [X_realA, X_realB, X_realC], [y1,y2] = generate_real_data(dataset, i, n_batch, n_patch)

              
              # generate a batch of fake samples for Coarse Generator
              out_shape = (int(X_realA.shape[1]/2),int(X_realA.shape[2]/2))
              [X_realA_half,X_realB_half, X_realC_half] = resize(X_realA,X_realB,X_realC,out_shape)
              [X_fakeC_half, x_global], y1_coarse = generate_fake_data_coarse(g_global_model, X_realA_half, X_realB_half, n_patch)


              # generate a batch of fake samples for Fine Generator
              X_fakeC, y1_fine= generate_fake_data_fine(g_local_model, X_realA, X_realB, x_global, n_patch)


              ## FINE DISCRIMINATOR  
              # update discriminator for real samples
              d_loss1 = d_model1.train_on_batch([X_realA, X_realC], y1)[0]
              # update discriminator for generated samples
              d_loss2 = d_model1.train_on_batch([X_realA, X_fakeC], y1_fine)[0]

              #d_loss1 = 0.5*(d_loss1_real[0]+d_loss1_fake[0])

              
              #d_loss2 = 0.5*(d_loss2_real[0]+d_loss2_fake[0])

              ## COARSE DISCRIMINATOR  
              # update discriminator for real samples
              d_loss3 = d_model2.train_on_batch([X_realA_half, X_realC_half], y2)[0]
              # update discriminator for generated samples
              d_loss4 = d_model2.train_on_batch([X_realA_half, X_fakeC_half], y1_coarse)[0]
          
          #if n_steps%425 ==0:

          # turn Global G1 trainable
          d_model1.trainable = False
          d_model2.trainable = False
          gan_model.trainable = False
          g_global_model.trainable = True
          g_local_model.trainable = False
          
          

          # select a batch of real samples for Local enhancer
          [X_realA, X_realB, X_realC], _ = generate_real_data(dataset, i,n_batch, n_patch)

          # Global Generator image fake and real
          out_shape = (int(X_realA.shape[1]/2),int(X_realA.shape[2]/2))
          [X_realA_half,X_realB_half, X_realC_half] = resize(X_realA,X_realB,X_realC,out_shape)
          [X_fakeC_half, x_global], _ = generate_fake_data_coarse(g_global_model, X_realA_half, X_realB_half, n_patch)
          

          # update the global generator
          g_global_loss,_ = g_global_model.train_on_batch([X_realA_half,X_realB_half], [X_realC_half])

          
          d_model1.trainable = False
          d_model2.trainable = False
          gan_model.trainable = False
          g_global_model.trainable = False
          g_local_model.trainable = True
          
          # update the Local Enhancer 
          g_local_loss = g_local_model.train_on_batch([X_realA,X_realB,x_global], X_realC)
          

          # turn G1, G2 and GAN trainable, not D1,D2 and D3
          d_model1.trainable = False
          d_model2.trainable = False
          gan_model.trainable = True
          g_global_model.trainable = True
          g_local_model.trainable = True
          # update the generator
          gan_loss,_,_,fm1_loss,fm2_loss,_,_,g_global_recon_loss, g_local_recon_loss = gan_model.train_on_batch([X_realA,X_realA_half,x_global,X_realB,X_realB_half,X_realC,X_realC_half], 
                                                                                                                                                      [y1, y2,
                                                                                                                                                        X_fakeC,X_fakeC_half,
                                                                                                                                                        X_fakeC_half,X_fakeC,
                                                                                                                                                        X_fakeC_half,X_fakeC])

          # summarize performance
          print('>%d, d1[%.3f] d2[%.3f] d3[%.3f] d4[%.3f] fm1[%.3f] fm2[%.3f] g_g[%.3f] g_l[%.3f] g_g_r[%.3f] g_l_r[%.3f] gan[%.3f]' % 
                (i+1, d_loss1, d_loss2, d_loss3, d_loss4, 
                  fm1_loss, fm2_loss, 
                  g_global_loss, g_local_loss, 
                  g_global_recon_loss, g_local_recon_loss, gan_loss))
                                                                                                                              
          d1_hist.append(d_loss1)
          d2_hist.append(d_loss2)
          d3_hist.append(d_loss3)
          d4_hist.append(d_loss4)
          fm1_hist.append(fm1_loss)
          fm2_hist.append(fm2_loss)
          g_global_hist.append(g_global_loss)
          g_local_hist.append(g_local_loss)
          g_global_recon_hist.append(g_global_recon_loss)
          g_local_recon_hist.append(g_local_recon_loss)
          gan_hist.append(gan_loss)
          # summarize model performance
      #if (i+1) % (bat_per_epo * 1) == 0:
        summarize_performance_global(b, g_global_model, dataset, n_samples=3,savedir=savedir)
          
        summarize_performance(b, g_global_model,g_local_model, dataset, n_samples=3,savedir=savedir)
        b = b + 1
        per_epoch_time = time.time()
        total_per_epoch_time = (per_epoch_time - start_time)/3600.0
        print(total_per_epoch_time)
    plot_history(d1_hist, d2_hist, d3_hist, d4_hist, fm1_hist, fm2_hist, g_global_hist,g_local_hist, g_global_recon_hist, g_local_recon_hist, gan_hist,savedir=savedir)
    to_csv(d1_hist, d2_hist, d3_hist, d4_hist, fm1_hist, fm2_hist, g_global_hist,g_local_hist, g_global_recon_hist, g_local_recon_hist, gan_hist,savedir=savedir)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--npz_file', type=str, default='DRIVE.npz', help='path/to/npz/file')
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--savedir', type=str, required=False, help='path/to/save_directory',default='RVGAN')
    parser.add_argument('--resume_training', type=str, required=False,  default='no', choices=['yes','no'])
    parser.add_argument('--weight_name_global',type=str, help='path/to/global/weight/.h5 file', required=False)
    parser.add_argument('--weight_name_local',type=str, help='path/to/local/weight/.h5 file', required=False)
    parser.add_argument('--inner_weight', type=float, default=0.5)
    args = parser.parse_args()
    
    K.clear_session()
    gc.collect()
    start_time = time.time()
    dataset = load_real_data(args.npz_file)
    print('Loaded', dataset[0].shape, dataset[1].shape)
    
    # define input shape based on the loaded dataset
    in_size = args.input_dim
    image_shape_coarse = (in_size//2,in_size//2,3)
    mask_shape_coarse = (in_size//2,in_size//2,1)
    label_shape_coarse = (in_size//2,in_size//2,1)


    image_shape_fine = (in_size,in_size,3)
    mask_shape_fine = (in_size,in_size,1)
    label_shape_fine = (in_size,in_size,1)
    
    image_shape_xglobal = (in_size//2,in_size//2,128)
    ndf=64
    ncf=128
    nff=128
    
    d_model1 = discriminator_ae(image_shape_fine,label_shape_fine,ndf,name="D1") 
    d_model2 = discriminator_ae(image_shape_coarse,label_shape_coarse,ndf,name="D2")
    
    
    g_model_fine = fine_generator(x_coarse_shape=image_shape_xglobal,input_shape=image_shape_fine,mask_shape=mask_shape_fine,nff=nff,n_blocks=3)
    g_model_coarse = coarse_generator(img_shape=image_shape_coarse,mask_shape=mask_shape_coarse,n_downsampling=2, n_blocks=9, ncf=ncf,n_channels=1)
    
    
    if args.resume_training =='yes':
      #weight_name_global = "global_model_000070.h5"
      g_model_coarse.load_weights(args.weight_name_global)

      #weight_name_local = "local_model_000070.h5"
      g_model_fine.load_weights(args.weight_name_local)
      
    rvgan_model = RVgan(g_model_fine,g_model_coarse, d_model1, d_model2,
                  image_shape_fine,image_shape_coarse,image_shape_xglobal,mask_shape_fine,mask_shape_coarse,label_shape_fine,label_shape_coarse,args.inner_weight)
    
    
    train(d_model1, d_model2,g_model_coarse, g_model_fine, rvgan_model, dataset, n_epochs=args.epochs, n_batch=args.batch_size, n_patch=[128,64],savedir=args.savedir)

    end_time = time.time()
    time_taken = (end_time-start_time)/3600.0
    print(time_taken)
