"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

visualization_metrics.py

Note: Use PCA or tSNE for generated and original datasets visualization
"""
import os

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

   
def visualization (ori_data, generated_data, analysis, args, run=None):
  """Using PCA or tSNE for generated and original datasets visualization.
  
  Args:
    - ori_data: original datasets
    - generated_data: generated synthetic datasets
    - analysis: tsne or pca
  """  
  # Analysis sample size (for faster computation)
  anal_sample_no = min([1000, len(ori_data)])
  idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    
  # Data preprocessing
  ori_data = np.asarray(ori_data)
  generated_data = np.asarray(generated_data)  
  
  ori_data = ori_data[idx]
  generated_data = generated_data[idx]
  
  no, seq_len, dim = ori_data.shape  
  
  for i in range(anal_sample_no):
    if (i == 0):
      prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
      prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
    else:
      prep_data = np.concatenate((prep_data, 
                                  np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
      prep_data_hat = np.concatenate((prep_data_hat, 
                                      np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
  # Visualization parameter        
  colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    
    
  if analysis == 'pca':
    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(prep_data)
    pca_results = pca.transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)
    
    # Plotting
    f, ax = plt.subplots(1)    
    plt.scatter(pca_results[:,0], pca_results[:,1],
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
  
    ax.legend()  
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    # plt.savefig(f'./figures/{args.rnn_type}/sine_pca_{args.z_dim}_h_dim_{args.hidden_dim}_w_kl_{args.w_kl}.pdf', transparent=True,
    #             bbox_inches='tight', pad_inches=0, dpi=300)
    if run:
      run["test/pca_ori_gen"].log(f)
    plt.show()

    
  elif analysis == 'tsne':
    
    # Do t-SNE Analysis together       
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)
      
    # Plotting
    f, ax = plt.subplots(1)
      
    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
  
    ax.legend()
      
    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    # plt.savefig(f'./figures/{args.rnn_type}/sine_tsne_{args.z_dim}_h_dim_{args.hidden_dim}_w_kl_{args.w_kl}.pdf', transparent=True,
    #             bbox_inches='tight', pad_inches=0, dpi=300)
    if run:
      run["test/tsne_ori_gen"].log(f)
    plt.show()

  elif analysis == "histogram":
    f, ax = plt.subplots(1)
    sns.distplot(prep_data, hist=False, kde=True, label='Original')
    sns.distplot(prep_data_hat, hist=False, kde=True, kde_kws={'linestyle': '--'}, label='VKAE (Ours)')
    # Plot formatting
    plt.legend()
    plt.xlabel('Data Value')
    plt.ylabel('Data Density Estimate')
    plt.rcParams['pdf.fonttype'] = 42
    plt.title('{} with {:.1f}% missing values'.format(args.dataset, args.missing_value))
    # path = "./figures/" + args.dataset + "/{:.1f}_histo.pdf".format(args.missing_value)
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    # plt.savefig(path, 0)
    if run:
      run["test/histo_ori_gen"].log(f)
    plt.show()
    plt.close()

    from scipy.spatial import distance

    p_ = sns.histplot(prep_data.flatten(), label='Original', stat='probability', bins=200).patches
    plt.close()
    q_ = sns.histplot(prep_data_hat.flatten(), label='VKAE (Ours)', stat='probability', bins=200).patches
    plt.close()
    p = np.array([h.get_height() for h in p_])
    q = np.array([h.get_height() for h in q_])
    if p.shape[0] < q.shape[0]:
        q = q[:p.shape[0]]
    else:
        p = p[:q.shape[0]]
    print(np.sum(p))
    print(np.sum(q))
    print(distance.jensenshannon(p, q))