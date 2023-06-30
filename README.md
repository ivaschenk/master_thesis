# Thesis Master Informtion Studies - Data Science  
## Revealing Hierarchical Structuring in Humans Mental Object Representations using Hyperbolic Space  
The study of mental representations of objects underlying similarity judgments has been a central topic of investigation in cognitive neuroscience. Previous research on similarity judgement using large-scale behavioural data primarily considered low-level dimensions and semantic relationships between objects but not whether or how these objects are hierarchically ordered. When researching hierarchies, learning the embeddings in hyperbolic rather than Euclidean space would be more suitable. Current state-of-the-art research on human object representations employs a computational model that learns object embeddings in Euclidean space, achieving an accuracy score of 64.6\% with a noise ceiling of 67.2\%. This study aims to learn object embeddings in hyperbolic space and compare them to Euclidean space across various model dimensionality levels. The findings demonstrate that hyperbolic space is at least as good as Euclidean space in learning object embeddings with a model dimensionality higher than 2, showing an accuracy increase ranging between 1.5 and 2.5 percentage points. However, for the 2-dimensional models, Euclidean space outperforms hyperbolic space with a higher accuracy of 1.7 percentage points. These results highlight the significant impact of hierarchical relationships on human similarity judgment and emphasize the suitability of hyperbolic space for learning object similarity embeddings.

**Folders**  
*SPoSE-orginial*: This is de Euclidean approach for modelling object similarity based on the odd-one-out task. The following reposetory is used, with a few adjustments: https://github.com/ViCCo-Group/SPoSE.
*SPoSE*: This is the hyperbolic approach of the thesis.
*Analysis_results*: Here are the notebooks that are used for the analysis of the thesis.

**Environmental set-up**  
1. The code uses Python 3.8, Pytorch 1.6.0 (Note that PyTorch 1.6.0 requires CUDA 10.2, if you want to run on a GPU)
2. Install PyTorch: pip install pytorch or conda install pytorch torchvision -c pytorch (the latter is recommended if you use Anaconda)
3. Install Python dependencies: pip install -r requirements.txt

**Train SPoSE models**

*Hyperparameters*  
* task = odd_one_out (odd-one-out task (i.e., 3AFC) is used for this research, don't chose other option)  
* modality = behavioral (behavioral used for this research, don’t chose other option)  
* triplets_dir = master_thesis/SPoSE/data/ (location of triplet data)  
* results_dir (optional specification of results directory (if not provided will resort to ./results/modality/version/dim/lambda/seed/))  
* plots_dir (optional specification of directory for plots (if not provided will resort to ./plots/modality/version/dim/lambda/seed/)  
* learning_rate (learning rate to be used in optimizer)  
* lmbda (1) EUCLIDEAN (SPoSE-original): lambda value determines l1-norm fraction to regularize loss; will be divided by number of items in the original data matrix. 2) HYPERBOLIC(SPoSE): lambda value determines the strengt of the uniformity loss.)  
* embed_dim (embedding dimensionality, i.e., output size of the neural network. In this research the dimensionalities 2, 5, 15, 60 are used)  
* batch_size (batch size)  
* epochs (maximum number of epochs to optimize SPoSE model for)  
* window_size (window size to be used for checking convergence criterion with linear regression)  
* sampling_method (sampling method; if soft, then you can specify a fraction of your training data to be sampled from during each epoch; else full train set will be used)  
* steps (save model parameters and create checkpoints every <steps> epochs)  
* resume (bool) (whether to resume training at last checkpoint; if not set training will restart)  
* p (fraction of train set to sample; only necessary for *soft* sampling)  
* device (CPU or CUDA)  
* rnd_seed (random seed)  
* early_stopping (bool) (train until convergence)  
* num_threads (number of threads used by PyTorch multiprocessing)  

*Run model*  
You can run the model with the following code. Hyperparameters can be adjusted.  

! python master_thesis/SPoSE/train.py --task odd_one_out --modality behavioral/ --triplets_dir master_thesis/SPoSE/data/ --learning_rate 0.00719 --lmbda 0.0005 --temperature 0.641 --embed_dim 5 --batch_size 128 --epochs 15 --window_size 50 --steps 5 --sampling_method normal --device cuda:0 --rnd_seed 42 --distance_metric hyperbolic --c 2.286
