# Phen-KGE
 Repurpose GWAS Data to Prioritize Functional Gene for Arabidopsis Using Knowledge Graph
一. DGL-KE
DGL-KE is a deep learning framework based on DGL graphs (https://github.com/dmlc/dgl), and a high-performance and highly scalable open source software library for the field of knowledge graph embedded learning methods. Phen-KGE uses the DGL-KE software package generated the low-latitude embedding vector representation of the Phen-KGE knowledge graph. The dataset of Phen-KGE is available at the folder of KG data.

import sys
sys.path.insert(1, '../utils')
from utils import
download_and_extractdownload_and_extract()
drkg_file = '../data/drkg/drkg.tsv'

The Phen-KGE knowledge graph contains a txt format file KGdata.txt, which contains all the triples of the knowledge graph. Before training, we randomly divide the data set into training set, validation set and test set at a ratio of 0.9: 0.05: 0.05 .
Then directly call the command line of the DGL-KE software package to train the low-latitude embedding vector representation of the Phen-KGE knowledge graph. In the case, we choose the TransE_l2 knowledge graph embedding algorithm, and use the AWS p3.16xlarge instance for multi-GPU parallel training (using other For the knowledge graph embedding algorithm and other models, please refer to the description in https://aws-dglke.readthedocs.io/en/latest/index.html).

!DGLBACKEND=pytorch dglke_train --dataset DRKG --data_path ./train --data_files drkg_train.tsv drkg_valid.tsv drkg_test.tsv --format 'raw_udd_hrt' --model_name TransE_l2 --batch_size 2048 \--neg_sample_size 256 --hidden_dim 400 --gamma 12.0 --lr 0.1 --max_step 100000 --log_interval 1000 --batch_size_eval 16 -adv --regularization_coef 1.00E-07 --test --num_thread 1 --gpu 0 1 2 3 4 5 6 7 --num_proc 8 --neg_sample_size_eval 10000 --async_update

After training, we will get two files: 1) PRKG_TransE_l2_entity.npy, the low-dimensional vector representation of the entity in Phen-KGE and 2) PRKG_TransE_l2_relation.npy, the low-dimensional vector representation of the relationship in Phen-KGE.

二. The relationship of Phenotype-gene Prediction
We define the gene prediction problem of the new coronavirus based on the Phen-KGE knowledge map as predicting the relationship between the gene and the new coronavirus entity between 'AraGWAS::PrG::Phenotype:Gene' and ' KnetMiner::PrG::Phenotype:Gene ' (ie Trait relationship) confidence assessment problem.
First, to verify the reliability of all methods, a “gold standard” data about functional genes of A. thaliana was downloaded from FLOR-ID (http://www.phytosystems.ulg.ac.be/florid/). FLOR-ID was used to calculate functional gene enrichment rate for the days to flowering trait.
Then, we predict the scores of all possible (gene and phenotype) triad combinations under the TrainsE_l2 algorithm, and finally sort the scores, and select the top 20% genes with the highest scores as recommended genes.
