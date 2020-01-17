# Week 1 Notes:

## Shuffle Shuffle

Shuffling the data is pretty intuitive, but I am not sure if it actually generalizes better. In all examples I have seen where they use LSTM classifiers or autoencoders, they shuffle. This should be tested

## Ways to expand LSTM classifier
 
* Currently, stacking LSTM layers (looking at different timescales), is not promising, nor is it intuitive
* Deeper network between LSTM and output layer is a promising approach, and makes sense for transfer learning
* Deeper network before the LSTM (maybe a convLSTM) makes a lot of sense. Would need to figure out how to get the data in the proper format at each step.
* How many nodes in single layer LSTM? No idea. Currently the simple LSTM outperforms the wide LSTM which in turn outperforms the stacked LSTM


## Transfer Learning with LSTMs

* We need to decide if we want to fine tune, build a progressive network, or what, and probably experiment between them
* A second promising approach, training the initial network in an unsupervised manner, as discussed in [this paper](https://arxiv.org/abs/1502.04681). We have two options here again:
	* Pretrained LSTM encoder -> attach classifier to the end 
	* Initialize LSTM classifier with weights from LSTM encoder


## Mahoney Magic

I have no idea if we can do the weight matrix tricks mahoney did in his lecture/paper with an RNN, however, we have the benefit of being the ones who train the network, which means it would be trivial to write a callback which pulls weight matrices every epoch, and then we can see what we can do from there

## Other thoughts

* It is easy to build a good classifier on HAR with just a standard autoencoded embedding (for example in N2D). May be worthwhile to figure out how to get the data in that format, and try that out. (I think a classifier which is Autoencoded Embedding -> UMAP to make the data easily seperable -> simple classifier could be promising, intuitive, and simple)
* I still dont know if the windowing function is right. I believe we have the same shape as the original paper, but not sure if the windows are sliding properly
* I think it might make sense to, instead of padding each series to the longest, which can give us for example 40+ zeroes at the end of our series, pad and cut so each series is of length 1000, or maybe keep them as lists, no padding or cutting, and write a generator
* I am not sure why, but the high pass butterworth filter is definitely helping, look into this!!


## How I would currently organize the paper

* Abstract and intro
	* Novelty for shaibal: use of recurrent transfer learning in a variety of settings, potential use of the matrix stuff to tell if our networks are properly trained, potential extrapolation to legs, potential use of unsupervised pretraining 
* Literature review: we will have plenty of stuff to review
* Discussion of data preprocessing and augmentation
* Discussion of the aspects of our models: TL, RNNs, maybe unsupervised pretraining, maybe representation learning with simple classifiers
	* Models I think are worth considering:
		* (Definite) LSTM classifier TL 
		* (Definite) ConvLSTM classifier TL
		* (Maybe) LSTM autoencoder transferred to classifier
		* (Maybe) LSTM autoencoder to something like UMAP to simple classifier
		* (Maybe) Previous two with ConvLSTM autoencoders
		* (Maybe) Previous two with Convolutional autoencoders, would be a good reference to original paper
		* (Maybe) Previous two with standard autoencoders and properly formatted data
	* Datasets to generalize to:
		* Our test set
		* NinaPro (try same 17 as original paper, maybe even try all 52!)
		* (if possible) MS dataset
* Discussion of training procedures as well as results (matrices go here). The original paper built the models on the pretraining dataset, saved those models, built models of the same architecture, trained them 1-4 cycles on evaluation training0 set, validated on Test0, tested on Test1, and then froze the pretrained networks, attaced on new networks, repeated, and compared results. I think it would be also useful to (when possible) evaluate how well the pretrained networks directly on the Test1 dataset to show how well or poorly they generalize and justify the use of transfer learning further
* Discussion of test results
* Conclusion
