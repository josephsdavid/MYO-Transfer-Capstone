# MYO armband transfer/representation learning project

## Overview

overview goes here

## Resources

### Papers

* [Dataset description and transfer learning task discussion](https://arxiv.org/abs/1801.07756)
	* Got about 70% accuracy on nina 5 17 class, unsurprisingly great accuracy on smaller dataset , used CNNs and transfer learning + spectrogram
* [Multi Stream CNN](https://www.sciencedirect.com/science/article/pii/S0167865517304439), used nina 1, voting cnn, 85%
* [CNN and Stacked Sparse Autoencoder paper](https://www.mdpi.com/1424-8220/18/8/2497/htm) used CNN and stakced sparse autoencoders on 7 classes of nina, and did over time accuracies, where they showed that stacked sparse autoencoders with engineered features and CNNs were robust to changes over time, provides good precedence, but again just 6 classes + rest, while we are dong 13, 17, 24 classes including rest. They used bipolar EMG as are we
* [EMG in era of deep learning](https://www.mdpi.com/2504-2289/2/3/21/htm), worked with High definition SEMGs (expensive), got good results. Section 3.2.3 sites a masters thesis which explains that LSTMs are powerful for this, which again provides us a lot of precedence
* [The cited thesis](http://publications.lib.chalmers.se/records/fulltext/254980/254980.pdf) used nina 7 (2kHz) with 200 ms windows, got about 90% acc. Very well written thesis
* [Compact DNN paper](https://arxiv.org/pdf/1806.08641.pdf) used CNNs to get good accuracy on self recorded 15 class data. I really like how they measure speed of the predictions, we can do something similar
* [LCNN PAPER](https://sci-hub.tw/10.1109/cac.2018.8623035) Uses convolutional LSTM type models to get about 60% accuracy on nina 5 17 class. Their model architecture should be looked into further
* [Domain adaptation paper](https://arxiv.org/pdf/1901.06958.pdf) Does some fancy two stage training on Nina 1, gets good results with 12 and 8 class classification
* [2D CNN PAPER](http://www.ece.upatras.gr/skodras/data/uploads/pubs/ans-c117-prepress.pdf) made a big CNN, their use of the confusion matrix we would do well to follow. Got 53% acc on nina, also proposed their own database, which we might also evaluate on: [database](https://www.ieee-dataport.org/open-access/myoup). Just need to make an IEE account
* [Super crazy ensemble paper](https://ieeexplore.ieee.org/document/8840853) gets about comparable accuracy to us on nina 5 with a crazy crazy model. Not sure how they evaluated at all if someone else can read its very confusing. However both our gigantic lstm and our simple state sharing bidirectional lstm perform pretty well


### Lectures

* [Mike Mahoney lecture on regularization/understanding if a model is properly trained](https://www.youtube.com/watch?v=ILV5Sc8WjPY)

### Other Resources

Find some resources on LSTMs for classification!

## Project Structure

organizational notes go here
