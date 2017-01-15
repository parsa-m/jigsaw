# jigsaw

Python-Caffe Implementation of Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles by Mehdi Noroozi and Paolo Favaro
with close to 2x training time speedup on author's implementation due to multiprocessing and batch reshaping replacing siamese net.   

This implementation achieved 66.8% on PASCAL VOC 2007 Classification.   

This implementation has parameters for training with ilsvrc 2012 dataset.  
Change load_data function in utils.py for loading a different dataset.   
Set solver parameters in utils.py.   
Set paths and tuning parameters in beginning of train.py.  


Thanks to Evan Shelhamer for batch reshape idea and guidance.      
Thanks to Deepak Pathak and Mehdi Noroozi for help with evaluation.       
Credit to Carl Doersch's deepcontext GitHub repo mostly for multiprocessing inspiration.      
Credit to Philipp Krähenbühl voc-classification Github repo for evaluation.    

