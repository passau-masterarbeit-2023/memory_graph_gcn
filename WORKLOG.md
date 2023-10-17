# GCN processing of memory graphs

Goal: ML for Key detection > We want to learn a model using a Graph Convolution Network on our annotated memory graphs.

## Logs

### Tue 17 Oct 2023

```
y_trained_in_list[0] type: <class 'numpy.ndarray'>
y_trained_in_list[0].shape: (788,)
X_train.shape: (35808, 128)
y_train.shape: (13032,)
X_test.shape: (11936, 128)
y_test.shape: (3674,)
```

This shows that the padding operation actually adds a lot of empty data, 35808 final rows for only 13032 original ones after having embedded 12 graphs (so a little more that 1000 nodes per graph).

### Mon 16 Oct 2023

IDEA: Gensim's Word2Vec: After generating random walks, you can treat them as "sentences" and the nodes as "words" to generate embeddings using Gensim's Word2Vec model. > UPDATE: Actually, the implementation of Node2Vec relies on Gensim Word2Vec.

The imbalanceness of the graph is too big. I need to find a way to reduce the number of nodes in graphs.

* [ ] Remove Chunk nodes that are solitary, i.e. that are not connected to any other node.
* [ ] Try removing and keeping nodes based on entropy.

I would also like to complete the Node2Vec embedding with additionnal information. Try adding embedding from other mem2graph embeddings.

### Sun 15 Oct 2023

Fixed data loading.

### Wed 11 Oct 2023

Re-starting again the work effort on GNC.

### Tue 8 Aug 2023

Re-starting to work on GNC project to get us back into the masterarbeit.

* [ ] INSTALL CUDA, for PyTorch...
* [ ] Update Conda environment
* [ ] Use a library to convert the graphviz graph into a GNC
* [ ] Learn the GNC
* [ ] Test it
