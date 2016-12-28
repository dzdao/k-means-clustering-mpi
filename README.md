# k-Means Clustering: *A Distributed MPI Implementation*

The aim of the [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) algorithm (also known as Lloyd's algorithm) is to take a given dataset and divide it up into k number of clusters. The goal is to have some sort of representative clusters for each of its surrounding data points. It algorithm is [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) method and hopes to gain some sort of insight into data that may be unseemingly unrelated.

### Distributed Functionalities

- Given that k-means clustering problem is a NP-Hard computationally intensive problem, a distributed approach seemed appropriate. By dividing up the computation load among many worker nodes, the program can be parallelized and sped up.
- The root node reads in the entire dataset and randomly chooses the initial k-means for each cluster.
- The root node broadcasts the k-means to each n number of workers.
- The root node distributes the dataset among n number of workers.
- Given its portion of the entire dataset, each worker does computational work against each k-means cluster and returns the result to the root.
- The algorithm repeats for X number of iterations.

#### Frameworks, modules, libraries, etc:
- [Docker Engine](https://docs.docker.com/engine/installation/linux/ubuntulinux/)
- [Docker base image of Alpine Linux](https://hub.docker.com/r/nlknguyen/alpine-mpich/) - Credit to NLKNguyen for taking the time to create this image
- [MPICH](https://www.mpich.org/documentation/guides/) - included in Docker base image

#### Deployment and running the program:
1. Install [Docker](https://docs.docker.com/engine/installation/linux/ubuntulinux/)
2. Make sure Docker is running:
```sh
$ sudo service docker start
```
3. Download NLKNguyen's prebuilt Docker base image:
```sh
$ docker pull nlknguyen/alpine-mpich
$ docker pull nlknguyen/alpine-mpich:onbuild
```
4. While in the directory containing source code, run the following the launch the image:
```sh
$ docker run --rm -it -v $(PWD):/project nlknguyen/alpine-mpich
```
5. Now with the Docker image running, compile the source code:
```sh
$ mpicc k-means.c -o k-means
```
6. Now run the program please include an argument in the command line right after the program name to match the number of running processes:
```sh
$ mpirun -n 3 ./k-means 3
```

### Current Limitations
- Creating too many worker nodes in hopes of further speeding up parallelization can actually impede its progress. There must be a balance between the workload to be distributed to each worker, and the actual number of worker nodes. If there are more workers than necessary, much of the computational efforts are spent distributing the work (message passing), and not spent computing the actual workload. Thus one of the defining characteristics of how well the distributed program will run is its message complexity and how many messages are being sent back and forth throughout the network.

### Future Goals
- Further optimize the algorithm by reducing message complexity
- Deploy on Amazon AWS cluster and test

