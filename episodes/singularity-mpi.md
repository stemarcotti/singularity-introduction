---
title: Running MPI parallel jobs using Singularity containers
teaching: 30
exercises: 40
---

::::::::::::::::::::::::::::::::::::::: objectives

- Learn how MPI applications within Singularity containers can be run on HPC platforms
- Understand the challenges and related performance implications when running MPI jobs via Singularity

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: questions

- How do I set up and run an MPI job from a Singularity container?

::::::::::::::::::::::::::::::::::::::::::::::::::

## Running MPI parallel codes with Singularity containers

### MPI overview

MPI - [Message Passing Interface](https://en.wikipedia.org/wiki/Message_Passing_Interface) - is a widely used standard for parallel programming. It is used for exchanging messages/data between processes in a parallel application. If you've been involved in developing or working with computational science software, you may already be familiar with MPI and running MPI applications.

When working with an MPI code on a large-scale cluster, a common approach is to compile the code yourself, within your own user directory on the cluster platform, building against the supported MPI implementation on the cluster. Alternatively, if the code is widely used on the cluster, the platform administrators may build and package the application as a module so that it is easily accessible by all users of the cluster.

### MPI codes with Singularity containers

We've already seen that building Singularity containers can be impractical without root access. Since we're highly unlikely to have root access on a large institutional, regional or national cluster, building a container directly on the target platform is not normally an option.

If our target platform uses [OpenMPI](https://www.open-mpi.org/), one of the two widely used source MPI implementations, we can build/install a compatible OpenMPI version on our local build platform, or directly within the image as part of the image build process. We can then build our code that requires MPI, either interactively in an image sandbox or via a definition file.

If the target platform uses a version of MPI based on [MPICH](https://www.mpich.org/), the other widely used open source MPI implementation, there is [ABI compatibility between MPICH and several other MPI implementations](https://www.mpich.org/abi/). In this case, you can build MPICH and your code on a local platform, within an image sandbox or as part of the image build process via a definition file, and you should be able to successfully run containers based on this image on your target cluster platform.

As described in Singularity's [MPI documentation](https://sylabs.io/guides/3.5/user-guide/mpi.html), support for both OpenMPI and MPICH is provided. Instructions are given for building the relevant MPI version from source via a definition file and we'll see this used in an example below.

#### **Container portability and performance on HPC platforms**

While building a container on a local system that is intended for use on a remote HPC platform does provide some level of portability, if you're after the best possible performance, it can present some issues. The version of MPI in the container will need to be built and configured to support the hardware on your target platform if the best possible performance is to be achieved. Where a platform has specialist hardware with proprietary drivers, building on a different platform with different hardware present means that building with the right driver support for optimal performance is not likely to be possible. This is especially true if the version of MPI available is different (but compatible). Singularity's [MPI documentation](https://sylabs.io/guides/3.5/user-guide/mpi.html) highlights two different models for working with MPI codes. The *[hybrid model](https://sylabs.io/guides/3.5/user-guide/mpi.html#hybrid-model)* that we'll be looking at here involves using the MPI executable from the MPI installation on the host system to launch singularity and run the application within the container. The application in the container is linked against and uses the MPI installation within the container which, in turn, communicates with the MPI daemon process running on the host system. In the following section we'll look at building a Singularity image containing a small MPI application that can then be run using the hybrid model.

### Building and running a Singularity image for an MPI code

#### **Building and testing an image**

This example makes the assumption that you'll be building a container image on a local platform and then deploying it to a cluster with a different but compatible MPI implementation. See [Singularity and MPI applications](https://sylabs.io/guides/3.5/user-guide/mpi.html#singularity-and-mpi-applications) in the Singularity documentation for further information on how this works.

We'll build an image from a definition file. Containers based on this image will be able to run MPI benchmarks using the [OSU Micro-Benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/) software.

In this example, the target platform is a remote HPC cluster that uses [Intel MPI](https://software.intel.com/content/www/us/en/develop/tools/mpi-library.html). The container can be built via the Singularity Docker image that we used in the previous episode of the Singularity material.

Begin by creating a directory and, within that directory, downloading and saving the "tarballs" for version 5.7.1 of the OSU Micro-Benchmarks from the [OSU Micro-Benchmarks page](https://mvapich.cse.ohio-state.edu/benchmarks/) and for MPICH version 3.4.2 from the [MPICH downloads page](https://www.mpich.org/downloads/).

In the same directory, save the following definition file content to a `.def` file, e.g. `osu_benchmarks.def`:

```output
Bootstrap: docker
From: ubuntu:20.04

%files
    /home/singularity/osu-micro-benchmarks-5.7.1.tgz /root/
    /home/singularity/mpich-3.4.2.tar.gz /root/

%environment
    export SINGULARITY_MPICH_DIR=/usr
    export OSU_DIR=/usr/local/osu/libexec/osu-micro-benchmarks/mpi

%post
    apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential libfabric-dev libibverbs-dev gfortran
    cd /root
    tar zxvf mpich-3.4.2.tar.gz && cd mpich-3.4.2
    echo "Configuring and building MPICH..."
    ./configure --prefix=/usr --with-device=ch3:nemesis:ofi && make -j2 && make install
    cd /root
    tar zxvf osu-micro-benchmarks-5.7.1.tgz
    cd osu-micro-benchmarks-5.7.1/
    echo "Configuring and building OSU Micro-Benchmarks..."
    ./configure --prefix=/usr/local/osu CC=/usr/bin/mpicc CXX=/usr/bin/mpicxx
    make -j2 && make install

%runscript
    echo "Rank ${PMI_RANK} - About to run: ${OSU_DIR}/$*"
    exec ${OSU_DIR}/$*
```

A quick overview of what the above definition file is doing:

- The image is being bootstrapped from the `ubuntu:20.04` Docker image.
- In the `%files` section: The OSU Micro-Benchmarks and MPICH tar files are copied from the current directory into the `/root` directory within the image.
- In the `%environment` section: Set a couple of environment variables that will be available within all containers run from the generated image.
- In the `%post` section:
  - Ubuntu's `apt-get` package manager is used to update the package directory and then install the compilers and other libraries required for the MPICH build.
  - The MPICH .tar.gz file is extracted and the configure, build and install steps are run. Note the use of the `--with-device` option to configure MPICH to use the correct driver to support improved communication performance on a high performance cluster.
  - The OSU Micro-Benchmarks .tar.gz file is extracted and the configure, build and install steps are run to build the benchmark code from source.
- In the `%runscript` section: A runscript is set up that will echo the rank number of the current process and then run the command provided as a command line argument.

*Note that base path of the the executable to run (`$OSU_DIR`) is hardcoded in the run script*. The command line parameter that you provide when running a container instance based on the image is then added to this base path. Example command line parameters include: `startup/osu_hello`, `collective/osu_allgather`, `pt2pt/osu_latency`, `one-sided/osu_put_latency`.

:::::::::::::::::::::::::::::::::::::::  challenge

## Build and test the OSU Micro-Benchmarks image

Using the above definition file, build a Singularity image named `osu_benchmarks.sif`.

Once the image has finished building, test it by running the `osu_hello` benchmark that is found in the `startup` benchmark folder.

*NOTE: If you're not using the Singularity Docker image to build your Singularity image, you will need to edit the path to the .tar.gz file in the `%files` section of the definition file.*

:::::::::::::::  solution

## Solution

You should be able to build an image from the definition file as follows:

```bash
$ singularity build osu_benchmarks.sif osu_benchmarks.def
```

*Note that if you're running the Singularity Docker container directly from the command line to undertake your build, you'll need to provide the full path to the `.def` file **within** the container* - it is likely that this will be different to the file path on your host system. For example, if you've bind mounted the directory on your local system containing the file to `/home/singularity` within the container, the full path to the `.def` file will be `/home/singularity/osu_benchmarks.def`.

Assuming the image builds successfully, you can then try running the container locally and also transfer the SIF file to a cluster platform that you have access to (that has Singularity installed) and run it there.

Let's begin with a single-process run of `startup/osu_hello` on *your local system* (where you built the container) to ensure that we can run the container as expected. We'll use the MPI installation *within* the container for this test. *Note that when we run a parallel job on an HPC cluster platform, we use the MPI installation on the cluster to coordinate the run so things are a little different...*

Start a shell in the Singularity container based on your image and then run a single process job via `mpirun`:

```bash
$ singularity shell --contain /home/singularity/osu_benchmarks.sif
Singularity> mpirun -np 1 $OSU_DIR/startup/osu_hello
```

You should see output similar to the following:

```output
# OSU MPI Hello World Test v5.7.1
This is a test with 1 processes
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

#### **Running Singularity containers via MPI**

Assuming the above tests worked, we can now try undertaking a parallel run of one of the OSU benchmarking tools within our container image.

This is where things get interesting and we'll begin by looking at how Singularity containers are run within an MPI environment.

If you're familiar with running MPI codes, you'll know that you use `mpirun` (as we did in the previous example), `mpiexec` or a similar MPI executable to start your application. This executable may be run directly on the local system or cluster platform that you're using, or you may need to run it through a job script submitted to a job scheduler. Your MPI-based application code, which will be linked against the MPI libraries, will make MPI API calls into these MPI libraries which in turn talk to the MPI daemon process running on the host system. This daemon process handles the communication between MPI processes, including talking to the daemons on other nodes to exchange information between processes running on different machines, as necessary.

When running code within a Singularity container, we don't use the MPI executables stored within the container (i.e. we DO NOT run `singularity exec mpirun -np <numprocs> /path/to/my/executable`). Instead we use the MPI installation on the host system to run Singularity and start an instance of our executable from within a container for each MPI process. Without Singularity support in an MPI implementation, this results in starting a separate Singularity container instance within each process. This can present some overhead if a large number of processes are being run on a host. Where Singularity support is built into an MPI implementation this can address this potential issue and reduce the overhead of running code from within a container as part of an MPI job.

Ultimately, this means that our running MPI code is linking to the MPI libraries from the MPI install within our container and these are, in turn, communicating with the MPI daemon on the host system which is part of the host system's MPI installation. In the case of MPICH, these two installations of MPI may be different but as long as there is [ABI compatibility](https://wiki.mpich.org/mpich/index.php/ABI_Compatibility_Initiative) between the version of MPI installed in your container image and the version on the host system, your job should run successfully.

We can now try running a 2-process MPI run of a point to point benchmark `osu_latency`. If your local system has both MPI and Singularity installed and has multiple cores, you can run this test on that system. Alternatively you can run on a cluster. Note that you may need to submit this command via a job submission script submitted to a job scheduler if you're running on a cluster. If you're attending a taught version of this course, some information will be provided below in relation to the cluster that you've been provided with access to.

:::::::::::::::::::::::::::::::::::::::  challenge

## Undertake a parallel run of the `osu_latency` benchmark (general example)

Move the `osu_benchmarks.sif` Singularity image onto the cluster (or other suitable) platform where you're going to undertake your benchmark run.

You should be able to run the benchmark using a command similar to the one shown below. However, if you are running on a cluster, you may need to write and submit a job submission script at this point to initiate running of the benchmark.

```bash
$ mpirun -np 2 singularity run osu_benchmarks.sif pt2pt/osu_latency
```

:::::::::::::::  solution

## Expected output and discussion

As you can see in the mpirun command shown above, we have called `mpirun` on the host system and are passing to MPI the `singularity` executable for which the parameters are the image file and any parameters we want to pass to the image's run script, in this case the path/name of the benchmark executable to run.

The following shows an example of the output you should expect to see. You should have latency values shown for message sizes up to 4MB.

```output
Rank 1 - About to run: /.../mpi/pt2pt/osu_latency
Rank 0 - About to run: /.../mpi/pt2pt/osu_latency
# OSU MPI Latency Test v5.6.2
# Size          Latency (us)
0                       0.38
1                       0.34
...
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::  challenge

## Undertake a parallel run of the `osu_latency` benchmark (taught course cluster example)

***Note to instructors:** Add details into this box relating to running the above example on your chosen cluster platform. The example SLURM script provided here is based on the UK's ARCHER2 HPC platform, you should replace the template file in the `files` directory of the repository with a submission script suited to your platform, if one is required.*

This version of the exercise, for undertaking a parallel run of the osu\_latency benchmark with your Singularity container that contains an MPI build, is specific to this run of the course.

The information provided here is specifically tailored to the HPC platform that you've been given access to for this taught version of the course.

Move the `osu_benchmarks.sif` Singularity image onto the cluster where you're going to undertake your benchmark run. You should use `scp` or a similar utility to copy the file.

The platform you've been provided with access to uses `Slurm` to schedule jobs to run on the platform. You now need to create a `Slurm` job submission script to run the benchmark.

Download this [template script]({{site.url}}files/osu_latency.slurm.template) and edit it to suit your configuration.

Submit the modified job submission script to the `Slurm` scheduler using the `sbatch` command.

```bash
$ sbatch osu_latency.slurm
```

:::::::::::::::  solution

## Expected output and discussion

As you will have seen in the commands using the provided template job submission script, we have called `mpirun` on the host system and are passing to MPI the `singularity` executable for which the parameters are the image file and any parameters we want to pass to the image's run script. In this case, the parameters are the path/name of the benchmark executable to run.

The following shows an example of the output you should expect to see. You should have latency values shown for message sizes up to 4MB.

```output
INFO:    Convert SIF file to sandbox...
INFO:    Convert SIF file to sandbox...
Rank 1 - About to run: /.../mpi/pt2pt/osu_latency
Rank 0 - About to run: /.../mpi/pt2pt/osu_latency
# OSU MPI Latency Test v5.6.2
# Size          Latency (us)
0                       1.49
1                       1.50
2                       1.50
...
4194304               915.44
INFO:    Cleaning up image...
INFO:    Cleaning up image...
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

This has demonstrated that we can successfully run a parallel MPI executable from within a Singularity container. However, in this case, the two processes will almost certainly have run on the same physical node so this is not testing the performance of the interconnects between nodes.

You could now try running a larger-scale test. You can also try running a benchmark that uses multiple processes, for example try `collective/osu_gather`.

:::::::::::::::::::::::::::::::::::::::  challenge

## Investigate performance when using a container image built on a local system and run on a cluster

To get an idea of any difference in performance between the code within your Singularity image and the same code built natively on the target HPC platform, try building the OSU benchmarks from source, locally on the cluster. Then try running the same benchmark(s) that you ran via the singularity container. Have a look at the outputs you get when running `collective/osu_gather` or one of the other collective benchmarks to get an idea of whether there is a performance difference and how significant it is.

Try running with enough processes that the processes are spread across different physical nodes so that you're making use of the cluster's network interconnects.

What do you see?

:::::::::::::::  solution

## Discussion

You may find that performance is significantly better with the version of the code built directly on the HPC platform. Alternatively, performance may be similar between the two versions.

How big is the performance difference between the two builds of the code?

What might account for any difference in performance between the two builds of the code?

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

If performance is an issue for you with codes that you'd like to run via Singularity, you are advised to take a look at using the *[bind model](https://sylabs.io/guides/3.5/user-guide/mpi.html#bind-model)* for building/running MPI applications through Singularity.

:::::::::::::::::::::::::::::::::::::::: keypoints

- Singularity images containing MPI applications can be built on one platform and then run on another (e.g. an HPC cluster) if the two platforms have compatible MPI implementations.
- When running an MPI application within a Singularity container, use the MPI executable on the host system to launch a Singularity container for each process.
- Think about parallel application performance requirements and how where you build/run your image may affect that.

::::::::::::::::::::::::::::::::::::::::::::::::::


