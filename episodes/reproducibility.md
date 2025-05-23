---
title: 'Containers in Research Workflows: Reproducibility and Granularity'
teaching: 20
exercises: 5
---

::::::::::::::::::::::::::::::::::::::: objectives

- Understand how container images can help make research more reproducible.
- Understand what practical steps I can take to improve the reproducibility of my research using containers.

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: questions

- How can I use container images to make my research more reproducible?
- How do I incorporate containers into my research workflow?

::::::::::::::::::::::::::::::::::::::::::::::::::

Although this workshop is titled "Reproducible computational environments using containers",
so far we have mostly covered the mechanics of using Docker and Singularity with only passing reference to
the reproducibility aspects. In this section, we discuss these aspects in more detail.

:::::::::::::::::::::::::::::::::::::::::  callout

## Work in progress...

Note that reproducibility aspects of software and containers are an active area of research, discussion and development so are subject to many changes. We will present some ideas and approaches here but best practices will likely evolve in the near future.


::::::::::::::::::::::::::::::::::::::::::::::::::

## Reproducibility

By *reproducibility* here we mean the ability of someone else (or your future self) being able to reproduce
what you did computationally at a particular time (be this in research, analysis or something else)
as closely as possible, even if they do not have access to exactly the same hardware resources
that you had when you did the original work.

What makes this especially important? With research being increasingly digital
in nature, more and more of our research outputs are a result of the use of
software and data processing or analysis. With complex software stacks or
groups of dependencies often being required to run research software, we need
approaches to ensure that we can make it as easy as possible to recreate an
environment in which a given research process was undertaken. There many
reasons why this matters, one example being someone wanting to reproduce
the results of a publication in order to verify them and then build on that
research. 

Some examples of why containers are an attractive technology to help with reproducibility include:

- The same computational work can be run seamlessly on different operating systems (e.g. Windows, macOS, Linux).
- You can save the exact process that you used for your computational work (rather than relying on potentially incomplete notes).
- You can save the exact versions of software and their dependencies in the container image.
- You can provide access to legacy versions of software and underlying dependencies which may not be generally available any more.
- Depending on their size, you can also potentially store a copy of key data within the container image.
- You can archive and share a container image as well as associating a persistent identifier with it, to allow other researchers to reproduce and build on your work.

## Sharing images

As we have already seen, the Docker Hub provides a platform for sharing container images publicly. Once you have uploaded a container image, you can point people to its public location and they can download and build upon it.

This is fine for working collaboratively with container images on a day-to-day basis but the Docker Hub is not a good option for long-term archiving of container images in support of research and publications as:

- free accounts have a limit on how long a container image will be hosted if it is not updated
- it does not support adding persistent identifiers to container images
- it is easy to overwrite tagged container images with newer versions by mistake.

## Archiving and persistently identifying container images using Zenodo

When you publish your work or make it publicly available in some way it is good practice to make container images that you used for computational work available in an immutable, persistent way and to have an identifier that allows people to cite and give you credit for the work you have done. [Zenodo](https://zenodo.org/) is one service that provides this functionality.

Zenodo supports the upload of *tar* archives and we can capture our Docker container images as tar archives using the `docker image save` command. For example, to export the container image we created earlier in this lesson:

```bash
docker image save alice/alpine-python:v1 -o alpine-python.tar
```

These tar container images can become quite large and Zenodo supports uploads up to 50GB so you may need to compress your archive to make it fit on Zenodo using a tool such as gzip (or zip):

```bash
gzip alpine-python.tar
```

Once you have your archive, you can [deposit it on Zenodo](https://zenodo.org/deposit/) and this will:

- Create a long-term archive snapshot of your Docker container image which people (including your future self) can download and reuse or reproduce your work.
- Create a persistent DOI (*Digital Object Identifier*) that you can cite in any publications or outputs to enable reproducibility and recognition of your work.

In addition to the archive file itself, the deposit process will ask you to provide some basic metadata to classify the container image and the associated work.

Note that Zenodo is not the only option for archiving and generating persistent DOIs for container images. There are other services out there -- for example, some organizations may provide their own, equivalent, service.

## Reproducibility good practice

- Make use of container images to capture the computational environment required for your work.
- Decide on the appropriate granularity for the container images you will use for your computational work -- this will be different for each project/area. Take note of accepted practice from contemporary work in the same area. What are the right building blocks for individual container images in your work?
- Document what you have done and why -- this can be put in comments in the `Dockerfile` and the use of the container image described in associated documentation and/or publications. Make sure that references are made in both directions so that the container image and the documentation are appropriately linked.
- When you publish work (in whatever way) use an archiving and DOI service such
  as Zenodo to make sure your container image is captured as it was used for
  the work and that it is assigned a persistent DOI to allow it to be cited and
  referenced properly.
- Make use of tags when naming your container images, this ensures that if you
  update the image in future, previous versions can be retained within a
  container repository to be easily accessed, if this is required.
- A built and archived container image can ensure a persistently bundled set of
  software and dependecies. However, a `Dockerfile` provides a lightweight
  means of storing a container definition that can be used to re-create a
  container image at a later time. If you're taking this approach, ensure that
  you specify software package and dependency versions within your `Dockerfile`
  rather than just specifying package names which will generally install the
  most up-to-date version of a package. This may be incompatible with other
  elements of your software stack. Also note that storing only a `Dockerfile`
  presents reproducibility challenges because required versions of packages may
  not be available indefinitely, potentially meaning that you're unable to
  reproduce the required environment and, hence, the research results.

## Container Granularity

As mentioned above, one of the decisions you may need to make when containerising your research workflows
is what level of *granularity* you wish to employ. The two extremes of this decision could be characterized
as:

- Create a single container image with all the tools you require for your research or analysis workflow
- Create many container images each running a single command (or step) of the workflow and use them together

Of course, many real applications will sit somewhere between these two extremes.

:::::::::::::::::::::::::::::::::::::::  challenge

## Positives and negatives

What are the advantages and disadvantages of the two approaches to container granularity for research
workflows described above? Think about this
and write a few bullet points for advantages and disadvantages for each approach in the course Etherpad.

:::::::::::::::  solution

## Solution

This is not an exhaustive list but some of the advantages and disadvantages could be:

### Single large container image

- Advantages:
  - Simpler to document
  - Full set of requirements packaged in one place
  - Potentially easier to maintain (though could be opposite if working with large, distributed group)
- Disadvantages:
  - Could get very large in size, making it more difficult to distribute
    - Could use [Docker multi-stage build](https://docs.docker.com/develop/develop-images/multistage-build) to reduce size
  - May end up with same dependency issues within the container image from different software requirements
  - Potentially more complex to test
  - Less re-useable for different, but related, work

### Multiple smaller container images

- Advantages:
  - Individual components can be re-used for different, but related, work
  - Individual parts are smaller in size making them easier to distribute
  - Avoid dependency issues between different pieces of software
  - Easier to test
- Disadvantage:
  - More difficult to document
  - Potentially more difficult to maintain (though could be easier if working with large, distributed group)
  - May end up with dependency issues between component container images if they get out of sync
    
    

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::  challenge

## Next steps with containers

Now that we're at the end of the lesson material, take a moment to reflect on
what you've learned, how it applies to you, and what to do next.

1. In your own notes, write down or diagram your understanding of Docker containers and container images:
  concepts, commands, and how they work.
2. In the workshop's shared notes document, write down how you think you might
  use containers in your daily work. If there's something you want to try doing with
  containers right away, what is a next step after this workshop to make that happen?

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: keypoints

- Container images allow us to encapsulate the computation (and data) we have used in our research.
- Using a service such as Docker Hub allows us to easily share computational work we have done.
- Using container images along with a DOI service such as Zenodo allows us to capture our work and enables reproducibility.

::::::::::::::::::::::::::::::::::::::::::::::::::


