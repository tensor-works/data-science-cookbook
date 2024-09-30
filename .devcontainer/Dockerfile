# Use a smaller CUDA runtime image instead of the devel version
# Set the base image
FROM tensorpod/torch-and-tf2-on-cuda:pt2.3-cu118-tf2.14.0-py3.9-no-ta-tv

# Configure OS encoding for stability
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# Point PATH to upcoming conda install
ENV PATH=/opt/conda/bin:$PATH

# Install minimal dependencies
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
    bzip2 \
    ca-certificates \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /workdir
RUN chmod -R +x /workdir

# Copy only the conda environment file first to leverage stable layer caching
COPY .devcontainer/linux-gnu.yml .devcontainer/linux-gnu.yml

# Install the conda environment
RUN conda env update -n base -f .devcontainer/linux-gnu.yml && \
    conda clean -afy

# Copy the rest of the files
COPY . .
