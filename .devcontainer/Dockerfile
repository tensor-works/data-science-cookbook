# Use a smaller CUDA runtime image instead of the devel version
# Set the base image
FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

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

# Install light weight mini-forge3
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O miniforge.sh -q && \
    bash miniforge.sh -b -p /opt/conda && \
    rm miniforge.sh && \
    conda clean -afy

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

# Force clean the pip cache
RUN rm -rf /root/.cache/pip 
