# CudaQ Environment

## Option 1: cscigpu (Recommended)

The cscigpu server `<user>@cscigpu.csuchico.edu` already has CudaQ for C++ and Python installed.

To access, first make sure you are connected through the CSU Chico GlobalProtect VPN, then run:
```
ssh <user>@cscigpu.csuchico.edu
```
Now you can move on to the [setup](#setup) section.

## Option 2: Local Installation

Based on instructions provided in the [CudaQ Documentation](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html#install-cuda-q)

- Verify that you have cuda version 12+ installed
```{sh}
nvcc --version
```
- Download latest CUDA-Q version [here](https://github.com/NVIDIA/cuda-quantum/releases)
- Install with the following command:
```{sh}
sudo -E bash install_cuda_quantum*.$(uname -m) --accept
. /etc/profile
```

# Setup

It is recommended to connect your github account with SSH: [Instructions here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

```{sh}
git clone git@github.com:collinsjacob127/quantum-research.git && \
cd quantum-research && \
make test-install && \
./test-install.x
```


