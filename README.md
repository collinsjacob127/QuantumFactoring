# Repo Summary

This repo is for CSU Chico students researching various quantum computing algorithms.

Code is written using [CUDA-Q C++](https://nvidia.github.io/cuda-quantum/latest/index.html).

## Notes

`QFT_OOP_addition.cpp` is the most recent update. 
It uses QFT-based addition to add the values in register 1 and register 2, with the output being sent to register 3. 
This is a necessary step towards implementing QFT-based multiplication for use in semiprime factoring.

## TODO:

- [x] Addition (Semiclassical)
- [x] Grover's (Simple)
- [ ] Grover's (Index search) ~90%
- [x] Inverse Addition (Semiclassical)
- [x] QFT Addition (Out of place)
- [x] QFT Inverse Addition (Out of place)
- [x] QFT Scaled Addition (Out of place)
- [x] QFT Multiplication (Out of place)
- [ ] **QFT Inverse Multiplication** (Out of place)
- [ ] QFT Addition (In-place)
- [ ] QFT Inverse Addition (In-place)
- [ ] QFT Scaled Addition (In-place)
- [ ] QFT Multiplication (In-place)
- [ ] **QFT Inverse Multiplication** (In-place)

## CudaQ Environment

### Option 1: cscigpu (Recommended)

The cscigpu server `<user>@cscigpu.csuchico.edu` already has CudaQ for C++ and Python installed.

To access, first make sure you are connected through the [CSU Chico GlobalProtect VPN](https://support.csuchico.edu/TDClient/1984/Portal/KB/?CategoryID=15690), then run:
```
ssh <user>@cscigpu.csuchico.edu
```
Now you can move on to the [setup](#setup) section.

### Option 2: Local Installation

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

## Setup

It is recommended to connect your github account with SSH: [Instructions here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

```{sh}
git clone git@github.com:collinsjacob127/QuantumFactoring.git && \
cd QuantumFactoring/test-install && \
make && \
./test-install.x
```


