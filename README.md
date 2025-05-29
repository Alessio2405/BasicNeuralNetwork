# Simple Neural Network in C

A modular, maintainable implementation of a feedforward neural network written from scratch in C.  
Supports arbitrary layer sizes via dynamic allocation, and includes example data loading, training, and testing.

---


## Features

- 🎯 **Dynamic**: Arbitrary input, hidden, and output sizes  
- 🔌 **Modular**: Clean separation into headers, implementation, and `main`  
- 🛠️ **Build**: `Makefile` for one-step compilation  

---

## Prerequisites

- **C compiler** (GCC recommended)  
- **make**  
- **Git**  

---

## Getting Started

```bash
git clone https://github.com/Alessio2405/BasicNeuralNetwork.git
cd simple_nn
make             # builds `bin/nn`
./bin/nn         # runs the XOR demo (using data/training_data_sample.txt)
```
