/**********************************
 * Description: 
 * Author: Jacob Collins
 * Instructions:
 *   Compile and run with:
 *   ```
 *   $> make QFT_addition.o
 *   $> ./QFT_addition.o 
 *   
 *  
 *   ```
 **********************************/
#include <cudaq.h>
#include <math.h>
#include <numbers>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "helpers.h"
// # define M_PIl          3.141592653589793238462643383279502884L /* pi */

#define ENABLE_DEBUG true
#define NUMBER_OF_SHOTS 2000


/****************** CUDAQ FUNCS ******************/
// Apply NOT-gates in accordance with bit-pattern of given integer.
// Bit-order is reversed. i.e. 3 (00011) would be (11000)
__qpu__ void setInt(const long val, cudaq::qview<> qs) {
  // Iterate through bits in val
  for (int i = 1; i <= qs.size(); ++i) {
    // Bit-shift for single bitwise AND to apply X on correct qubits
    auto target_bit_set = (1 << (qs.size() - i)) & val;
    if (target_bit_set) x(qs[qs.size() - i]);
  }
}

// Converted from python from cudaq docs
// https://nvidia.github.io/cuda-quantum/latest/applications/python/quantum_fourier_transform.html#Quantum-Fourier-Transform
__qpu__ void quantumFourierTransform(cudaq::qview<> qs) {
  const int nbits = qs.size();
  double phase;
  for (int i = 0; i < nbits; ++i) {
    h(qs[i]);
    for (int j = i + 1; j < nbits; ++j) {
      phase = (2 * std::numbers::pi) / pow(2, j - i + 1);
      cr1(phase, qs[j], qs[i]);
    }
  }
}

struct QFTAdder {
  void operator()(cudaq::qview<> y_reg, cudaq::qview<> z_reg) __qpu__ {
    const int nbits_y = y_reg.size();
    const int nbits_z = z_reg.size();
    int exponent_term;
    double phase;
    for (int z_ind = nbits_z-1; z_ind >= 0; ++z_ind) {
      for (int y_ind = 0; y_ind < nbits_y; ++y_ind) {
        int s = y_ind + 1;
        int j = z_ind + 1;
        exponent_term = (nbits_y - j) - s;
        if (exponent_term > 0) {
          phase = 2 * std::numbers::pi / pow(2, (j + s - nbits_y));
        } else {
          phase = 2 * std::numbers::pi * pow(2, (nbits_y - j - s));
          // phase = 0;
        }
        //TODO: This has been rewritten like 8 times and is still super wonky.
        //      Got to re-read this: https://arxiv.org/pdf/2309.10204
        // phase = pow((double) c * M_PI / 2, j);
        // phase = 2 * std::numbers::pi / pow(2, j);
        // if (ENABLE_DEBUG) { printf("Phase: %lf\n  y_i: %d\n  z_i: %d\n  j: %d\n", phase, y_ind, z_ind, j); }
        cr1(phase, y_reg[y_ind], z_reg[z_ind]);
        // r1<cudaq::ctrl>(phase, y_reg[y_ind], z_reg[z_ind]);
      }
    }
  }
};

/****************** CUDAQ STRUCTS ******************/
// Driver for adder
struct run_adder {
  __qpu__ auto operator()(const long y, const long z,
                          const int nbits_y, const int nbits_z) {
    // 1. Initialize Registers
    QFTAdder add_op;
    cudaq::qvector q_reg(nbits_y + nbits_z);  // Value 1 reg
    cudaq::qview y_reg = q_reg.front(nbits_y);
    cudaq::qview z_reg = q_reg.back(nbits_z);  // Value 2 reg
    setInt(y, y_reg);
    setInt(z, z_reg);

    // 2. QFT
    for (int i = 0; i < 10; ++i) {
      quantumFourierTransform(z_reg);

      // 2. Add
      add_op(y_reg, z_reg);
      // cudaq::adjoint(add_op, y_reg, z_reg, c);

      // 4. IQFT
      cudaq::adjoint(quantumFourierTransform, z_reg);
    }
    // 3. Measure
    mz(q_reg);
  }
};

void displayFullResults(std::vector<std::tuple<std::string, size_t>> results, long y, long z, int nbits_y, int nbits_z, size_t n_printed=5) {
  size_t n_shots = NUMBER_OF_SHOTS;
  size_t total_correct = 0;
  int i = 0;
  for (auto item : results) {
    // Binary result string
    std::string result = std::get<0>(item);
    // Count of this outcome being measured
    size_t count = std::get<1>(item);
    // Parse
    std::string y_out = result.substr(0, nbits_y);
    std::string z_out = result.substr(nbits_y, nbits_z);
    int y_val = binToInt(y_out);
    int z_val = binToInt(z_out);
    // % of whole
    if (i < n_printed) {
      printf("%lu + %d = %d (%lu/%lu  |  %.2f%%)\n", z, y_val, z_val, count, n_shots, (float) 100 * count / n_shots);
      if (ENABLE_DEBUG) {
        printf("  Full result: %s\n", result.c_str());
        printf("  y: %d (%s)\n", y_val, binStr(y_val, nbits_y).c_str());
        printf("  z+y: %d (%s)\n", z_val, binStr(z_val, nbits_z).c_str());
      }
    }
    if (z_val == 0 || y_val == 0) {
      i++;
      continue;
    }
    if (z_val == z+y) {
      total_correct += count;
    }
    i++;
  }
  if (n_printed < results.size()) {
    printf("More results hidden...\n");
  }
  // The percentage of results that were correct.
  printf("%lu / %lu Correct. (%.2f%%)\n", total_correct, n_shots, (float) 100 * total_correct / n_shots);
}

void runQFTAdder(long y, long z) {
  // PARSE INPUT VALUES
  // Values to add, optionally passed in cmdline
  // long y = 0b1000;
  // long z = 0b0100;
  // long c = 0b0010;

  // Necessary # bits computed based on input values. Min 1.
  int nbits_y = ceil(log2(max(std::vector<long>({y, z, 1})) + 1));
  int nbits_z = nbits_y + 1;

  printf("\nVERIFIED INPUTS\n");
  printf("y: %ld (%s)\n", y, binStr(y, nbits_y).c_str());
  printf("z: %ld (%s)\n", z, binStr(z, nbits_z).c_str());
  printf("\nEXPECTED VALUES\n");
  printf("z+y: %ld (%s)\n", z + y, binStr(z + y, nbits_z).c_str());
  printf("Adding values: %ld + %ld = %ld\n", z, y, z + y);
  printf("Expected Full Out: (%s%s)\n", binStr(y, nbits_y).c_str(),
         binStr(z + y, nbits_z).c_str());

  // GENERATE AND RUN CIRCUIT
  auto start = std::chrono::high_resolution_clock::now();

  int n_shots = NUMBER_OF_SHOTS; // Get a lot of samples
  auto counts = cudaq::sample(n_shots, run_adder{}, y, z, nbits_y, nbits_z);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\nAdder finished in %s.\n", formatTime(duration).c_str());

  std::vector<std::tuple<std::string, size_t>> results = sortMap(counts.to_map());
  printf("\nMEASURED RESULTS\n");
  displayFullResults(results, y, z, nbits_y, nbits_z, 10);

  // // REVIEW RESULTS
  // std::string result = counts.most_probable();
  // printf("Full out: (%s)\n", result.c_str());
  // std::string val2_out = result.substr(0, nbits_y);
  // std::string sum_out = result.substr(nbits_y, nbits_z);
  // printf("Sum: %d (%s)\n", binToInt(sum_out), sum_out.c_str());
}

/****************** CUDAQ STRUCTS ******************/
int main() {
  printf("This will attempt to use QFT to compute z + y\n");
  long z, y;
  printf("Enter y: ");
  std::cin >> y;
  printf("Enter z: ");
  std::cin >> z;
  runQFTAdder(y, z);
  printf("\n");
}
