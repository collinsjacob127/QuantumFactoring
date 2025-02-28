/**********************************
 * Description: 
 *      Using grover's algorithm to invert QFT multiplication
 *      to achieve quantum SP factoring.
 * Note:
 *      Must be updated with methods from [this paper](https://arxiv.org/pdf/2312.10054)
 *      to reduce error and increase qubit efficiency.
 *      ^ IP in `factor.cpp`
 * Author: Jacob Collins
 **********************************/

#include <cudaq.h>
#include <cudaq/algorithms/draw.h>

#include <math.h>
#include <numbers>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define ENABLE_DEBUG false // Displays full bitwise output
#define ENABLE_CIRCUIT_FIG false
#define ENABLE_MISC_DEBUG false
#define ENABLE_STATEVECTOR false

#define NUMBER_OF_SHOTS 100
#define NUM_RESULTS_DISPLAYED 5

/**************************************************
******************* HELPER FUNCS ******************
***************************************************/

// Convert some value to a string
template <typename T>
std::string to_string(T value) {
  std::stringstream ss;
  ss << value;
  return ss.str();
}

// Convert nanoseconds to a string displaying subdivisions of time
std::string format_time(long long nanoseconds, bool sep=false) {
  long long milliseconds = nanoseconds / 1'000'000;  // From ns to ms
  long long seconds = milliseconds / 1'000;          // From ms to seconds
  milliseconds = milliseconds % 1'000;               // Remaining milliseconds

  long long remaining_nanoseconds =
      nanoseconds % 1'000'000;  // Remaining nanoseconds
  long long microseconds = remaining_nanoseconds / 1'000;  // From ns to µs
  remaining_nanoseconds = remaining_nanoseconds % 1'000;   // Remaining ns

  // Create the formatted string
  if (sep) {
    return to_string(seconds) + "s " + to_string(milliseconds) + "ms " +
            to_string(microseconds) + "µs " + to_string(remaining_nanoseconds) +
            "ns";
  } else {
    float s = (float) nanoseconds / 1'000'000'000;
    return to_string(s) + "s";
  }
}

// Convert an unordered_map to a sorted vector of tuples.
// ( The unordered map is the result of cudaq::sample() )
std::vector<std::tuple<std::string, size_t>> sort_map(const std::unordered_map<std::string, size_t>& myMap) {
    // Create a vector of tuples from the unordered_map
    std::vector<std::tuple<std::string, size_t>> vec;
    for (const auto& pair : myMap) {
        vec.emplace_back(pair.first, pair.second);
    }

    // Sort the vector in descending order based on the size_t value
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        return std::get<1>(a) > std::get<1>(b); // Compare the size_t values
    });

    return vec;
}

// Convert value to binary string
template <typename T>
std::string bin_str(T val, int nbits, bool qorder=false) {
  std::stringstream ss;
  if (qorder) {
    for (int i = nbits; i >= 1; --i) {
      // Shift through the bits in val
      auto target_bit_set = (1 << (nbits-i)) & val;
      // Add matching val to string
      if (target_bit_set) {
        ss << '1';
      } else {
        ss << '0';
      }
    }
    return ss.str();
  }
  for (int i = 1; i <= nbits; ++i) {
    // Shift through the bits in val
    auto target_bit_set = (1 << (nbits - i)) & val;
    // Add matching val to string
    if (target_bit_set) {
      ss << '1';
    } else {
      ss << '0';
    }
  }
  return ss.str();
}

// Return max value in an array
template <typename T>
T max(std::vector<T> arr) {
  T max = arr[0];
  for (auto &v : arr) {
    if (v > max) {
      max = v;
    }
  }
  return max;
}

// Convert bin string to int. 1101 -> 13
int bin_to_int(std::string &s, bool qorder=true) {
  int result = 0;
  int len = s.length();
  if (qorder) {
    for (int i = len-1; i >= 0; --i) {
      if (s[(len-1)-i] == '1') {
        result += pow(2, i);
      }
    }
    return result;
  }
  for (int i = 0; i < len; ++i) {
    if (s[i] == '1') {
      result += pow(2, i);
    }
  }
  return result;
}


/****************** CUDAQ FUNCS ******************/
// Apply NOT-gates in accordance with bit-pattern of given integer.
// Bit-order is reversed. i.e. 3 (00011) would be (11000)
__qpu__ void setInt(const long val, cudaq::qview<> qs, bool qorder=true) {
  if (qorder) {
    // Iterate through bits in val
    for (int i = 1; i <= qs.size(); ++i) {
      // Bit-shift for single bitwise AND to apply X on correct qubits
      auto target_bit_set = (1 << (qs.size() - i)) & val;
      if (target_bit_set) x(qs[i-1]);
    }
  } else {
    // Iterate through bits in val
    for (int i = 1; i <= qs.size(); ++i) {
      // Bit-shift for single bitwise AND to apply X on correct qubits
      auto target_bit_set = (1 << (qs.size() - i)) & val;
      if (target_bit_set) x(qs[qs.size() - i]);
    }
  }
}

// Converted from qml function
// https://docs.pennylane.ai/en/stable/_modules/pennylane/templates/subroutines/qft.html#QFT
// ~WORKS PERFECTLY~
__qpu__ void quantumFourierTransform(cudaq::qview<> qs) {
  const int nbits = qs.size();
  int shift_len = nbits-1;
  long double shifts[nbits-1];
  long double idx;
  // Calculate phase shifts 
  for (int i = 0; i < shift_len; ++i) {
    idx = i+2;
    shifts[i] = (long double) 2 * std::numbers::pi * pow(2, -idx); 
    if (ENABLE_MISC_DEBUG) { printf("QFT PHASE %d: %Lf\n", i, shifts[i]); }
  }
  // Apply phase shifts
  for (int i = 0; i < nbits; ++i) {
    h(qs[i]);
    for (int j = i + 1; j < nbits; ++j) {
      cr1(shifts[j-(i+1)], qs[j], qs[i]);
    }
  }
  // SWAP first half and reversed last half
  for (int i = 0; i < nbits / 2; ++i) {
    swap(qs[i], qs[(nbits-1)-i]);
  }
}

// Add an increment of k*pi/2^j across this register
__qpu__ void addKFourier(cudaq::qview<> qs, const int k) {
  const int nbits = qs.size();
  long double phase; 
  // Apply phase shifts
  for (int j = 0; j < nbits; ++j) {
    phase = (long double) k * std::numbers::pi / pow(2, j);
    rz(phase, qs[j]);
  }
}

__qpu__ void addRegScaled(cudaq::qview<> y_reg, cudaq::qview<> z_reg, const int c) {
  const int nbits_y = y_reg.size();
  int k; 
  // Add y
  for (int i = 0; i < nbits_y; ++i) {
    k = c * (pow(2, nbits_y - (i + 1)));
    cudaq::control(addKFourier, y_reg[i], z_reg, k);
  }
}

// Inversion about the mean
struct reflect_uniform {
  void operator()(cudaq::qview<> ctrl, cudaq::qview<> tgt) __qpu__ {
    h(ctrl);
    x(ctrl);
    x(tgt);
    z<cudaq::ctrl>(ctrl, tgt[0]);
    x(tgt);
    x(ctrl);
    h(ctrl);
  }
};

/**
 * @brief Grover's oracle to search for target_state
 * @param ctrl - Register to search.
 * @param tgt - Qubit on which to apply Toffoli and z-gates.
 */
struct oracle {
  const long target_state;

  void operator()(cudaq::qview<> ctrl, cudaq::qview<> tgt) __qpu__ {
    // Define good search state (secret)
    for (int i = 1; i <= ctrl.size(); ++i) {
      auto target_bit_set = (1 << (ctrl.size() - i)) & target_state;
      if (!target_bit_set) x(ctrl[i - 1]);
    }
    // Mark if found
    x<cudaq::ctrl>(ctrl, tgt[0]);
    z(tgt[0]);
    x<cudaq::ctrl>(ctrl, tgt[0]);
    // Undefine good search state
    for (int i = 1; i <= ctrl.size(); ++i) {
      auto target_bit_set = (1 << (ctrl.size() - i)) & target_state;
      if (!target_bit_set) x(ctrl[i - 1]);
    }
  }
};

// Based on pennylane.ai implementation
struct QFTMult {
  void operator()(cudaq::qview<> x_reg, cudaq::qview<> y_reg, cudaq::qview<> z_reg) __qpu__ {
    const int nbits_x = x_reg.size();
    // const int nbits_z = z_reg.size();
    int c;

    quantumFourierTransform(z_reg);

    // Add y repeatedly, scaled by powers of 2 per bit in x
    for (int i = 0; i < nbits_x; ++i) {
        c = (pow(2, nbits_x - (i + 1)));
        cudaq::control(addRegScaled, x_reg[i], y_reg, z_reg, c);
    }

    cudaq::adjoint(quantumFourierTransform, z_reg);
  }
};

/****************** CUDAQ STRUCTS ******************/
struct runFactorization {
  __qpu__ auto operator()(const long semiprime,
                          const int nbits_x, const int nbits_y, const int nbits_z) {
    // 1. Initialize Registers
    QFTMult mult_op;
    reflect_uniform diffuse_op{};
    oracle oracle_op{.target_state = semiprime};
    cudaq::qvector q_reg(nbits_x + nbits_y + nbits_z);  // Value 1 reg
    cudaq::qview x_reg = q_reg.front(nbits_x);
    cudaq::qview y_reg = q_reg.slice(nbits_x, nbits_y);
    cudaq::qview z_reg = q_reg.back(nbits_z);  // Value 2 reg
    cudaq::qvector tgt(1);
    h(x_reg);
    h(y_reg);

    // ( pi / 4 ) * sqrt( N / k )
    // N: Size of search space (2^n choose 2)
    // k: Number of valid matching entries (assumed 2 for SP: (p1,p2), (p2,p1), (1,sp), (sp,1))
    // int n_iter = (0.785398) * sqrt(pow(2, nbits_x) * (pow(2, nbits_x)) / 4);
    // int n_iter = (std::numbers::pi/4) * sqrt(pow(2, (nbits_z)/2));
    // int n_iter = (std::numbers::pi/4) * sqrt(pow(2, (nbits_x+nbits_y)/2));
    // int n_iter = (std::numbers::pi/4) * sqrt(pow(2, (nbits_z)/2));
    int n_iter = (std::numbers::pi/4) * pow(2, (nbits_y + nbits_x)/2) / sqrt(2);
    for (int i = 0; i < n_iter; i++) {
      // 2. Multiply
      mult_op(x_reg, y_reg, z_reg);

      // 3. Grover's Oracle
      oracle_op(z_reg, tgt.front(1));

      // 4. Undo mult
      cudaq::adjoint(mult_op, x_reg, y_reg, z_reg);

      // 5. Diffusion to maximize probability
      diffuse_op(q_reg.front(nbits_x + nbits_y), tgt.front(1));
    }

    // 2. Mult
    mult_op(x_reg, y_reg, z_reg);

    // 3. Measure
    mz(q_reg);
  }
};

void display_full_results(std::vector<std::tuple<std::string, size_t>> results, long z, int nbits_x, int nbits_y, int nbits_z, size_t n_printed=5) {
  size_t n_shots = NUMBER_OF_SHOTS;
  size_t total_correct = 0;
  int i = 0;
  for (auto item : results) {
    // Binary result string
    std::string result = std::get<0>(item);
    // Count of this outcome being measured
    size_t count = std::get<1>(item);
    // Parse
    std::string x_out = result.substr(0, nbits_x);
    std::string y_out = result.substr(nbits_x, nbits_y);
    std::string z_out = result.substr(nbits_y+nbits_x, nbits_z);
    int x_val = bin_to_int(x_out);
    int y_val = bin_to_int(y_out);
    int z_val = bin_to_int(z_out);
    // % of whole
    if (i < n_printed) {
      if (z_val == x_val*y_val && z_val == z) {
        printf("%d * %d = %d (%lu/%lu = %.2f%%) ✓ \n", x_val, y_val, z_val, count, n_shots, (float) 100 * count / n_shots);
      } else {
        printf("%d * %d != %d (%lu/%lu = %.2f%%) X\n", x_val, y_val, z_val, count, n_shots, (float) 100 * count / n_shots);
      }
      if (ENABLE_DEBUG) {
        printf("  Full result: %s_%s_%s\n", x_out.c_str(), y_out.c_str(), z_out.c_str());
        printf("  (R1) x: %d (%s)\n", x_val, x_out.c_str());
        printf("  (R2) y: %d (%s)\n", y_val, y_out.c_str());
        printf("  (R3) N: %d (%s)\n", z_val, z_out.c_str());
      }
    }
    if (z_val == x_val*y_val && z_val == z) {
      total_correct += count;
    }
    i++;
  }
  if (n_printed < results.size()) {
    printf("More results hidden...\n");
  }
  // The percentage of results that were correct.
  printf("%lu / %lu Shots Correct. (%.2f%%)\n", total_correct, n_shots, (float) 100 * total_correct / n_shots);
}

// Return the minimum # bits needed to represent some number
int min_bits(long x) {
    return ceil(log2(max(std::vector<long>({x, 1})) + 1));
}

void run_SP_factor(long z) {
  // Necessary # bits computed based on input values. 
  int nbits_z = 2*min_bits(z);
  int nbits_x = min_bits(sqrt(z)+1);
  int nbits_y = nbits_x;

//   int nbits_z = (int) (1.5 * min_bits(z));
//   int nbits_y = min_bits((z/3));
//   int nbits_y = nbits_x;
//   int nbits_x = min_bits(sqrt(z));

//   int nbits_x = min_bits(z)-1;
//   int nbits_y = nbits_x;
//   int nbits_z = 2*(nbits_x+1);

  printf("\nVERIFIED INPUTS\n");
  printf("N: %ld (%s)\n", z, bin_str(z, nbits_z).c_str());

  // Draw circuit and view statevector
  if (ENABLE_CIRCUIT_FIG)
    std::cout << cudaq::draw(runFactorization{}, z, nbits_x, nbits_y, nbits_z);
  if (ENABLE_STATEVECTOR) {
    auto state = cudaq::get_state(runFactorization{}, z, nbits_x, nbits_y, nbits_z);
    state.dump();
  }
  
  // GENERATE AND RUN CIRCUIT
  auto start = std::chrono::high_resolution_clock::now();

  int n_shots = NUMBER_OF_SHOTS; // Get a lot of samples
  auto counts = cudaq::sample(n_shots, runFactorization{}, z, nbits_x, nbits_y, nbits_z);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\nSP Factoring finished in %s.\n", format_time(duration).c_str());
  std::vector<std::tuple<std::string, size_t>> results = sort_map(counts.to_map());
  printf("\nMEASURED RESULTS\n");
  display_full_results(results, z, nbits_x, nbits_y, nbits_z, NUM_RESULTS_DISPLAYED);
  printf("\n");
}

/****************** CUDAQ STRUCTS ******************/
int main(int argc, char *argv[]) {
  // PARSE INPUT VALUES
  // Default search value
  printf("Usage: ./factor.x [N = Semiprime]\n");
  long z = 15;
  if (argc >= 2) {
    z = strtol(argv[1], nullptr, 10);
  }
  printf("Finding p1, p2 | p1 * p2 = N, N > 8.\n");

  if (z < 9 || z % 2 == 0) { printf("Invalid input for z\n"); }

  run_SP_factor(z);
}
