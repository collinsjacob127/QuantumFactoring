/**********************************
 * Description: Using Grover's search to find integer 
 *              components that multiply to a given product.
 * Author: Jacob Collins
 * Citations:
 * QFT Arithmetic: https://arxiv.org/pdf/1411.5949
 * QFT Mult Exponent Adder: https://arxiv.org/pdf/2309.10204
 * Quantum Factoring Via Grover's: https://arxiv.org/pdf/2312.10054
 **********************************/
#include <cudaq.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>

#define ENABLE_DEBUG true
#define NUMBER_OF_SHOTS 2000

/**************************************************
******************* HELPER FUNCS ******************
***************************************************/

// Convert some value to a string
template <typename T>
std::string toString(T value) {
  std::stringstream ss;
  ss << value;
  return ss.str();
}

// Convert nanoseconds to a string displaying subdivisions of time
std::string formatTime(long long nanoseconds) {
  long long milliseconds = nanoseconds / 1'000'000;  // From ns to ms
  long long seconds = milliseconds / 1'000;          // From ms to seconds
  milliseconds = milliseconds % 1'000;               // Remaining milliseconds

  long long remaining_nanoseconds =
      nanoseconds % 1'000'000;  // Remaining nanoseconds
  long long microseconds = remaining_nanoseconds / 1'000;  // From ns to µs
  remaining_nanoseconds = remaining_nanoseconds % 1'000;   // Remaining ns

  // Create the formatted string
  return toString(seconds) + "s " + toString(milliseconds) + "ms " +
         toString(microseconds) + "µs " + toString(remaining_nanoseconds) +
         "ns";
}

// Convert an unordered_map to a sorted vector of tuples.
// ( The unordered map is the result of cudaq::sample() )
std::vector<std::tuple<std::string, size_t>> sortMap(const std::unordered_map<std::string, size_t>& myMap) {
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
std::string binStr(T val, int nbits) {
  std::stringstream ss;
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
int binToInt(std::string &s, bool reverse = true) {
  int result = 0;
  int len = s.length();
  if (reverse) {
    std::reverse(s.begin(), s.end());
  }
  for (int i = len - 1; i >= 0; --i) {
    if (s[i] == '1') {
      result += pow(2, len - 1 - i);
    }
  }
  return result;
}

/**************************************************
******************** QUANTUM OPS ******************
***************************************************/

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

// Inversion about the mean
// TODO: Does this need to match the diagram? RY(pi/2)?
struct Diffusor {
  void operator()(cudaq::qview<> ctrl, cudaq::qview<> tgt) __qpu__ {
    for (int i = 0; i < ctrl.size(); ++i) {
      ry(std::numbers::pi / 2, ctrl[i]);
    }
    z<cudaq::ctrl>(ctrl, tgt[0]);
    for (int i = 0; i < ctrl.size(); ++i) {
      ry(-std::numbers::pi / 2, ctrl[i]);
    }
  }
};

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

struct ScaledAdder {
  void operator()(cudaq::qview<> y_reg, cudaq::qview<> z_reg, long c) __qpu__ {
    const int nbits_y = y_reg.size();
    const int nbits_z = z_reg.size();
    int j;
    double phase;
    for (int y_ind = 0; y_ind < nbits_y; ++y_ind) {
      for (int z_ind = y_ind; z_ind < nbits_z; ++z_ind) {
        j = z_ind - y_ind;
        // phase = pow((double) c * M_PI / 2, j);
        phase = c * std::numbers::pi / pow(2, j);
        // if (ENABLE_DEBUG) { printf("Phase: %lf\n  y_i: %d\n  z_i: %d\n  j: %d\n", phase, y_ind, z_ind, j); }
        cr1(phase, y_reg[y_ind], z_reg[z_ind]);
        // r1<cudaq::ctrl>(phase, y_reg[y_ind], z_reg[z_ind]);
      }
    }
  }
};

struct Multiply {
  void operator()(cudaq::qview<> x_reg, cudaq::qview<> y_reg, cudaq::qview<> z_reg) __qpu__ {
    ScaledAdder add_op;
    quantumFourierTransform(z_reg); 
    int nbits_x = x_reg.size();

    for (int i = 0; i < nbits_x; ++i) {
      cudaq::control(add_op, x_reg[i], y_reg, z_reg, pow(2, i));
    }
    cudaq::adjoint(quantumFourierTransform, z_reg);
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
      if (target_bit_set)
        x(ctrl[ctrl.size() - i]);
    }
    // Mark if found
    x<cudaq::ctrl>(ctrl, tgt[0]);
    z(tgt[0]);
    x<cudaq::ctrl>(ctrl, tgt[0]);
    // Undefine good search state
    for (int i = 1; i <= ctrl.size(); ++i) {
      auto target_bit_set = (1 << (ctrl.size() - i)) & target_state;
      if (target_bit_set)
        x(ctrl[ctrl.size() - i]);
    }
  }
};

// __qpu__ void zeroQubitReg(cudaq::qview<> qs) {
//   for (int i = 0; i < qs.size(); ++i) {
//     x<cudaq::ctrl>(qs[i], qs[i]);
//   }
// }

/********************************************
*************** DRIVER & MAIN ***************
*********************************************/

// Driver for adder inversion
struct QuantumFactorization {
  __qpu__ auto operator()(const long N, int nbits_x, int nbits_y, int nbits_z, int n_grov_iter) {
    // 1. Initialize Registers
    int nbits_vals = nbits_x + nbits_y;
    cudaq::qvector v_reg(nbits_vals);  // Values reg. Both 'input' values are stored here.
    cudaq::qvector c_reg(nbits_z);   // Sum reg
    cudaq::qvector tgt(1);
    Multiply mult_op;
    Diffusor diffuse_op{};
    oracle oracle_op{.target_state = N};
    cudaq::qview x_reg = v_reg.front(nbits_x);
    cudaq::qview y_reg = v_reg.back(nbits_y);
    cudaq::qview z_reg = c_reg.front(nbits_z);

    // 2. Put our values in superposition
    h(v_reg);

    // ( pi / 4 ) * sqrt( N / k )
    // N: Size of search space (2^n choose 2)
    // k: Number of valid matching entries, assumed biprime, 2
    int n_iter = n_grov_iter;
    for (int i = 0; i < n_iter; i++) {
      // if (ENABLE_DEBUG) { printf("  Grover's iter %d...\n", i); }
      // zeroQubitReg(z_reg);

      // 3. QFT Multiplication
      mult_op(x_reg, y_reg, z_reg);

      // 4. Use oracle to mark our search val
      oracle_op(z_reg, tgt.front(1));

      // 5. Undo our addition
      cudaq::adjoint(mult_op, x_reg, y_reg, z_reg);

      // 6. Inversion about the mean to find the right inputs
      diffuse_op(v_reg.front(v_reg.size()), tgt.front(1)); 
    }

    // 7. Measure
    // Sum is {c_reg[0], v_reg2}
    mz(v_reg, c_reg);
  }
};

long calculateM(long N) {
  if (N % 2 == 0 || N % 3 == 0) {
    printf("Invalid N, trivial factor\n");
  }
  // S is -1 or 1
  int S = (int) 1.5 - (0.5 * (N % 6));
  if (ENABLE_DEBUG) { printf("S: %d\n", S); }
  long M = (N - S)/6 - 1;
  if (ENABLE_DEBUG) { printf("S: %ld\n", M); }
  return (int) M;
}

long displayFullResults(std::vector<std::tuple<std::string, size_t>> results, long N, int nbits_x, int nbits_y, int nbits_z, int n_printed=5) {
  int n_shots = NUMBER_OF_SHOTS;
  int nbits_vals = nbits_x + nbits_y;
  long correct_result = 1;
  size_t total_correct = 0;
  int i = 0;
  for (auto item : results) {
    printf("i: %d\n", i);
    // Binary result string
    std::string result = std::get<0>(item);
    // Count of this outcome being measured
    size_t count = std::get<1>(item);
    // Parse
    std::string x_out = result.substr(0, nbits_x);
    std::string y_out = result.substr(nbits_x, nbits_y);
    std::string z_out = result.substr(nbits_vals, nbits_z);
    int x_val = binToInt(x_out);
    int y_val = binToInt(y_out);
    int z_val = binToInt(z_out);
    // % of whole
    if (N % x_val == 0) {
      total_correct += count;
      correct_result = x_val;
    } else if (N % y_val == 0) {
      total_correct += count;
      correct_result = y_val;
    }
    if (i < n_printed) {
      printf("%d * %d = %d (%.2f%%)\n", x_val, y_val, z_val, (float) 100 * count / n_shots);
      // printf("x_val: %d (%s)\n", binToInt(val1_out), val1_out.c_str());
      // printf("y_val: %d (%s)\n", binToInt(val2_out), val2_out.c_str());
      if (ENABLE_DEBUG) {
        printf("  Full result: %s\n", result.c_str());
        printf("  x: %d (%s)\n", x_val, binStr(x_val, nbits_x).c_str());
        printf("  y: %d (%s)\n", y_val, binStr(y_val, nbits_y).c_str());
        printf("  z: %d (%s)\n", z_val, binStr(z_val, nbits_z).c_str());
      }
    }
    i++;
  }
  if (n_printed < results.size()) {
    printf("More results hidden...\n");
  }
  // The percentage of results that were correct.
  printf("%lu / %d Correct. (%.2f%%)\n", total_correct, n_shots, (float) 100 * total_correct / n_shots);
  return correct_result;
}


/**
 * @brief Function to use grover's oracle to calculate semiprimes
 */
long calculateSemiPrimes(long N) {
  if (N % 3 == 0 || N % 2 == 0) return -1;
  // Necessary # bits computed based on input. Min 1.
  int nbits_z = ceil(log2(max(std::vector<long>({N, 1}))));
  int nbits_y = ceil(log2(max(std::vector<long>({N/3, 1}))));
  int nbits_x = nbits_y;
  int nbits_vals = nbits_x + nbits_y;
  int nbits_total = nbits_vals + nbits_z;
  printf("Finding factors of: %ld (%s)\n", N, binStr(N, nbits_z).c_str());
  printf("Using %d simulated qubits.\n", nbits_total);
  int n_grov_iter = floor((std::numbers::pi / 4) * pow(2, (nbits_x+nbits_y)/2));
  printf("Grover's requires %d iterations in this case.\n", n_grov_iter);

  // GENERATE AND RUN CIRCUIT
  auto start = std::chrono::high_resolution_clock::now(); // Timer start

  int n_shots = NUMBER_OF_SHOTS; // Get a lot of samples
  auto counts = cudaq::sample(n_shots, QuantumFactorization{}, N, nbits_x, nbits_y, nbits_z, n_grov_iter);

  auto end = std::chrono::high_resolution_clock::now(); // Timer end
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Q-Alg finished in %s.\n", formatTime(duration).c_str());

  // REVIEW RESULTS
  // Counts are an unordered_map
  // Converting to sorted vector of tuples
  std::vector<std::tuple<std::string, size_t>> results = sortMap(counts.to_map());
  return displayFullResults(results, N, nbits_x, nbits_y, nbits_z);
}

int main() {
  calculateSemiPrimes(35);
  return 1;
}
