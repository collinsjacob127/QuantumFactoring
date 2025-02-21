/**********************************
 * Description: Using Grover's search to find integer
 *              components that multiply to a given product.
 * Author: Jacob Collins
 * Usage:
 *  ...
 **********************************/
#include <cudaq.h>

#include <chrono>
#include <cmath>
#include <numbers>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

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
std::vector<std::tuple<std::string, size_t>> sortMap(
    const std::unordered_map<std::string, size_t>& myMap) {
  // Create a vector of tuples from the unordered_map
  std::vector<std::tuple<std::string, size_t>> vec;
  for (const auto& pair : myMap) {
    vec.emplace_back(pair.first, pair.second);
  }

  // Sort the vector in descending order based on the size_t value
  std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
    return std::get<1>(a) > std::get<1>(b);  // Compare the size_t values
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
  for (auto& v : arr) {
    if (v > max) {
      max = v;
    }
  }
  return max;
}

// Convert bin string to int. 1101 -> 13
int binToInt(std::string& s, bool reverse = true) {
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
    h(ctrl);
    x(ctrl);
    x(tgt);
    z<cudaq::ctrl>(ctrl, tgt[0]);
    x(tgt);
    x(ctrl);
    h(ctrl);
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
        // phase = c * std::numbers::pi / pow(2, j);
        phase = c * std::numbers::pi / pow(2, j);
        // if (ENABLE_DEBUG) { printf("Phase: %lf\n  y_i: %d\n  z_i: %d\n  j:
        // %d\n", phase, y_ind, z_ind, j); }
        cr1(phase, y_reg[y_ind], z_reg[z_ind]);
        // r1<cudaq::ctrl>(phase, y_reg[y_ind], z_reg[z_ind]);
      }
    }
  }
};

struct Multiply {
  void operator()(cudaq::qview<> x_reg, cudaq::qview<> y_reg,
                  cudaq::qview<> z_reg) __qpu__ {
    ScaledAdder add_op;
    int nbits_x = x_reg.size();

    for (int i = 0; i < nbits_x; ++i) {
      cudaq::control(add_op, x_reg[i], y_reg, z_reg, pow(2, i));
    }
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
      if (target_bit_set) x(ctrl[ctrl.size() - i]);
    }
    // Mark if found
    x<cudaq::ctrl>(ctrl, tgt[0]);
    z(tgt[0]);
    x<cudaq::ctrl>(ctrl, tgt[0]);
    // Undefine good search state
    for (int i = 1; i <= ctrl.size(); ++i) {
      auto target_bit_set = (1 << (ctrl.size() - i)) & target_state;
      if (target_bit_set) x(ctrl[ctrl.size() - i]);
    }
  }
};

/********************************************
*************** DRIVER & MAIN ***************
*********************************************/

// Driver for adder inversion
struct QftMultiplication {
  __qpu__ auto operator()(const long N, int nbits_x, int nbits_y, int nbits_z,
                          long x, long y) {
    // 1. Compute Necessary Bits
    // Necessary # bits computed based on input values. Min 1.
    int nbits_vals = nbits_x + nbits_y;

    // 2. Initialize Registers
    cudaq::qvector v_reg( nbits_vals);  // Values reg. Both 'input' values are stored here.
    cudaq::qvector c_reg(nbits_z);  // Sum reg
    // cudaq::qvector tgt(1);
    Multiply mult_op;
    cudaq::qview x_reg = v_reg.front(nbits_x);
    cudaq::qview y_reg = v_reg.back(nbits_y);
    cudaq::qview z_reg = c_reg.front(nbits_z);

    // 3. Initialize products
    setInt(x, x_reg);
    setInt(y, y_reg);

    quantumFourierTransform(z_reg);
    mult_op(x_reg, y_reg, z_reg);
    cudaq::adjoint(quantumFourierTransform, z_reg);

    // 7. Measure
    mz(v_reg, c_reg);
  }
};

long calculateM(long N) {
  if (N % 2 == 0 || N % 3 == 0) {
    printf("Invalid N, trivial factor\n");
  }
  // S is -1 or 1
  int S = (int)1.5 - (0.5 * (N % 6));
  if (ENABLE_DEBUG) {
    printf("S: %d\n", S);
  }
  long M = (N - S) / 6 - 1;
  if (ENABLE_DEBUG) {
    printf("S: %ld\n", M);
  }
  return (int)M;
}

long displayFullResults(std::vector<std::tuple<std::string, size_t>> results,
                        long N, int nbits_x, int nbits_y, int nbits_z,
                        int n_printed = 5) {
  int n_shots = NUMBER_OF_SHOTS;
  int nbits_vals = nbits_x + nbits_y;
  long correct_result = -1;
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
    std::string z_out = result.substr(nbits_vals, nbits_z);
    int x_val = binToInt(x_out);
    int y_val = binToInt(y_out);
    int z_val = binToInt(z_out);
    // % of whole
    if (z_val % x_val == 0 && z_val % y_val == 0 && z_val == N) {
      total_correct += count;
      correct_result = x_val;
    }

    if (i < n_printed) {
      printf("%d * %d = %d (%.2f%%)\n", x_val, y_val, z_val,
             (float)100 * count / n_shots);
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
  printf("%lu / %d Correct. (%.2f%%)\n", total_correct, n_shots,
         (float)100 * total_correct / n_shots);
  return correct_result;
}

/**
 * @brief Function to use grover's oracle to calculate semiprimes
 */
long testMultiplication(long x, long y) {
  long N = x * y;
  if (N % 3 == 0 || N % 2 == 0) return -1;
  // Necessary # bits computed based on input. Min 1.
  int nbits_z = ceil(log2(max(std::vector<long>({N, 1}))));
  int nbits_y = ceil(log2(max(std::vector<long>({N / 3, 1}))));
  int nbits_x = nbits_y;
  int nbits_vals = nbits_x + nbits_y;
  int nbits_total = nbits_vals + nbits_z;
  printf("Finding factors of: %ld (%s)\n", N, binStr(N, nbits_z).c_str());
  printf("Using %d simulated qubits.\n", nbits_total);

  // GENERATE AND RUN CIRCUIT
  auto start = std::chrono::high_resolution_clock::now();  // Timer start

  int n_shots = NUMBER_OF_SHOTS;  // Get a lot of samples
  auto counts =
      cudaq::sample(n_shots, QftMultiplication{}, N, nbits_x, nbits_y, nbits_z, x, y);

  auto end = std::chrono::high_resolution_clock::now();  // Timer end
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Q-Alg finished in %s.\n", formatTime(duration).c_str());

  // REVIEW RESULTS
  // Counts are an unordered_map
  // Converting to sorted vector of tuples
  std::vector<std::tuple<std::string, size_t>> results =
      sortMap(counts.to_map());
  return displayFullResults(results, N, nbits_x, nbits_y, nbits_z);
}

int main() {
  long x = 0, y = 0;
  while (x < 3 || y < 3) {
    printf("Enter x:");
    std::cin >> x;
    printf("Enter y:");
    std::cin >> y;
  }

  testMultiplication(x, y);
  return 1;
}
