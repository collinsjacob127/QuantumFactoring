# Simple script to generate example semiprimes written with ChatGPT

def generate_primes(n):
    """Return a list of all prime numbers up to n using the Sieve of Eratosthenes."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def main():
    limit = 100000
    # Generate primes up to limit, then remove 2.
    primes = generate_primes(limit)
    primes = [p for p in primes if p != 2]
    
    semiprimes = []
    
    # For each pair (p, q) with p <= q, both primes are now never 2.
    for i, p in enumerate(primes):
        if p * p >= limit:
            break
        for q in primes[i:]:
            product = p * q
            if product >= limit:
                break
            # Mark with an asterisk if the semiprime is a square (i.e. p == q)
            mark = (p == q)
            semiprimes.append((product, f"{product}{'*' if mark else ''}"))
    
    # Sort semiprimes by their numerical value.
    semiprimes.sort(key=lambda x: x[0])
    
    # Prepare output lines.
    output_lines = [s for _, s in semiprimes]
    
    # Save the semiprimes to the file.
    with open("example_semiprimes.txt", "w") as f:
        for line in output_lines:
            f.write(line + "\n")
    
    # Print the output.
    for line in output_lines:
        print(line)

if __name__ == "__main__":
    main()

