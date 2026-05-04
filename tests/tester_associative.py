import sys
import os
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from substrate.hdc import HDCSubstrate, AssociativeMemory
from substrate.mhn import ModernHopfieldNetwork

def calculate_normalized_hamming(v1, v2):
    """Calculates Normalized Hamming Distance."""
    return np.mean(v1 != v2)

def benchmark_noise_tolerance():
    print("=== Associative Substrate Benchmarking (HDC vs MHN) ===\n")
    
    dim = 1000
    num_patterns = 20
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Initialize networks
    hdc_sub = HDCSubstrate(dimension=dim)
    hdc_mem = AssociativeMemory(hdc_sub)
    mhn = ModernHopfieldNetwork(dimension=dim, beta=10.0)
    
    # Generate and store patterns
    patterns = {}
    for i in range(num_patterns):
        label = f"Pattern_{i}"
        vec = hdc_sub.random_hypervector() # Use bipolar for both for simplicity
        patterns[label] = vec
        hdc_mem.store(label, vec)
        mhn.store(label, vec)
        
    print(f"Stored {num_patterns} patterns of dimension {dim}.\n")
    print(f"{'Noise %':<10} | {'HDC Acc':<10} | {'MHN Acc':<10}")
    print("-" * 35)
    
    for noise in noise_levels:
        hdc_correct = 0
        mhn_correct = 0
        trials = 50
        
        for _ in range(trials):
            # Pick random pattern
            target_label = f"Pattern_{np.random.randint(num_patterns)}"
            original_vec = patterns[target_label]
            
            # Apply noise (bit flips)
            noisy_vec = original_vec.copy()
            flip_mask = np.random.random(dim) < noise
            noisy_vec[flip_mask] *= -1
            
            # HDC Search
            hdc_results = hdc_mem.search(noisy_vec, threshold=-1.0) # Take best result
            if hdc_results and hdc_results[0][0] == target_label:
                hdc_correct += 1
                
            # MHN Search
            mhn_match = mhn.find_match(noisy_vec)
            if mhn_match and mhn_match[0] == target_label:
                mhn_correct += 1
                
        print(f"{noise*100:<10.0f}% | {hdc_correct/trials:<10.2f} | {mhn_correct/trials:<10.2f}")

    # --- Semantic Consistency Verification ---
    print("\n--- Semantic Consistency (HDC Bind/Bundle) ---")
    vec_a = hdc_sub.random_hypervector()
    vec_b = hdc_sub.random_hypervector()
    bound = hdc_sub.bind(vec_a, vec_b)
    bundle = hdc_sub.bundle([vec_a, vec_b])
    
    print(f"Dist(A, B): {calculate_normalized_hamming(vec_a, vec_b):.4f} (Expected: ~0.5)")
    print(f"Dist(A, Bound): {calculate_normalized_hamming(vec_a, bound):.4f} (Expected: ~0.5 - Orthogonal)")
    print(f"Dist(A, Bundle): {calculate_normalized_hamming(vec_a, bundle):.4f} (Expected: <0.5 - Similar)")
    
    # --- Reconstruction Confidence ---
    print("\n--- Reconstruction Confidence (MHN Step-1) ---")
    query = patterns["Pattern_0"]
    retrieved = mhn.retrieve(query)
    confidence = np.dot(query, retrieved) / (np.linalg.norm(query) * np.linalg.norm(retrieved))
    print(f"Pattern_0 Reconstruction Confidence: {confidence:.4f}")

if __name__ == "__main__":
    benchmark_noise_tolerance()
