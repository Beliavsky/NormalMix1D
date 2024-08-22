import numpy as np
import matplotlib.pyplot as plt
from mix import simulate_normal_mix, fit_normal_mix

def main():
    nfits = 3
    niter_em = 30
    plot_data = False
    # True parameters
    p1_true = 0.3
    m1_true = 0.0
    m2_true = 5.0
    sd1_true = 1.0
    sd2_true = 1.5
    
    # Simulate data
    np.random.seed(42)
    nobs = 10**4
    print("#obs:", nobs, "\n#iter_em:", niter_em, end="\n\n")
    for _ in range(nfits):
        x = simulate_normal_mix(nobs, p1_true, m1_true, m2_true, sd1_true, sd2_true)

        if plot_data:
            # Plot the simulated data
            plt.hist(x, bins=30, density=True, alpha=0.6, color='g')
            plt.title("Histogram of Simulated Data")
            plt.show()

        # Initial guesses for the parameters
        p1_init = 0.5
        m1_init = 0.0 # -1.0
        m2_init = 0.0 # 6.0
        sd1_init = 1.0 # 2.0
        sd2_init = 3.0 # 2.0
        fmt_r = "%8.4f" 
        fmt_s = "%8s"
        # Fit the mixture model
        p1_fit, m1_fit, m2_fit, sd1_fit, sd2_fit = fit_normal_mix(x, p1_init,
            m1_init, m2_init, sd1_init, sd2_init, niter=niter_em)

        # Print the results
        print("".join(fmt_s%s for s in ["", "p1", "p2", "m1", "m2", "sd1", "sd2"]))
        print(fmt_s%"Guessed:", "".join([fmt_r%xx for xx in
            [p1_init, 1-p1_init, m1_init, m2_init, sd1_init, sd2_init]]))
        print(fmt_s%"True:", "".join([fmt_r%xx for xx in
            [p1_true, 1-p1_true, m1_true, m2_true, sd1_true, sd2_true]]))
        print(fmt_s%"Fitted:", "".join([fmt_r%xx for xx in
            [p1_fit, 1-p1_fit, m1_fit, m2_fit, sd1_fit, sd2_fit]]))
        print(fmt_s%"Diff:", "".join([fmt_r%xx for xx in
            [p1_fit-p1_true, (1-p1_fit)-(1-p1_true), m1_fit-m1_true, m2_fit-m2_true,
                sd1_fit-sd1_true, sd2_fit-sd2_true]]), end="\n\n")

        
if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print("time elapsed (s):", "%6.3f"%(time.time() - start))