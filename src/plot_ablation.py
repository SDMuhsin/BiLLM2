import matplotlib.pyplot as plt

# Plot 1: Tikhonov Regularization (varying lambda)
lambdas = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
perplexity_lambda = [54.66, 50.20, 63.54, 65.54, 66.40, 68.40, 69.26, 64.30]

plt.figure(figsize=(8, 6))
plt.plot(lambdas, perplexity_lambda, marker='o', linestyle='-')
plt.xscale('log')
plt.xlabel(r'$\lambda$', fontsize=14)
plt.ylabel('Perplexity', fontsize=14)
plt.title('Ablation Study: Tikhonov Regularization', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('./output/ab2.png')

# Plot 2: Correlation Damping (varying gamma)
gamma = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
perplexity_gamma = [61.73, 63.54, 57.30, 65.00, 72.67, 87.50, 100.64, 112.08, 160.69]

plt.figure(figsize=(8, 6))
plt.plot(gamma, perplexity_gamma, marker='o', linestyle='-')
plt.xlabel(r'$\gamma$', fontsize=14)
plt.ylabel('Perplexity', fontsize=14)
plt.title('Ablation Study: Correlation Damping', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('./output/ab1.png')

