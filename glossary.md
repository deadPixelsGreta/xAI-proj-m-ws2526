# Glossary

### Domain Generalization

In Domain Generalization, we are given $M$ training (source) domains $S_{train} = \\{S^i | 1, ..., M\\}$ where $S^i = \\{(x_j^i, y_j^i)\\}_{j=1}^{n_i}$ denotes the $i$-th domain.

The joint distributions between each pair of domains are different: $P_{XY}^i \neq P_{XY}^j$, $1 \leq i \neq j \leq M$.

The goal of domain generalization is to lean a robust and generalizable predictive function $h: ~ X \to Y$ from the $M$ training domains to achieve a minimum prediction error on an _unseen_ test domain $S_{test}$ (i.e., $S_{test}$ cannot be accessed in training and $P_{XY}^{test} \neq P_{XY}^i$ for $i \in \\{1, ..., M\\}$):

$$\min\limits_h \\mathbb{E}_{(\\mathbf{x},y)} \in S_{test} \[ \ell (h(\\mathbf{x}), y) \]$$

where $\mathbb{E}$ is the expectation and $\ell(\cdot, \cdot)$ is the loss function.

---

Paper: Generalizing to Unseen Domains: A Survey on Domain Generalization

Citation: Wang, J., Lan, C., Liu, C., Ouyang, Y., Qin, T., Lu, W., Chen, Y., Zeng, W., & Yu, P. S. (2023). Generalizing to Unseen Domains: A Survey on Domain Generalization. IEEE Transactions on Knowledge and Data Engineering, 35(8), 8052–8072. https://doi.org/10.1109/TKDE.2022.3178128
