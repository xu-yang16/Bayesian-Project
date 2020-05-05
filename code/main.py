import numpy as np
from copy import deepcopy
from scipy.stats import multivariate_normal, invgamma, invwishart, norm, bernoulli
import pdb

# define hyper-parameters
P = 104
K = 2
R = 50
n1 = 150
n2 = 50

# generate data
mu01 = [0.8] * 4
mu02 = [-0.8] * 4
B1 = np.ones((2, 4)) * 0.8
B2 = np.ones((2, 4)) * 0.8
Sigma1 = np.ones((4, 4)) * 0.5
Sigma1 += np.diag(np.ones(4) * 0.5)
Sigma2 = deepcopy(Sigma1)
noise = multivariate_normal(np.zeros(100), np.ones((100, 100)) * 0.1 + np.diag(np.ones(100) * 0.9))

# bayesian reference
a0 = 3
ak = [3] * K
b0 = 0.1
bk = [0.1] * K
b0k = [0] * K
dk = [P + 3] * K
Q = 0.1 * np.diag(np.ones(P))
m_0k = [0] * K
h = 4
h1 = 1
e = -3
f = 0
wrk = np.array([np.ones(50) * 0.05] * K)


class MCMC_sampler:
    def __init__(self, X, Z, g):
        self.X = X  # N * P
        self.Z = Z  # N * R
        self.g = g  # N * 1
        self.gamma = bernoulli(0.05).rvs(P)  # P * 1
        self.delta = []
        for k in range(2):
            self.delta.append(np.apply_along_axis(lambda x: bernoulli(x).rvs(R), 0, wrk[k, :]))
        self.delta = np.array(self.delta)  # R * K
        self.m_0k = m_0k

    def init(self):
        self.sigma2 = np.apply_along_axis(
            lambda ab: invgamma(ab[0], scale=ab[1]).rvs(P),
            0, [[a0] + ak, [b0] + bk]
        ).squeeze()  # P * (K+1)

        self.beta = []
        self.Gamma_0 = []
        self.nu = []
        self.mu_0 = []
        for k in range(2):
            sigma2_k = self.sigma2[:, k + 1]  # P

            self.beta.append(np.apply_along_axis(lambda sigma: norm(b0k[k], sigma).rvs(R), 0, [sigma2_k]))
            Gamma_0k = invwishart(dk[k], Q).rvs(1)  # P * P
            self.Gamma_0.append(Gamma_0k)

            m_0k = np.ones(P) * self.m_0k[k]
            nu_k = multivariate_normal(m_0k, h1 * Gamma_0k).rvs(1).squeeze()  # P
            self.nu.append(nu_k)
            self.mu_0.append(multivariate_normal(nu_k, Gamma_0k).rvs(1))  # P

        self.beta = np.array(self.beta)  # K * R * P
        self.Gamma_0 = np.array(self.Gamma_0)  # K * P * P
        self.nu = np.array(self.nu)  # K * P
        self.mu_0 = np.array(self.mu_0)  # K * P

    def one_epoch(self):
        gamma_N = self.random_walk_gamma()
        self.gamma = self.M_H_gamma(gamma_N)

        delta_N = self.random_walk_delta()
        self.delta = self.M_H_delta(delta_N)

        mu_N = self.random_walk_mu_0()
        self.mu_0 = self.M_H_mu_0(mu_N)

    def random_walk_gamma(self):
        p1 = bernoulli(0.05).rvs(P)
        p1[np.where(self.gamma == 1)] *= -1
        gamma_N = self.gamma + p1
        return gamma_N

    def random_walk_delta(self):
        p1 = bernoulli(0.05).rvs(R)
        p1[np.where(self.delta == 1)] *= -1
        delta_N = self.delta + p1
        return delta_N

    def random_walk_mu_0(self):
        errors = norm(0, 0.1).rvs(K * P).reshape(2, -1)
        return self.mu_0 + errors

    def M_H_gamma(self, gamma_N):
        ratio = self.p_gamma(self.gamma) - self.p_gamma(gamma_N)
        if ratio >= 0:
            return self.gamma
        elif np.log(np.random.uniform()) < ratio:
            return self.gamma
        else:
            return gamma_N

    def M_H_delta(self, delta_N):
        ratio = self.p_delta(self.delta) - self.p_gamma(delta_N)
        if ratio >= 0:
            return self.delta
        elif np.log(np.random.uniform()) < ratio:
            return self.delta
        else:
            return delta_N

    def M_H_mu_0(self, mu_0_N):
        ratio = self.p_mu_0(self.mu_0) - self.p_mu_0(mu_0_N)
        if ratio >= 0:
            return self.mu_0
        elif np.log(np.random.uniform()) < ratio:
            return self.mu_0
        else:
            return mu_0_N

    def mu_k_gamma(self, k, gamma, delta, mu_0=None):
        if not mu_0:
            mu_0 = self.mu_0
        return mu_0[k, gamma == 1] + np.matmul(self.beta[delta == 1].T, self.Z[delta == 1])  # P_gamma

    def p_x_gamma(self, gamma, delta=None, mu_0=None):
        if not delta:
            delta = self.delta
        x_gamma = self.X[:, gamma == 1]

        def calc_p_k(k):
            Sigma_gamma = np.diag(self.sigma2[gamma == 1, k + 1])  # P_gamma * P_gamma
            return np.sum(np.log(
                np.apply_along_axis(
                    lambda x: multivariate_normal.pdf(x, self.mu_k_gamma(k, gamma, delta, mu_0), Sigma_gamma),
                    1, x_gamma[self.g == k]
                )
            ))

        return calc_p_k(0) + calc_p_k(1)

    def p_x_gamma_c(self, gamma):
        Omega_gamma_c = np.diag(self.sigma2[:, 0][gamma == 0])  # (P - P_gamma) * (P - P_gamma)
        x_gamma_c = self.X[:, gamma == 0]  # N * P_gamma
        return np.sum(np.log(
            np.apply_along_axis(
                lambda x: multivariate_normal.pdf(x, np.zeros_like(x), Omega_gamma_c),
                0, x_gamma_c
            )
        ))

    def p_mu_0kgamma(self, gamma, mu_0=None, k=None):
        if not mu_0:
            mu_0 = self.mu_0
        def calc_p_k(k):
            return np.log(multivariate_normal.pdf(mu_0[k, gamma == 1],
                                                  self.nu[gamma == 1, k],
                                                  h1 * self.Gamma_0[k, gamma == 1, gamma == 1]))

        if not k:
            return calc_p_k(0) + calc_p_k(1)
        else:
            return calc_p_k(k)

    def p_mu_0kgamma_c(self, gamma):
        def calc_p_gamma_c(k):
            return np.log(multivariate_normal.pdf(self.mu_0[k, gamma == 0],
                                                  np.zeros(sum(gamma == 0)),
                                                  self.Gamma_0[k, gamma == 0, gamma == 0]))

        return calc_p_gamma_c(0) + calc_p_gamma_c(1)

    def p_gamma(self, gamma):
        return self.p_x_gamma_c(gamma) + self.p_x_gamma(gamma) + self.p_mu_0kgamma(gamma) + \
               self.p_mu_0kgamma_c(gamma) + np.sum(np.log(np.apply_along_axis(bernoulli(0.05).pmf, 0, gamma)))

    def p_delta(self, delta):
        return self.p_x_gamma(self.gamma, delta) + np.sum(np.log(np.apply_along_axis(bernoulli(0.05).pmf, 0, delta)))

    def p_mu_0(self, mu_0, k):
        return self.p_x_gamma(self.gamma, self.delta, mu_0) + self.p_mu_0kgamma(self.gamma, mu_0, k)


if __name__ == "__main__":
    # generate data samples
    Z = multivariate_normal(np.zeros(R), np.diag(np.ones(R))).rvs(n1 + n2)  # R * N
    x1 = np.zeros((n1, P))
    x2 = np.zeros((n2, P))

    for i in range(n1):
        x1[i, :4] = multivariate_normal(mu01 + np.matmul(np.transpose(B1), Z[i, :2]), Sigma1).rvs(1)
        x1[i, 4:] = noise.rvs(1)

    for i in range(n2):
        x2[i, :4] = multivariate_normal(mu02 + np.matmul(np.transpose(B2), Z[i, :2]), Sigma2).rvs(1)
        x2[i, 4:] = noise.rvs(1)

    X = np.vstack((x1, x2))
    g = np.concatenate((np.zeros(n1), np.ones(n2)))

    # begin simulation
    agent = MCMC_sampler(X, Z, g)
    agent.init()

    for i in range(1000):
        agent.one_epoch()
