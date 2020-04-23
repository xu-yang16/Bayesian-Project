# Bayesian-Project😕

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)


## 内容列表😮

- [当前问题](#当前问题)
- [内容](#内容)
- [使用许可](#使用许可)

## 当前问题
1. 不明白怎么对$j$求积。Appendix中MCMC第一步，第一项是对$j$求积，但是之前公式中是对$i$。（$i$是sample，$j$是feature）
$$
    \prod_{j:\gamma_j=0}p(x_{j(\gamma^c)}|\gamma^N)
$$
$$
    x_{i(\gamma^c)}|\cdot\sim N(0,\Omega_{(\gamma^c)})
$$

2. 两个$\Sigma_{k(\gamma)}$是一样的吗？
$$
    \beta_{rk(\gamma)}\sim (1-\delta_{rk})I_0(\beta_{rk(\gamma)})+\delta_{rk}N(b_{0k(\gamma)},h\Sigma_{k(\gamma)})
$$
$$
    x_{i(\gamma)}|g_i=k,\cdot\sim N(\mu_{k(\gamma)},\Sigma_{k(\gamma)})
$$

3. 找不到$\mu_{(\gamma^c)}$的定义。
$$
    p(\mu_{0jk(\gamma^c)}|\gamma^N)
$$


## 内容 🌝

* **Report**文件夹内是本次Project的报告文件，包含tex和pdf等。
* ***.py**文件是python代码。（未来可能迁移到R）

### 使用许可

[MIT](LICENSE) © YX & FLH