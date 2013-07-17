import numpy as np
from bart import _george

x = 10 * np.random.rand(50)
y = np.sin(x) + 0.1 * np.random.randn(len(x))
yerr = 0.1 * np.ones_like(y)

p1, p2 = 100.0, 5.0
eps = 1e-6

ll0, g0 = _george.gradlnlikelihood(x, y, yerr, p1, p2)
llp1, g1 = _george.gradlnlikelihood(x, y, yerr, p1 + eps, p2)
llm1, g1 = _george.gradlnlikelihood(x, y, yerr, p1 - eps, p2)
print np.abs(g0[0] - 0.5 * (llp1 - llm1) / eps)

llp1, g1 = _george.gradlnlikelihood(x, y, yerr, p1, p2 + eps)
llm1, g1 = _george.gradlnlikelihood(x, y, yerr, p1, p2 - eps)
print np.abs(g0[1] - 0.5 * (llp1 - llm1) / eps)
