import pyDOE2
import numpy as np
from scipy.stats.distributions import norm
from scipy.stats.distributions import uniform


def central_biased_lhs_sample(parameters, bias, nb_sample):

    nb_variable = 0
    for one_variable in bias:
        if one_variable != 0:
            nb_variable += 1
    lhs = (pyDOE2.lhs(nb_variable, samples=nb_sample) - 0.5) * 2
    design = np.ndarray((nb_sample, np.size(parameters)), None)
    j = 0
    for i in range(np.size(parameters)):
        if bias[i] == 0:
            design[:, i] = parameters[i]
        else:
            design[:, i] = parameters[i] + lhs[:, j] * parameters[i] * bias[i]
            j += 1
    return design


def boundary_lhs_sample(parameters_lower_bound, parameters_upper_bound, nb_sample):
    nb_variable = len(parameters_lower_bound)
    lhs = pyDOE2.lhs(nb_variable, samples=nb_sample)
    design = np.ndarray((nb_sample, np.size(parameters_lower_bound)), None)
    for i in range(nb_sample):
        design[i] = parameters_lower_bound + lhs[i] * (parameters_upper_bound - parameters_lower_bound)
    return design


if __name__ == '__main__':
    design1 = np.array([33.01, 18.53, 17.56, 27.48, 0.3009, -0.9081, 0.27, 0.04616, 0.3944, 0.65, 0.3])
    design2 = np.array([33.01, 18.53, 17.56, 28.18, 0.3009, -0.9081, 0.27, 0.04616, 0.19, 0.65, 0.3])
    designs = boundary_lhs_sample(design2, design2, 100)
    print(designs)


