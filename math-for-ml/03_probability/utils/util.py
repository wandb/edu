import scipy.integrate as integrate

def integrates_to_one(pdf):
    integral, err_bound = integrate.quad(pdf, 0, 1)
    return abs(1 - integral) <= err_bound
