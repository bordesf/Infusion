##### PARZEN CODE GAN
#### https://raw.githubusercontent.com/goodfeli/adversarial/master/parzen_ll.py
import theano.tensor as T
import theano as th
import numpy as np
import time


def get_parzen_estimator(X_data, samples_mu, dataset, cross_validate_parzen=False, printing=False):
    # Resize arrays
    samples_mu = samples_mu.reshape((samples_mu.shape[0], np.prod(samples_mu.shape[1:])))
    data_test = X_data.reshape((X_data.shape[0], np.prod(X_data.shape[1:])))
    # Find sigma
    if cross_validate_parzen:
        print "Cross validation sigma: Start"
        sigma_start = -1
        sigma_end = 0
        cross_val = 10
        sigma_range = np.logspace(sigma_start, sigma_end, num=cross_val)
        sigma_parzen = cross_validate_sigma(samples_mu, data_test, sigma_range, 25)
        print "Cross validation sigma: End"
    else:
        sigma_parzen = 0.1 if dataset == "TFD" else 0.17
    if printing:
        print "Using Sigma: {}".format(sigma_parzen)
    # Take 10 000 examples
    samples_mu = samples_mu[0:10000]
    parzen = theano_parzen(samples_mu, sigma_parzen)
    ll = get_nll(data_test, parzen, batch_size=25)
    se = ll.std() / np.sqrt(data_test.shape[0])
    return ll.mean(), se


def get_nll(x, parzen, batch_size=10):
    inds = range(x.shape[0])
    n_batches = int(np.ceil(float(len(inds)) / batch_size))
    times = []
    nlls = []
    for i in range(n_batches):
        begin = time.time()
        nll = parzen(x[inds[i::n_batches]])
        end = time.time()
        times.append(end-begin)
        nlls.extend(nll)
        # if i % 10 == 0:
        #    print i, np.mean(times), np.mean(nlls)

    return np.array(nlls)


def log_mean_exp(a):
    max_ = a.max(1)
    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def theano_parzen(mu, sigma):
    x = T.matrix()
    mu = th.shared(mu)
    a = (x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1)) / sigma
    E = log_mean_exp(-0.5*(a**2).sum(2))
    Z = mu.shape[1] * T.log(sigma * np.sqrt(np.pi * 2))

    return th.function([x], E - Z)


def cross_validate_sigma(samples, data, sigmas, batch_size):
    lls = []
    for sigma in sigmas:
        print sigma
        parzen = theano_parzen(samples, sigma)
        tmp = get_nll(data, parzen, batch_size)
        lls.append(np.asarray(tmp).mean())
        del parzen

    ind = np.argmax(lls)
    return sigmas[ind]
