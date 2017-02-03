import theano
import theano.tensor as T
import numpy as np
import lasagne
import lasagne.layers as ll
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne import updates as lasagne_updates
from normalization import batch_norm
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.datasets import MNIST
from fuel.transformers import Cast, ScaleAndShift
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from parzen import get_parzen_estimator
import nn

srng = RandomStreams()
#### HYPER PARAMETERS ####
num_epochs = 100
num_steps = 5
infuse_rate = 0.0
infuse_rate_growth = 0.08
batch_size = 512
var_scale = 1
var_target_value = 1e-4
#var_target_value = 0.025

### Saves images
def print_samples(prediction, nepoch, batch_size, filename_dest):
    plt.figure()
    batch_size_sqrt = int(np.sqrt(batch_size))
    input_dim = prediction[0].shape[1]
    prediction = np.clip(prediction, 0, 1)
    pred = prediction.reshape((batch_size_sqrt, batch_size_sqrt, input_dim, input_dim))
    pred = pred.swapaxes(2, 1)
    pred = pred.reshape((batch_size_sqrt*input_dim, batch_size_sqrt*input_dim))
    fig, ax = plt.subplots(figsize=(batch_size_sqrt, batch_size_sqrt))
    ax.axis('off')
    ax.imshow(pred, cmap='Greys_r')
    ax.grid()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(filename_dest, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    plt.close()


#### DATASET ####
# Batch iterator over fuel dataset
def batch_iterator(dataset, batchsize, shuffle=False):
    if shuffle:
        train_scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=batchsize)
    else:
        train_scheme = SequentialScheme(examples=dataset.num_examples, batch_size=batchsize)
    stream = DataStream.default_stream(dataset=dataset, iteration_scheme=train_scheme)
    stream_scale = ScaleAndShift(stream, 1./255.0, 0, which_sources=('features',))
    stream_data = Cast(stream_scale, dtype=theano.config.floatX, which_sources=('features',))
    return stream_data.get_epoch_iterator()


# Get the train, valid and test set on mnist
def create_MNIST_data_streams():
    train_set = MNIST(('train',), subset=slice(0, 50000), sources=('features', 'targets'), load_in_memory=True)
    valid_set = MNIST(('train',), subset=slice(50000, 60000), sources=('features', 'targets'), load_in_memory=True)
    test_set = MNIST(('test',), sources=('features', 'targets'), load_in_memory=True)
    return train_set, valid_set, test_set


# Get gaussians probabilities distribution for each pixel over the train set
def Compute_probs_gaussian(dataset):
    X_train = dataset
    data = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2] * X_train.shape[3]))
    print data.max()
    data = data.swapaxes(0, 1)
    data = data.swapaxes(1, 2)
    mean = np.zeros((X_train.shape[1], X_train.shape[2] * X_train.shape[3]))
    var = np.zeros((X_train.shape[1], X_train.shape[2] * X_train.shape[3]))
    for k in range(X_train.shape[1]):
        for i in xrange(X_train.shape[2] * X_train.shape[3]):
            mean[k][i] = np.mean(data[k][i])
            var[k][i] = np.var(data[k][i])
    mean = mean.reshape((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    var = var.reshape((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    return mean, var


# Return probability over a dataset in gaussian case
def get_probabilites(X_train, batch_size):
    epsilon = 1e-5
    # Get probability distribution of the dataset
    print("Computing probability distribution...")
    data_train = np.concatenate(np.asarray([batch[0] for batch in batch_iterator(X_train, batch_size, shuffle=False)]))
    mu_prior, var_prior = Compute_probs_gaussian(data_train)
    mu_prior = np.repeat(np.expand_dims(mu_prior, axis=0), batch_size, axis=0)
    var_prior = np.repeat(np.expand_dims(var_prior, axis=0), batch_size, axis=0)
    mu_prior = np.cast[theano.config.floatX](mu_prior)
    var_prior = np.cast[theano.config.floatX](var_prior) + epsilon
    return mu_prior, var_prior


#### Build model with Lasagne ####
def build_MLP_mnist_bn():
    net = {}
    net['input'] = ll.InputLayer(shape=(None, 1, 28, 28), input_var=None)
    net['d0'] = batch_norm(ll.DenseLayer(net['input'], num_units=1200, nonlinearity=lasagne.nonlinearities.rectify), steps=num_steps)
    net['mu1'] = ll.DenseLayer(net['d0'], num_units=28*28, nonlinearity=lasagne.nonlinearities.sigmoid)
    net['var1'] = ll.DenseLayer(net['d0'], num_units=28*28, nonlinearity=lasagne.nonlinearities.sigmoid)
    net['mu'] = ll.ReshapeLayer(net['mu1'], (([0], 1, 28, 28)))
    net['var'] = ll.ReshapeLayer(net['var1'], (([0], 1, 28, 28)))
    return net

#### Create theano graph ####
# Sample function from normal distribution
def sample_normal(mu, var):
    return srng.normal(mu.shape, mu, T.sqrt(var))


# Sample function from a two gaussian mixtur with scalar coeeficient infuse_rate
def samples_mix(mu_model, var_model, mu_target, var_target, infuse_rate):
    # Get samples for both distribution
    sample_model = sample_normal(mu_model, var_model)
    sample_target = sample_normal(mu_target, var_target)
    # Get mask (We use the same mask on each channel)
    shape = (mu_target.shape[0], mu_target.shape[2], mu_target.shape[3])
    mask = srng.binomial(shape, p=infuse_rate, dtype=mu_target.dtype).dimshuffle(0, 'x', 1, 2)
    # Compute the samples from the mixture
    return mask * sample_target + (1 - mask) * sample_model, (1 - mask)


# Eval gaussian density for given mu, var and x
def eval_log_gaussian(mu, var, x):
    return - 0.5 * (T.log(2.*np.pi) + T.log(var)) - (((x - mu)**2) / (2 * var))


# Eval log add exponential
def log_add_exp(la, lb):
    return la + T.nnet.nnet.softplus(lb - la)


# Compute the first term \log(\frac{p(z^(0))}{q(z^(0))})
def get_first_term(mu_prior, var_prior, mu_target, var_target, infuse_rate):
    # Sample z^(0)
    z_T_first, new_mask = samples_mix(mu_prior, var_prior, mu_target, var_target, infuse_rate)
    # Compute p(z^(0))
    first_logp = eval_log_gaussian(mu_prior, var_prior, z_T_first)
    # Compute q(z^(0) | x)
    first_logq = log_add_exp(T.log(1 - infuse_rate) + first_logp, T.log(infuse_rate) + eval_log_gaussian(mu_target, var_target, z_T_first))
    return first_logp, first_logq, z_T_first, new_mask


# Compute a step of training
def compute_step_train(network, z_T, infuse_rate, mu_target, var_target, det, valid=False, coeff_scale_var=1, t=0, epsilon=1e-4):
    z_T = theano.gradient.disconnected_grad(z_T)
    # Get the outputs of the network according to z_T
    if valid:
        outputs = ll.get_output([network['mu'], network['var']], z_T, deterministic=det, steps=t, batch_norm_update_averages=True, batch_norm_use_averages=False)
    elif not valid and det:
        outputs = ll.get_output([network['mu'], network['var']], z_T, deterministic=det, steps=t, batch_norm_use_averages=True)
    else:
        outputs = ll.get_output([network['mu'], network['var']], z_T, deterministic=det, steps=t, batch_norm_update_averages=False)
    # Get the mean
    mu_model = outputs[0]
    # Get the variance
    var_model = outputs[1] * coeff_scale_var + epsilon
    if t < (num_steps - 1):
        # Samples to get the next z^t
        z_t, new_mask = samples_mix(mu_model, var_model, mu_target, var_target, infuse_rate)
        # Compute log(p(z^(t) | z^(t-1))))
        log_p = eval_log_gaussian(mu_model, var_model, z_t)
        # Compute log(q(z^(t) | z^(t-1)))
        log_q = log_add_exp(T.log(1. - infuse_rate) + log_p, T.log(infuse_rate) + eval_log_gaussian(mu_target, var_target, z_t))
    else:
        z_t, new_mask = samples_mix(mu_model, var_model, mu_target, var_target, 0.0)
        log_p = T.constant(0)
        log_q = T.constant(0)
    # Compute \log(p(x | z^(t-1)))
    p_x0_x1 = eval_log_gaussian(mu_model, var_model, mu_target)
    return z_t, infuse_rate, log_p, log_q, p_x0_x1, new_mask, mu_model, var_model


# Run chain on training
def run_chaine_train_for(network, num_steps, mu_prior, var_prior, mu_target, var_target, infuse_rate, infuse_rate_growth, coeff_scale_var, valid, det=False):
    # Get the first term
    first_logp, first_logq, z_T, new_mask = get_first_term(mu_prior, var_prior, mu_target, var_target, infuse_rate)
    log_p_sum = first_logp
    log_q_sum = first_logq
    # Compute p(x | z^0) (Doesn't depend of the model parameters)
    log_pzx_sum = eval_log_gaussian(mu_prior, var_prior, mu_target)
    # Run the chain
    for i in range(num_steps):
        z_T, noise_level_args, log_p, log_q, p_x0_x1, new_mask, mu_model, var_model = compute_step_train(network, z_T, infuse_rate, mu_target, var_target, det, valid, coeff_scale_var, i)
        # Compute \sum_{t=1}^(T-1) log(p(z^(t) | z^(t-1)))
        log_p_sum = log_p + log_p_sum
        # Compute \sum_{t=1}^(T-1) log(q(z^(t) | z^(t-1)))
        log_q_sum = log_q + log_q_sum
        # Compute \sum_{t=1}^(T-1) \frac{t}{T} log(p(x | z^(t-1)))
        log_pzx_sum = log_pzx_sum + (p_x0_x1 * (i+1) / num_steps)
        # Increase infusion rate
        if infuse_rate_growth:
            infuse_rate += infuse_rate_growth
    return z_T, log_p_sum, log_q_sum, p_x0_x1, log_pzx_sum


# Compute a step of sampling
def compute_step_samples(network, z_T, coeff_scale_var=1, t=0, epsilon=1e-5):
    outputs = ll.get_output([network['mu'], network['var']], z_T, deterministic=True, steps=t)#, batch_norm_use_averages=False)
    # Get the mean
    mu_model = outputs[0]
    # Get the variance
    var_model = outputs[1] * coeff_scale_var + epsilon
    # Samples to get x^t-1
    new_z_T = sample_normal(mu_model, var_model)
    return new_z_T, mu_model, var_model


## Get samples ##
def run_chaine_samples_for(network, num_steps, mu_prior, var_prior, coeff_scale_var, more=0):
    # Sample from prior distribution
    z_T_samples = sample_normal(mu_prior, var_prior)
    list_samples_chain = []
    # Run the chain
    for i in range(num_steps):
        z_T_samples, mu_model_samples, var_model_samples = compute_step_samples(network, z_T_samples, coeff_scale_var, i)
        list_samples_chain = list_samples_chain + [z_T_samples]
    for k in range(more):
        z_T_samples, mu_model_samples, var_model_samples = compute_step_samples(network, z_T_samples, coeff_scale_var, i)
        list_samples_chain = list_samples_chain + [z_T_samples]
    samples_chain = T.stack(list_samples_chain)
    return z_T_samples, samples_chain, mu_model_samples, var_model_samples


# Define theano symbolic variables
mu_target = T.tensor4('mu_target')
var_target = T.tensor4('var_target')
mu_prior = T.tensor4('mu_prior')
var_prior = T.tensor4('var_prior')

# Build network
network = build_MLP_mnist_bn()
### Training phase ###
z_T, log_p_sum, log_q_sum, p_x0_x1, log_pzx_sum = run_chaine_train_for(network, num_steps, mu_prior, var_prior, mu_target, var_target, infuse_rate, infuse_rate_growth, var_scale, valid=False)
# Compute lower_bound
lower_bound = (p_x0_x1 + log_p_sum - log_q_sum).sum(axis=(3,2,1)).mean()
# Select loss
loss_train = - log_pzx_sum.mean()
# loss_train = - lower_bound.mean()
# Select parameters of the model
gen_params = ll.get_all_params([network['mu'], network['var']], trainable=True)
# Select updates
updates = lasagne_updates.adam(loss_train, gen_params)

### Valid phase Norm ###
x_T_norm, log_p_sum_norm, log_q_sum_norm, p_x0_x1_norm, log_pzx_sum_norm = run_chaine_train_for(network, num_steps, mu_prior, var_prior, mu_target, var_target, infuse_rate, infuse_rate_growth, var_scale, valid=True, det=False)
# Compute lower_bound
lower_bound_norm = (p_x0_x1_norm + log_p_sum_norm - log_q_sum_norm).sum(axis=(3,2,1)).mean()

### Valid phase ###
z_T_val, log_p_sum_val, log_q_sum_val, p_x0_x1_val, log_pzx_sum_val = run_chaine_train_for(network, num_steps, mu_prior, var_prior, mu_target, var_target, infuse_rate, infuse_rate_growth, var_scale, valid=False, det=False)
# Compute lower_bound
lower_bound_val = (p_x0_x1_val + log_p_sum_val - log_q_sum_val).sum(axis=(3,2,1)).mean()

# Det valid phase
x_T_val_det, log_p_sum_val_det, log_q_sum_val_det, p_x0_x1_val_det, log_pzx_sum_val_det = run_chaine_train_for(network, num_steps, mu_prior, var_prior, mu_target, var_target, infuse_rate, infuse_rate_growth, var_scale, valid=False, det=True)
lower_bound_val_det = (p_x0_x1_val_det + log_p_sum_val_det - log_q_sum_val_det).sum(axis=(3,2,1)).mean()

### Samples phase ###
z_T_samples, samples_chain, mu_model_samples, var_model_samples = run_chaine_samples_for(network, num_steps, mu_prior, var_prior, var_scale)
# Theano functions
train_batch = theano.function(inputs=[mu_prior, var_prior, mu_target, var_target], outputs=[z_T, loss_train, lower_bound], updates=updates, allow_input_downcast=True, on_unused_input='ignore')
val_batch_norm = theano.function(inputs=[mu_prior, var_prior, mu_target, var_target], outputs=[x_T_norm, lower_bound_norm], allow_input_downcast=True, on_unused_input='ignore')
val_batch_det = theano.function(inputs=[mu_prior, var_prior, mu_target, var_target], outputs=[x_T_val_det, lower_bound_val_det], allow_input_downcast=True, on_unused_input='ignore')
val_batch = theano.function(inputs=[mu_prior, var_prior, mu_target, var_target], outputs=[z_T_val, lower_bound_val], allow_input_downcast=True, on_unused_input='ignore')
fn_sample = theano.function(inputs=[mu_prior, var_prior], outputs=[z_T_samples, samples_chain, mu_model_samples, var_model_samples], allow_input_downcast=True, on_unused_input='ignore')

# Get data
X_train, X_val, X_test = create_MNIST_data_streams()

list_cost = []
list_train = []
list_val = []
# Get probability distribution of the train set
mu_prior, var_prior = get_probabilites(X_train, batch_size)
#d_norm = ll.get_all_params(network['d1'])
# Main Loop
for epoch in range(num_epochs):
    begin = time.time()
    loss_total, loss_total_train = 0., 0.
    train_batches = 0
    sigma_target = np.zeros((mu_prior.shape)) + var_target_value
    # Go through all mini batch of the training set
    for batch in batch_iterator(X_train, batch_size, shuffle=True):
        # Get batch of training data
        mu_target = batch[0]
        # Get current batch size
        curr_size = batch[0].shape[0]
        # Resize to the current batch size
        sigma_t = sigma_target[0:curr_size]
        mu = mu_prior[0:curr_size]
        sigma = var_prior[0:curr_size]
        # Call theano training function
        x_T, loss_train, lower_bound = train_batch(mu, sigma, mu_target, sigma_t)
        train_batches += 1
        loss_total += lower_bound
        loss_total_train += loss_train
    # Compute the mean loss over all batches
    loss_total /= train_batches
    loss_total_train /= train_batches
    list_cost.append(loss_total_train)
    list_train.append(loss_total)
    #print d_norm[3].get_value()
    # Compute Lower Bound on validation set
    val_batches = 0
    loss_total_val = 0
    for batch in batch_iterator(X_val, batch_size, shuffle=True):
        # Get batch of valid data
        mu_target = batch[0]
        # Get current batch size
        curr_size = batch[0].shape[0]
        # Resize to the current batch size
        sigma_t = sigma_target[0:curr_size]
        mu = mu_prior[0:curr_size]
        sigma = var_prior[0:curr_size]
        # Call theano validation function
        x_T, lower_bound = val_batch(mu, sigma, mu_target, sigma_t)
        val_batches += 1
        loss_total_val += np.mean(lower_bound)
    loss_total_val /= val_batches
    list_val.append(loss_total_val)
    # Print training infos
    print("Iteration %d, time = %ds, loss_train = %.4f, bound_train = %.4f, bound_val = %.4f" % (epoch, time.time()-begin, loss_total_train, loss_total, loss_total_val))

#np.savetxt('dump_bounds.txt', np.transpose([list_cost, list_train, list_val]))
loss_total, loss_total_train = 0., 0.
train_batches = 0

# Go through all mini batch of the training set
d_norm = ll.get_all_params(network['d0'])
print d_norm
print d_norm[3].get_value()
print d_norm[4].get_value()
print d_norm[11].get_value()
for batch in batch_iterator(X_train, batch_size, shuffle=True):
    # Get batch of training data
    mu_target = batch[0]
    # Get current batch size
    curr_size = batch[0].shape[0]
    # Resize to the current batch size
    sigma_t = sigma_target[0:curr_size]
    mu = mu_prior[0:curr_size]
    sigma = var_prior[0:curr_size]
    # Call theano training function
    x_T, lower_bound = val_batch_norm(mu, sigma, mu_target, sigma_t)
    train_batches += 1
    loss_total += lower_bound
print loss_total / train_batches
print d_norm[3].get_value()
print d_norm[4].get_value()
print d_norm[11].get_value()
# Compute Lower Bound on test set
test_batches = 0
loss_total_test = 0
for batch in batch_iterator(X_test, batch_size, shuffle=True):
    # Get batch of valid data
    mu_target = batch[0] + np.random.uniform(0, 1/255.0, batch[0].shape)
    # Get current batch size
    curr_size = batch[0].shape[0]
    # Resize to the current batch size
    sigma_t = sigma_target[0:curr_size]
    mu = mu_prior[0:curr_size]
    sigma = var_prior[0:curr_size]
    # Call theano validation function
    x_T, lower_bound = val_batch_det(mu, sigma, mu_target, sigma_t)
    test_batches += 1
    loss_total_test += np.mean(lower_bound)
loss_total_test /= test_batches

# Get parzen estimator over 10000 samples
data_test = np.concatenate(np.asarray([batch[0] for batch in batch_iterator(X_test, batch_size, shuffle=False)]))
for i in range(20):
    x_T, samples_chain, mu_prediction, var_prediction = fn_sample(mu_prior, var_prior)
    if i == 0:
        samples_mu = mu_prediction
    else:
        samples_mu = np.concatenate((samples_mu, mu_prediction))
parzen, se = get_parzen_estimator(data_test, samples_mu, "mnist")
print "Lower bound on test set: %.4f" % (loss_total_test)
print "Parzen on test set: %.4f" % (parzen.mean())
# Save some samples and the mean
x_T, samples_chain, mu_pred, var_pred = fn_sample(mu_prior[0:144], var_prior[0:144])
print_samples(x_T, epoch, 144, 'samples_more.png')
print_samples(mu_pred, epoch, 144, 'mean_more.png')
"""
for means in d_norm:
    if means.name == "mean_" + str(num_steps-1):
        tmp_mean = means.get_value()
        print tmp_mean
for sigmas in d_norm:
    if sigmas.name == "inv_std_" + str(num_steps-1):
        tmp_std = sigmas.get_value()
        print tmp_std
for means in d_norm:
    if means.name == "mean_" + str(i):
        means.set_value(tmp_mean)
for sigmas in d_norm:
    if sigmas.name == "inv_std_" + str(i):
        sigmas.set_value(tmp_std)
"""
#x_T, samples_chain, mu_pred, var_pred = fn_sample(mu_pred, var_pred)
#print_samples(mu_pred, epoch, 144, 'mean2.png')
x_T, samples_chain, mu_pred, var_pred = fn_sample(mu_prior, var_prior)
print_samples(mu_pred[0:144], epoch, 144, 'mean3_more.png')
x_T, samples_chain, mu_pred, var_pred = fn_sample(mu_prior[0:256], var_prior[0:256])
print_samples(mu_pred[0:144], epoch, 144, 'mean4_more.png')
