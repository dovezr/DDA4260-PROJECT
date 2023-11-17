import numpy as np
import rbm
import projectLib as lib
import neighborhood as nb

training = lib.getTrainingData()
validation = lib.getValidationData()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

allUsersRatings = lib.getAllUsersRatings(trStats["u_users"], training)

K = 5

# SET PARAMETERS HERE!!!
# number of hidden units
F = 20
epochs = 40
gradientLearningRate = 0.1

# Initialise all our arrays
W = rbm.getInitialWeights(trStats["n_movies"], F, K)
grad = np.zeros(W.shape)
posprods = np.zeros(W.shape)
negprods = np.zeros(W.shape)

### hyper-parameter for extensions ###
# momentum
beta = 0.5
momentum = np.zeros([100, F, K])

# adaptive learning rate
size = 5
improve_threshold = 0.005
rate = 0.01



# early stopping
stop_threshold = 0.0005

# regularisation
reg = 0.01

# mini-batch
batch_size = 1

# bias
# W = np.random.normal(0, 0.1, (trStats["n_movies"], F, K))
# b_h = np.random.normal(0, 0.1, (F))
# b_v = np.random.normal(0, 0.1, (trStats["n_movies"], K))
# grad_bh = np.zeros(b_h.shape)
# grad_bv = np.zeros(b_v.shape)
# momentum_val_bh = np.zeros(b_h.shape)
# momentum_val_bv = np.zeros(b_v.shape)

training_loss_list = []
validation_loss_list = []
pre_rmse = 1
best_RMSE = float('inf')
best_weights = None

for epoch in range(1, epochs+1):
    # in each epoch, we'll visit all users in a random order
    visitingOrder = np.array(trStats["u_users"])
    np.random.shuffle(visitingOrder)
    # count = 0
    
    # Initialize batchGrad matrix
    batchGrad = np.zeros_like(grad)
    for batch_start in range(0, len(visitingOrder), batch_size):
        batch_end = batch_start + batch_size
        batch_users = visitingOrder[batch_start:batch_end]

        for user in batch_users:
            # Rest of the code remains the same
            # count += 1
            # get the ratings of that user
            ratingsForUser = allUsersRatings[user]
            # build the visible input
            v = rbm.getV(ratingsForUser)

            # get the weights associated to movies the user has seen
            weightsForUser = W[ratingsForUser[:, 0], :, :]

            ### LEARNING ###
            # propagate visible input to hidden units
            posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser)
            # get positive gradient
            # note that we only update the movies that this user has seen!
            posprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(v, posHiddenProb)

            ### UNLEARNING ###
            # sample from hidden distribution
            sampledHidden = rbm.sample(posHiddenProb)
            # propagate back to get "negative data"
            negData = rbm.hiddenToVisible(sampledHidden, weightsForUser)
            # propagate negative data to hidden units
            negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser)
            # get negative gradient
            # note that we only update the movies that this user has seen!
            negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(negData, negHiddenProb)
            # we average over the number of users in the batch (if we use mini-batch)
            # grad[ratingsForUser[:, 0], :, :] = gradientLearningRate/batch_size * \
            #                         (posprods[ratingsForUser[:, 0], :, :] - negprods[ratingsForUser[:, 0], :, :])
            
            # gradient descent for bias
            # grad_bv[ratingsForUser[:, 0], :] = v - negData
            # grad_bh = posHiddenProb - negHiddenProb

            # extension
            # momentum
            momentum[ratingsForUser[:, 0], :, :] = beta * momentum[ratingsForUser[:, 0], :, :] + \
                                    (posprods[ratingsForUser[:, 0], :, :] - negprods[ratingsForUser[:, 0], :, :])
            W[ratingsForUser[:, 0], :, :] += gradientLearningRate * (momentum[ratingsForUser[:, 0], :, :] / batch_size + \
                                                                     reg * W[ratingsForUser[:, 0], :, :])
            # momentum_val_bh = beta * momentum_val_bh + grad_bh
            # grad_bh = momentum_val_bh
            # momentum_val_bv[ratingsForUser[:, 0], :] = beta * momentum_val_bv[ratingsForUser[:, 0], :]
            # grad_bv[ratingsForUser[:, 0], :] = momentum_val_bv[ratingsForUser[:, 0], :]
            
            # regularisation
            # W[ratingsForUser[:, 0], :, :] -= gradientLearningRate * 

            # mini-batch
            # W[ratingsForUser[:, 0], :, :] += grad[ratingsForUser[:, 0], :, :] / batch_size
            # b_v[ratingsForUser[:, 0], :] += gradientLearningRate * grad_bv[ratingsForUser[:, 0], :] / batch_size
            # b_h += gradientLearningRate * grad_bh / batch_size
    # Print the current RMSE for training and validation sets
    # this allows you to control for overfitting e.g
    # We predict over the training set
    tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, allUsersRatings)
    # tr_r_hat = nb.predict_N(trStats, W, allUsersRatings)
    # tr_r_hat = nb.neighbourhood(training, tr_r_hat)
    trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

    # We predict over the validation set
    vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, allUsersRatings)
    # vl_r_hat = nb.predict_N(trStats, W, allUsersRatings)
    # vl_r_hat = nb.neighbourhood(validation, vl_r_hat)
    vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

    training_loss_list.append(trRMSE)
    validation_loss_list.append(vlRMSE)

    # if pre_rmse - vlRMSE <= stop_threshold:
    #     print("early stop")
    #     break

    # adaptive learning rate
    gradientLearningRate *= lib.adaptive_lr(validation_loss_list, size, improve_threshold, rate)

    print("### EPOCH %d ###" % epoch)
    print("Training loss = %f" % trRMSE)
    print("Validation loss = %f" % vlRMSE)

lib.plot_loss(training_loss_list, validation_loss_list)
### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
# predictedRatings = np.array([rbm.predictForUser(user, W, allUsersRatings) for user in trStats["u_users"]])
# predictedRatings = np.clip(predictedRatings, 1, 5)
# np.savetxt("predictedRatings.txt", predictedRatings)
