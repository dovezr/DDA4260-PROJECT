import numpy as np
import rbm
import projectLib as lib

training = lib.getTrainingData()
validation = lib.getValidationData()


trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

allUsersRatings = lib.getAllUsersRatings(trStats["u_users"], training)

K = 5

# SET PARAMETERS HERE!!!
# number of hidden units
F = 5
epochs = 10
gradientLearningRate = 0.1

# Set early stopping variables
best_RMSE = float('inf')
best_weights = None
# Initialise all our arrays
W = rbm.getInitialWeights(trStats["n_movies"], F, K)
grad = np.zeros(W.shape)
grad_squared = np.zeros(W.shape)
posprods = np.zeros(W.shape)
negprods = np.zeros(W.shape)

### hyper-parameter for extensions ###
# momentum
beta = 0.5
momentum = np.zeros([100, F, K])

# regularization
# reg = 0.01

for epoch in range(1, epochs+1):
    # in each epoch, we'll visit all users in a random order
    visitingOrder = np.array(trStats["u_users"])
    np.random.shuffle(visitingOrder)

    for user in visitingOrder:
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


        # extension
        # momentum
        grad[ratingsForUser[:, 0], :, :] = gradientLearningRate * (posprods[ratingsForUser[:, 0], :, :] - negprods[ratingsForUser[:, 0], :, :])
        momentum[ratingsForUser[:, 0], :, :] = beta * momentum[ratingsForUser[:, 0], :, :] + (posprods[ratingsForUser[:, 0], :, :] - negprods[ratingsForUser[:, 0], :, :])
        # W[ratingsForUser[:, 0], :, :] += grad[ratingsForUser[:, 0], :, :]
        W[ratingsForUser[:, 0], :, :] += gradientLearningRate * momentum[ratingsForUser[:, 0], :, :]
        
        # regularization
        # if reg != 0: # put this after other changes to W
        #     grad[ratingsForUser[:, 0], :, :] += - gradientLearningRate * reg_coe * W[ratingsForUser[:, 0], :, :]

    # Print the current RMSE for training and validation sets
    # this allows you to control for overfitting e.g
    # We predict over the training set
    tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, allUsersRatings)
    tr_r_hat = np.clip(tr_r_hat, 1, 5)
    trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

    # We predict over the validation set
    vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, allUsersRatings)
    vl_r_hat = np.clip(vl_r_hat, 1, 5)
    vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)


    print("### EPOCH %d ###" % epoch)
    print("Training loss = %f" % trRMSE)
    print("Validation loss = %f" % vlRMSE)

### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
predictedRatings = np.array([rbm.predictForUser(user, W, allUsersRatings) for user in trStats["u_users"]])
predictedRatings = np.clip(predictedRatings, 1, 5)
np.savetxt("predictedRatings.txt", predictedRatings)
