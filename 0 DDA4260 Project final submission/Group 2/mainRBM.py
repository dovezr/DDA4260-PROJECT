import numpy as np
import rbm
import projectLib as lib
import neighbour as nb


training = lib.getTrainingData()
validation = lib.getValidationData()


trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)


allUsersRatings = lib.getAllUsersRatings(trStats["u_users"], training)


# SET PARAMETERS HERE!!!
K = 5
F = 5
epochs = 950
gradientLearningRate = 0.05


# parameters in momentum
beta = 0.5
momentum = np.zeros([100, F, K])


# parameters in adaptive learning rate
decay_factor = 0.99


# parameters in early stopping
best_rmse = 999
bh_best = 999
bv_best = 999
W_best = rbm.getInitialWeights(trStats["n_movies"], F, K)


# parameters in regularisation
lambda_reg = 0.1
reg_coe = 0.01


# Initialise all our arrays
W = rbm.getInitialWeights(trStats["n_movies"], F, K)
grad = np.zeros(W.shape)
posprods = np.zeros(W.shape)
negprods = np.zeros(W.shape)


# parameter in mini batch
batch_size = 1

# parameters in biases
bh = np.random.normal(0, 0.1, (F))
bv = np.random.normal(0, 0.1, (trStats["n_movies"], K))
gradbh = np.zeros(bh.shape)
gradbv = np.zeros(bv.shape)
momentumbh = np.zeros(bh.shape)
momentumbv = np.zeros(bv.shape)


for epoch in range(1, epochs):
    # in each epoch, we'll visit all users in a random order
    visitingOrder = np.array(trStats["u_users"])
    np.random.shuffle(visitingOrder)

    # parameter in mini batch
    num = 0

    # extension of adaptive learning rates
    gradientLearningRate *= decay_factor

    for batch_start in range(0, len(visitingOrder), batch_size):
        batch_end = batch_start + batch_size
        batch_users = visitingOrder[batch_start:batch_end]

        for user in batch_users:
            # get the ratings of that user
            ratingsForUser = allUsersRatings[user]

            # build the visible input
            v = rbm.getV(ratingsForUser)

            # get the weights associated to movies the user has seen
            weightsForUser = W[ratingsForUser[:, 0], :, :]

            ### LEARNING ###
            # propagate visible input to hidden units
            posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser, bh)
            # get positive gradient
            # note that we only update the movies that this user has seen!
            posprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(v, posHiddenProb)

            ### UNLEARNING ###
            # sample from hidden distribution
            sampledHidden = rbm.sample(posHiddenProb)

            # propagate back to get "negative data"
            negData = rbm.hiddenToVisible(
                sampledHidden, weightsForUser, bv[ratingsForUser[:, 0], :]
            )
            # propagate negative data to hidden units
            negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser, bh)
            # get negative gradient
            # note that we only update the movies that this user has seen!
            negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(
                negData, negHiddenProb
            )

            # we average over the number of users in the batch (if we use mini-batch)

            # the extension of the biases (Initialization operation)
            grad[ratingsForUser[:, 0], :, :] = gradientLearningRate * (
                posprods[ratingsForUser[:, 0], :, :]
                - negprods[ratingsForUser[:, 0], :, :]
            )
            gradbv[ratingsForUser[:, 0], :] = v - negData
            gradbh = posHiddenProb - negHiddenProb

            # the extension of the momentum
            momentum[ratingsForUser[:, 0], :, :] = (
                beta * momentum[ratingsForUser[:, 0], :, :]
                + grad[ratingsForUser[:, 0], :, :]
            )
            momentumbh = beta * momentumbh + gradbh
            momentumbv[ratingsForUser[:, 0], :] = (
                beta * momentumbv[ratingsForUser[:, 0], :]
                + gradbv[ratingsForUser[:, 0], :]
            )
            gradbh = momentumbh
            gradbv[ratingsForUser[:, 0], :] = momentumbv[ratingsForUser[:, 0], :]

            # the extension of the regularisation
            grad[ratingsForUser[:, 0], :, :] += (
                -gradientLearningRate * reg_coe * W[ratingsForUser[:, 0], :, :]
            )
            gradbv[ratingsForUser[:, 0], :] += (
                -gradientLearningRate * reg_coe * gradbv[ratingsForUser[:, 0], :]
            )
            gradbh += -gradientLearningRate * reg_coe * gradbh

            # the extension of the biases (Iterative operation)

        W[ratingsForUser[:, 0], :, :] += momentum[ratingsForUser[:, 0], :, :]
        bv[ratingsForUser[:, 0], :] += (
            gradientLearningRate * gradbv[ratingsForUser[:, 0], :]
        )
        bh += gradientLearningRate * gradbh
        num = 0

    # Print the current RMSE for training and validation sets
    # this allows you to control for overfitting e.g
    # We predict over the training set
    tr_r_hat = rbm.predict(
        trStats["movies"], trStats["users"], W, allUsersRatings, bh, bv
    )
    trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

    # We predict over the validation set
    vl_r_hat = rbm.predict(
        vlStats["movies"], vlStats["users"], W, allUsersRatings, bh, bv
    )
    vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

    print("### EPOCH %d ###" % epoch)
    print("Training loss = %f" % trRMSE)
    print("Validation loss = %f" % vlRMSE)

    # the extension of the early stopping
    if vlRMSE < best_rmse:
        best_rmse = vlRMSE
        W_best = W
        bh_best = bh
        bv_best = bv


### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
predictedRatings = np.array(
    [
        rbm.predictForUser(user, W_best, allUsersRatings, bh_best, bv_best)
        for user in trStats["u_users"]
    ]
)


# the extension of the neibourhood method (the detail can be seen in the package neibour)
predictedRatings = nb.main(predictedRatings)


np.savetxt("predictedRatings.txt", predictedRatings)
