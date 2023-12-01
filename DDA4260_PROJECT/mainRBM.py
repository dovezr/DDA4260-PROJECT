import numpy as np
import rbm
import projectLib as lib
# import neighborhood
from sklearn.model_selection import KFold

# kf = KFold(n_splits=5, shuffle=True)  # 创建K折交叉验证对象

training = lib.getTrainingData()
validation = lib.getValidationData()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
allUsersRatings = lib.getAllUsersRatings(trStats["u_users"], training)

K = 5
# patameters = [[15,0.02,0.001],[25,0.02,0.0015]]
# SET PARAMETERS HERE!!!
# number of hidden units5
F = 5
epochs = 5
gradientLearningRate = 0.01
# regularisation
reg = 0.001
# mini-batch
batch_size = 1
beta = 0.5
###########
n_folds = 5 ##

# Initialise all our arrays
W = rbm.getInitialWeights(trStats["n_movies"], F, K)
grad = np.zeros(W.shape)
posprods = np.zeros(W.shape)
negprods = np.zeros(W.shape)

### hyper-parameter for extensions ###
# momentum
momentum = np.zeros([trStats["n_users"], F, K])

# bias
b_h = np.random.normal(0, 0.1, (F))
b_v = np.random.normal(0, 0.1, (trStats["n_movies"], K))
grad_bh = np.zeros(b_h.shape)
grad_bv = np.zeros(b_v.shape)
momentum_bh = np.zeros(b_h.shape)
momentum_bv = np.zeros(b_v.shape)

training_loss_list = []
validation_loss_list = []
best_RMSE = float('inf')
best_weights = None
best_bh = None
best_bv = None

kf = KFold(n_splits=n_folds, shuffle=True, random_state=42) ##

for fold, (train_index, val_index) in enumerate(kf.split(training)):
    print(f"### Fold {fold + 1} ###")

    training_fold = training[train_index]
    validation_fold = training[val_index]

    trStats = lib.getUsefulStats(training_fold)
    vlStats = lib.getUsefulStats(validation_fold)
    allUsersRatings = lib.getAllUsersRatings(trStats["u_users"], training_fold)


    for epoch in range(epochs):
        # in each epoch, we'll visit all users in a random order
        visitingOrder = np.array(trStats["u_users"])
        np.random.shuffle(visitingOrder)
        
        for batch_start in range(0, len(visitingOrder), batch_size):
            batch_end = batch_start + batch_size
            batch_users = visitingOrder[batch_start:batch_end]
            
            for user in batch_users:
                # Rest of the code remains the same
                # get the ratings of that user
                ratingsForUser = allUsersRatings[user]
                # build the visible input
                v = rbm.getV(ratingsForUser)
                # get the weights associated to movies the user has seen
                weightsForUser = W[ratingsForUser[:, 0], :, :]

                ### LEARNING ###
                # propagate visible input to hidden units
                posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser, b_h)
                # get positive gradient
                # note that we only update the movies that this user has seen!
                posprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(v, posHiddenProb)

                ### UNLEARNING ###
                # sample from hidden distribution
                sampledHidden = rbm.sample(posHiddenProb) 
                # propagate back to get "negative data"
                negData = rbm.hiddenToVisible(sampledHidden, weightsForUser, b_v[ratingsForUser[:, 0], :])
                # propagate negative data to hidden units
                negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser, b_h)
                # get negative gradient
                # note that we only update the movies that this user has seen!
                negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(negData, negHiddenProb)
                # we average over the number of users in the batch (if we use mini-batch)
                grad[ratingsForUser[:, 0], :, :] = 1/batch_size * \
                            (posprods[ratingsForUser[:, 0], :, :] - negprods[ratingsForUser[:, 0], :, :])
                grad_bh = 1/batch_size * (posHiddenProb - negHiddenProb)
                grad_bv[ratingsForUser[:, 0], :] = 1/batch_size * (v - negData).mean(axis=0)
                # momentum:
                # momentum[ratingsForUser[:, 0], :, :] *= (1-beta)
                momentum[ratingsForUser[:, 0], :, :] = beta * (momentum[ratingsForUser[:, 0], :, :]) + \
                                    grad[ratingsForUser[:, 0], :, :]
                # momentum_bh *= (1-beta)
                momentum_bh = beta * (momentum_bh) + grad_bh
                # momentum_bh += grad_bh
                # momentum_bv *= (1-beta)
                momentum_bv[ratingsForUser[:, 0], :] = beta * momentum_bv[ratingsForUser[:, 0], :] \
                                + grad_bv[ratingsForUser[:, 0], :]
                # momentum_bv[ratingsForUser[:, 0], :] += grad_bv[ratingsForUser[:, 0], :]
            
            W[ratingsForUser[:, 0], :, :] += gradientLearningRate * momentum[ratingsForUser[:, 0], :, :]
            b_h += gradientLearningRate * momentum_bh
            # b_h += momentum_bh
            b_v[ratingsForUser[:, 0], :] +=  gradientLearningRate * momentum_bv[ratingsForUser[:, 0], :]
            # regularision
            W[ratingsForUser[:, 0], :, :] -= reg * W[ratingsForUser[:, 0], :, :]
            # b_h -=  reg * b_h 
            # b_v[ratingsForUser[:, 0], :] -= reg * b_v[ratingsForUser[:,0], :]

        # Print the current RMSE for training and validation sets
        # this allows you to control for overfitting e.g
        # We predict over the training set
        tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, allUsersRatings, b_h, b_v)
        trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

        # We predict over the validation set
        vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, allUsersRatings, b_h, b_v)
        vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

        training_loss_list.append(trRMSE)
        validation_loss_list.append(vlRMSE)

        print("### EPOCH %d ###" % (epoch+1))
        print("Training loss = %f" % trRMSE)
        print("Validation loss = %f" % vlRMSE)
        
        # if lib.early_stop(validation_loss_list, stop_size=5, stop_threshold=-0.005): break
        # gradientLearningRate *= lib.adaptive_lr(validation_loss_list, test_size=3, improve_threshold=0.05, rate=0.5)

        if vlRMSE < best_RMSE:
            best_weights = W
            best_bh = b_h
            best_bv = b_v
            best_RMSE = vlRMSE
    # lib.plot_loss(training_loss_list, validation_loss_list)
    ### END ###
            # This part you can write on your own
            # you could plot the evolution of the training and validation RMSEs for example
predictedRatings = np.array([rbm.predictForUser(user, best_weights, allUsersRatings, best_bh, best_bv) for user in trStats["u_users"]])
for (movie, user, rating) in training:
    predictedRatings[user, movie] = rating
for (movie, user, rating) in validation:
    predictedRatings[user, movie] = rating
name = str(best_RMSE) + str(F) + str(gradientLearningRate) + str(reg) + ".txt"
np.savetxt(name, predictedRatings, delimiter=' ')
# neighborhood.neighbourhood(training, predictedRatings)
