import matplotlib.pyplot as plt
import numpy as np
import rbm
import projectLib as lib

def calculate_rmse(F1, gradientLearningRate1, beta1, decay_factor1, reg_coe1, batch_size1):
    training = lib.getTrainingData()
    validation = lib.getValidationData()
    trStats = lib.getUsefulStats(training)
    vlStats = lib.getUsefulStats(validation)

    allUsersRatings = lib.getAllUsersRatings(trStats["u_users"], training)

    # SET PARAMETERS HERE!!!
    K = 5
    epochs = 10   

    # plot parameters
    gradientLearningRate = gradientLearningRate1
    F = F1
    beta = beta1
    decay_factor = decay_factor1
    reg_coe = reg_coe1
    batch_size = batch_size1

    momentum = np.zeros([100, F, K])

    # parameters in early stopping
    best_rmse = float('inf')

    # Initialise all our arrays
    W = rbm.getInitialWeights(trStats["n_movies"], F, K)
    grad = np.zeros(W.shape)
    posprods = np.zeros(W.shape)
    negprods = np.zeros(W.shape)

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

    return best_rmse

# Function to find the optimal values for each parameter

best_F = None
best_learning_rate = None
best_beta = None
best_decay_factor = None
best_reg_coe = None
best_batch_size = None


def find_optimal_parameters_F(F_values):
    global best_F
    global best_rmse
    best_rmse = float('inf')
    # Iterate over F values
    F_rmse_values = []
    for F in F_values:
        vlRMSE = calculate_rmse(F, 0.01, 0.5, 0.99, 0.01, 1)
        F_rmse_values.append(vlRMSE)
        if vlRMSE < best_rmse:
            best_rmse = vlRMSE
            best_F = F
    print("local best F", best_F)
    return F_rmse_values


def find_optimal_parameters_learning_rate(learning_rate_values):
    global best_learning_rate
    global best_rmse
    best_rmse = float('inf')
    # Iterate over F values
    learning_rate_rmse_values = []
    for learning_rate in learning_rate_values:
        vlRMSE = calculate_rmse(best_F, learning_rate, 0.5, 0.99, 0.01, 1)
        learning_rate_rmse_values.append(vlRMSE)
        if vlRMSE < best_rmse:
            best_rmse = vlRMSE
            best_learning_rate = learning_rate
    print("local best learning_rate", best_learning_rate)

    return learning_rate_rmse_values


def find_optimal_parameters_beta(beta_values):
    global best_beta
    global best_rmse
    best_rmse = float('inf')
    # Iterate over F values
    beta_rmse_values = []
    for beta in beta_values:
        vlRMSE = calculate_rmse(best_F, best_learning_rate, beta, 0.99, 0.01, 1)
        beta_rmse_values.append(vlRMSE)
        if vlRMSE < best_rmse:
            best_rmse = vlRMSE
            best_beta = beta
    print("local best beta", best_beta)
        
    #plt.scatter(F_values, F_rmse_values, label=f'Epochs={best_epochs}, Batch Size={best_batch_size}')
    return beta_rmse_values



def find_optimal_parameters_decay_factor(decay_factor_values):
    global best_decay_factor
    global best_rmse
    best_rmse = float('inf')
    # Iterate over F values
    decay_factor_rmse_values = []
    for decay_factor in decay_factor_values:
        vlRMSE = calculate_rmse(best_F, best_learning_rate, best_beta, decay_factor, 0.01, 1)
        decay_factor_rmse_values.append(vlRMSE)
        if vlRMSE < best_rmse:
            best_rmse = vlRMSE
            best_decay_factor = decay_factor
    print("local best decay_factor", best_decay_factor)
        
    #plt.scatter(F_values, F_rmse_values, label=f'Epochs={best_epochs}, Batch Size={best_batch_size}')
    return decay_factor_rmse_values



def find_optimal_parameters_reg_coe(reg_coe_values):
    global best_reg_coe
    global best_rmse
    best_rmse = float('inf')
    # Iterate over F values
    reg_coe_rmse_values = []
    for reg_coe in reg_coe_values:
        vlRMSE = calculate_rmse(best_F, best_learning_rate, best_beta, best_decay_factor, reg_coe, 1)
        reg_coe_rmse_values.append(vlRMSE)
        if vlRMSE < best_rmse:
            best_rmse = vlRMSE
            best_reg_coe = reg_coe
    print("local best reg_coe", best_reg_coe)
        
    #plt.scatter(F_values, F_rmse_values, label=f'Epochs={best_epochs}, Batch Size={best_batch_size}')
    return reg_coe_rmse_values


def find_optimal_parameters_batch_size(batch_size_values):
    global best_batch_size
    global best_rmse
    best_rmse = float('inf')
    # Iterate over F values
    batch_size_rmse_values = []
    for bs in batch_size_values:
        vlRMSE = calculate_rmse(best_F, best_learning_rate, best_beta, best_decay_factor, best_reg_coe, bs)
        batch_size_rmse_values.append(vlRMSE)
        if vlRMSE < best_rmse:
            best_rmse = vlRMSE
            best_batch_size = bs
    print("local best_batch_size", best_batch_size)

    return batch_size_rmse_values



def main():
    global best_rmse
    best_rmse = float('inf')
    F_values = [5, 25, 50, 75]
    learning_rate_values = [0.005, 0.01, 0.05, 0.01]
    beta_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    decay_factor_values = [0.9, 0.93, 0.96, 0.99]
    reg_coe_values = [0.005, 0.01, 0.05]
    batch_size_values = [1, 12, 24, 48]

    F_rmse_values = find_optimal_parameters_F(F_values)
    learning_rate_rmse_values = find_optimal_parameters_learning_rate(learning_rate_values)
    beta_rmse_values = find_optimal_parameters_beta(beta_values)
    decay_factor_rmse_values = find_optimal_parameters_beta(decay_factor_values)
    reg_coe_rmse_values = find_optimal_parameters_reg_coe(reg_coe_values)
    batch_size_rmse_values = find_optimal_parameters_batch_size(batch_size_values)
    total = [(F_values,F_rmse_values),(learning_rate_values,learning_rate_rmse_values),
             (beta_values,beta_rmse_values),(decay_factor_values,decay_factor_rmse_values),
             (reg_coe_values,reg_coe_rmse_values),(batch_size_values,batch_size_rmse_values)]
    for (value, rmses) in total:
        # Plot the lines
        plt.plot(value, rmses)

        # Add a title and legend to the plot
        plt.title("Line Plots")
        plt.legend()

        # Save the plot to a file
        plt.savefig("line_plots1.png")

        # Create a new plot
        plt.clf()

if __name__ == "__main__":
    main()