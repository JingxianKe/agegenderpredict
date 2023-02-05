# import the necessary packages
import logging

def one_off_callback(model, trainIter, testIter, oneOff):
    def _callback(iterNum):
        # compute one-off metric for both the training and testing data
        trainMAE = _compute_one_off(model, trainIter, oneOff)
        testMAE = _compute_one_off(model, testIter, oneOff)

        # log the values
        logging.info("Epoch[{}] Train-one-off={:.2f}".format(iterNum, trainMAE))
        logging.info("Epoch[{}] Test-one-off={:.2f}".format(iterNum, testMAE))

    # return the callback method
    return _callback

def _compute_one_off(model, dataIter, oneOff):
    # initialize the total number of samples along with the
    # number of correct (maximum of one off) classifications
    total = 0
    correct = 0

    # loop over the predictions of batches
    for i, data in enumerate(dataIter):
        inputs, labels = data
        preds = model(inputs)

        # convert the batch of predictions and labels to NumPy arrays
        predictions = preds[0].asnumpy().argmax(axis=1)
        labels = labels[0].asnumpy().astype("int")

        # loop over the predicted labels and ground-truth labels in the batch
        for (pred, label) in zip(predictions, labels):
            # if correct label is in the set of "one off"
            # predictions, then update the correct counter
            if label in oneOff[pred]:
                correct += 1

            # increment the total number of samples
            total += 1

        # finish computing the one-off metric
        return correct / float(total)

