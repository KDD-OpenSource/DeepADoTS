import numpy as np
import torch


def fit_norm_distribution_param(
        model, train_dataset, prediction_window_size, device_type, channel_idx=0):
    predictions = []
    organized = []
    errors = []
    with torch.no_grad():
        # Turn on evaluation mode which disables dropout.
        model.eval()
        pasthidden = model.init_hidden(1)
        for t in range(len(train_dataset)):
            out, hidden = model.forward(train_dataset[t].unsqueeze(0), pasthidden)
            predictions.append([])
            organized.append([])
            errors.append([])
            predictions[t].append(out.data.cpu()[0][0][channel_idx])
            pasthidden = model.repackage_hidden(hidden)
            for prediction_step in range(1, prediction_window_size):
                out, hidden = model.forward(out, hidden)
                predictions[t].append(out.data.cpu()[0][0][channel_idx])

            if t >= prediction_window_size:
                for step in range(prediction_window_size):
                    organized[t].append(
                        predictions[step + t - prediction_window_size][prediction_window_size - 1 - step])
                organized[t] = torch.FloatTensor(organized[t]).to(device_type)
                errors[t] = organized[t] - train_dataset[t][0][channel_idx]
                errors[t] = errors[t].unsqueeze(0)

    errors_tensor = torch.cat(errors[prediction_window_size:], dim=0)
    mean = errors_tensor.mean(dim=0)
    cov = errors_tensor.t().mm(errors_tensor) / errors_tensor.size(0) - mean.unsqueeze(1).mm(mean.unsqueeze(0))
    # cov: positive-semidefinite and symmetric.

    return mean, cov


def anomalyScore(model, dataset, mean, cov, prediction_window_size, device_type, channel_idx=0, score_predictor=None):
    predictions = []
    rearranged = []
    errors = []
    hiddens = []
    predicted_scores = []
    with torch.no_grad():
        # Turn on evaluation mode which disables dropout.
        model.eval()
        pasthidden = model.init_hidden(1)
        for t in range(len(dataset)):
            out, hidden = model.forward(dataset[t].unsqueeze(0), pasthidden)
            predictions.append([])
            rearranged.append([])
            errors.append([])
            hiddens.append(model.extract_hidden(hidden))
            if score_predictor is not None:
                predicted_scores.append(
                    score_predictor.predict(model.extract_hidden(hidden).numpy()))

            predictions[t].append(out.data.cpu()[0][0][channel_idx])
            pasthidden = model.repackage_hidden(hidden)
            for prediction_step in range(1, prediction_window_size):
                out, hidden = model.forward(out, hidden)
                predictions[t].append(out.data.cpu()[0][0][channel_idx])

            if t >= prediction_window_size:
                for step in range(prediction_window_size):
                    start_idx = step + t - prediction_window_size
                    end_idx = prediction_window_size - 1 - step
                    rearranged[t].append(predictions[start_idx][end_idx])
                rearranged[t] = torch.FloatTensor(rearranged[t]).to(device_type).unsqueeze(0)
                errors[t] = rearranged[t] - dataset[t][0][channel_idx]
            else:
                rearranged[t] = torch.zeros(1, prediction_window_size).to(device_type)
                errors[t] = torch.zeros(1, prediction_window_size).to(device_type)

    predicted_scores = np.array(predicted_scores)
    scores = []
    for error in errors:
        mult1 = error - mean.unsqueeze(0)  # [ 1 * prediction_window_size ]
        mult2 = torch.inverse(cov)  # [ prediction_window_size * prediction_window_size ]
        mult3 = mult1.t()  # [ prediction_window_size * 1 ]
        score = torch.mm(mult1, torch.mm(mult2, mult3))
        scores.append(score[0][0])

    scores = torch.stack(scores)
    rearranged = torch.cat(rearranged, dim=0)
    errors = torch.cat(errors, dim=0)

    return scores, rearranged, errors, hiddens, predicted_scores
