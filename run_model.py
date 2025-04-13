import torch
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
from helper import evaluate_model
from B03train_normalizing_flow_template import DataLoader, get_test_data, nf_loss
from B03train_normalizing_flow_template import CombinedModel, TinyCNNEncoder

plt.close()
def plot(argument):
  MODEL_FOLDER = f"model_120\\120_{argument}"
  MODEL_PATH = f'{MODEL_FOLDER}/model.pth'
  # DATA_PATH = f'Data'

  model = CombinedModel(TinyCNNEncoder, nf_type=argument)
  device = torch.device("cpu")
  state_dict = torch.load(MODEL_PATH, map_location=device)

  model.load_state_dict(state_dict)
  model.to(device)
  model.double()
  # model = torch.load(f'{MODEL_PATH}')
  test_loader,_ = get_test_data(batch_size=32)
  predictions, true_labels, first_batch_spectra, first_batch_labels, avg_test_loss = evaluate_model(model, test_loader,nf_loss,device)
  print(f'Avg test loss: {avg_test_loss}')


  test_loader,test_size = get_test_data(batch_size=1)
  y_pred = np.zeros((test_size,6))
  y_test = np.zeros((test_size,3))
  model.eval()
  # test_loss = 0
  itr = 0

  with torch.no_grad():
    for batch_x, batch_y in test_loader:
      batch_x = batch_x.unsqueeze(1)
      predictions = model(batch_x)
      y_pred[itr,:]= predictions.numpy()
      y_test[itr,:] = batch_y.numpy()
      itr+=1

  for integer in [0,1,2]:
    plt.scatter(y_pred[:,integer],y_test[:,integer])
    plt.plot([np.min(y_pred[:,integer])*1.05,np.max(y_pred[:,integer])*1.05],[np.min(y_test[:,integer])*1.05,np.max(y_test[:,integer])*1.05],"b--")
    plt.xlabel('pred')
    plt.ylabel('label')
    plt.title(f"Label {integer} for {argument}")
    plt.savefig(f'{MODEL_FOLDER}/plots/PredVLabel_{integer}.png')
    plt.close()


# # plot(0,arguments[0])
# plot(1,arguments[0])
# plot(2,arguments[0])

# plot(0,arguments[1])
# plot(1,arguments[1])
# plot(2,arguments[1])

# plot(0,arguments[2])
# plot(1,arguments[2])
# plot(2,arguments[2])

arguments = ["diagonal_gaussian", "full_gaussian", "full_flow"]
# plot(arguments[0])
# plot(arguments[1])
plot(arguments[2])