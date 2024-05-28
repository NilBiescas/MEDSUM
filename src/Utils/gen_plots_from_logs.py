import json
import matplotlib.pyplot as plt
import seaborn as sns

# Code to generate plots of the loss from the logs of the training process

name_model = "LoRA_MedicalReports_new_data"
# Load the logs
with open(f"/hhome/nlp2_g05/MEDSUM/src/Logs/logs_{name_model}.json", "r") as f:
    training_logs = json.load(f)


# Obtain the values logged during training
training_loss_values_batch = [x['loss'] for x in training_logs if 'loss' in x.keys()]
training_loss_steps = [x['epoch'] for x in training_logs if 'loss' in x.keys()]
eval_loss_values = [x['eval_loss'] for x in training_logs if 'eval_loss' in x.keys()]


# Plot training and validation loss
plt.figure(figsize=(15, 5))
sns.lineplot(x=training_loss_steps, y=training_loss_values_batch, label="Training Loss")
sns.lineplot(x=range(1, len(eval_loss_values)+1), y=eval_loss_values, label="Validation Loss")
plt.savefig(f"/hhome/nlp2_g05/MEDSUM/Training_plots/training_validation_loss{name_model}.png")
plt.close()
