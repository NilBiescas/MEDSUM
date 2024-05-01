import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load the logs
with open("/hhome/nlp2_g05/Asho_NLP/logs.json", "r") as f:
    training_logs = json.load(f)


# Obtain the values logged during training
training_loss_values_batch = [x['loss'] for x in training_logs if 'loss' in x.keys()]
training_loss_steps = [x['epoch'] for x in training_logs if 'loss' in x.keys()]
eval_loss_values = [x['eval_loss'] for x in training_logs if 'eval_loss' in x.keys()]


# Plot training and validation loss
plt.figure(figsize=(15, 5))
sns.lineplot(x=training_loss_steps, y=training_loss_values_batch, label="Training Loss")
sns.lineplot(x=range(1, len(eval_loss_values)+1), y=eval_loss_values, label="Validation Loss")
plt.savefig("training_validation_loss.png")
plt.close()
