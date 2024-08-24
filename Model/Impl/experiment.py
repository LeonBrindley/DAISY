import data
import plantnet
import validate
import serialisation

# Load data
train_loader, val_loader, train_dataset, val_dataset, label_map = data.load_data((229,229), "~/gp/dataset-v4.csv")

# Fine tune PlantNet
model, train_stats, val_stats = plantnet.fine_tune_plantnet(train_loader, val_loader, learning_rate=0.001, epochs=1)

# Perform validation inference
val_predictions = validate.validation_inference(model, val_dataset)

# Save the results
model_dir = f"/rds/user/omsst2/hpc-work/gp/results/multiclass_plantnet_lr0.001_e1"
serialisation.save_model(model, train_stats, val_stats, val_predictions, model_dir)