import data
import validate
import serialisation

# Load data
train_loader, val_loader, train_dataset, val_dataset, label_map = data.load_data((229,229), "~/gp/dataset-v4.csv")

# Load model
model_dir = f"/rds/user/omsst2/hpc-work/gp/results/multiclass_plantnet_lr0.001_e1"
# model_dir = "C:/Users/omart/Desktop/gp/results/multiclass_plantnet_lr0.001_e1"
# model, train_stats, val_stats, val_predictions = serialisation.load_model(model_dir, plantnet.load_PlantNet_model())
train_stats, val_stats, val_predictions = serialisation.load_only_stats(model_dir)

# Show stats
validate.show_training_history(train_stats, val_stats, "Fine-tuned PlantNet training history")
validate.show_confusion(val_dataset, label_map, val_predictions)
sklearn_accuracy, keras_acc = validate.show_accuracy(val_dataset, val_predictions)