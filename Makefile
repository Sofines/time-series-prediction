CONFIG_FILE=config.json
TRAIN_SCRIPT=training/train_model.py
EVAL_SCRIPT=evaluation/evaluate_model.py
RESULTS_FOLDER=results/
PREDICTIONS_FILE=$(RESULTS_FOLDER)predictions.csv
METRICS_FILE=$(RESULTS_FOLDER)evaluation_metrics.json

# Define the virtual environment activation command if you're using one
# VENV_ACTIVATE=source venv/bin/activate

all: help

train:
	@echo "Training the model..."
	python $(TRAIN_SCRIPT) $(CONFIG_FILE)
	@echo "Training complete."

evaluate:
	@echo "Evaluating the model..."
	python $(EVAL_SCRIPT) $(CONFIG_FILE)
	@echo "Evaluation complete."

clean:
	@echo "Cleaning up results..."
	rm -f $(PREDICTIONS_FILE) $(METRICS_FILE)
	@echo "Results folder cleaned."

help:
	@echo "Available commands:"
	@echo "  make train      - Train the model using the configuration file"
	@echo "  make evaluate   - Evaluate the model and save predictions/metrics"
	@echo "  make clean      - Remove all files from the results folder"
	@echo "  make help       - Show this help message"

