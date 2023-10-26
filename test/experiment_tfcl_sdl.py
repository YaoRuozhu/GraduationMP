import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/Users/meruozhu/Downloads/MP_data/MP_codes/MP')
import time
from datetime import datetime 
from pathlib import Path
import json
import csv
import numpy as np
import pandas as pd
from experiments.config import ConfigParser
from experiments.config import Recorder
from DPM.task_free_continual_learning.main_provide_data_model import main as tfcl_main

def save_figure(training_losses, future_losses, config):
    #print("{0}: task {1}: {2}".format(tag,task,test_losses[tag][task][-1]))
    datetime_str= datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Define the folder to save the CSV files
    output_folder_results =Path(__file__).parent/"results"/f"{Path(config.get_dataset()).stem}_rbs{config.get_recent_buffer_size()}_hbs{config.get_hard_buffer_size()}_{datetime_str}/figures"
    output_folder_results.mkdir(parents=True, exist_ok=True)
    print(output_folder_results)
    figure_name = f"{Path(config.get_dataset()).stem}_rbs{config.get_recent_buffer_size()}_hbs{config.get_hard_buffer_size()}_{datetime_str}.png"
    figure_path = os.path.join(output_folder_results, figure_name)
    # # Save the list to a CSV file
    # with open(filepath, "w", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(test_losses[tag][task])
    plt.figure(figsize=(10, 6))  # create a new figure with specified size
    # x = np.arange(len(test_losses[tag][task])) * recent_buffer_size
    x = np.arange(len(future_losses)) * config.get_recent_buffer_size()
    #plt.plot(x, test_losses['Online Continual'][0], marker='o', label='Test')
    plt.plot(x, future_losses, marker='o', label='Future')
    plt.plot(x, training_losses, marker='o', label='Train')
    plt.title('Accuracy')  # set the title of the plot
    plt.xlabel('Event Number')  # set the label for x-axis
    plt.ylabel('Accurcy')  # set the label for y-axis
    plt.grid(True)  # add a grid
    plt.legend()  # display the labels in a legend
    plt.savefig(figure_path)
    for dataname in ['loss_window_means','update_tags','loss_window_variances']:
        plt.figure(figsize=(10, 6))  # create a new figure with specified size
        plt.title(dataname)
        #for i in range(ntasks): plt.axvline(x=(i+1)*ntrain,color='gray')
        plt.plot(eval(dataname))
        plt.savefig(os.path.join(output_folder_results, f"{dataname}.png"))
    # plt.show()   

def save_results(training_losses, future_losses, running_time, prediction_results, batch_accuracy, config):
    # df = pd.DataFrame(batch_accuracy)
    datetime_str= datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Define the folder to save the CSV files
    output_folder_results =Path(__file__).parent/"results"/f"{Path(config.get_dataset()).stem}_rbs{config.get_recent_buffer_size()}_hbs{config.get_hard_buffer_size()}_{datetime_str}"
    if output_folder_results.exists():
        print(f"Folder {output_folder_results} already exists. Results will be overwritten.")
    output_folder_results.mkdir(parents=True, exist_ok=True)
    print(output_folder_results)

    experiment_record = {}

    experiment_record["accuracy"] = {
        "future_losses": future_losses,
    }
    experiment_record["running_time"] = running_time
    experiment_record["ratio"] = config.get_ratio()

    # # Save the dictionary as JSON
    with open(str(output_folder_results/"evaluation_res.json"), "w") as file:
        json.dump(experiment_record, file, indent=4)

    print(f"Experiment record saved to {output_folder_results}")

    filename = f"{Path(config.get_dataset()).stem}_rbs{config.get_recent_buffer_size()}_hbs{config.get_hard_buffer_size()}_{datetime_str}.json"
    filepath = os.path.join(output_folder_results, filename)

    # # Combine the lists into a list of tuples
    # data = list(zip(training_losses, test_losses['Online Continual'][0], future_losses))

    # Define the dictionary containing the experiment records
    experiment_record = {}

    experiment_record["accuracy"] = {
        "training_losses": training_losses,
        # "test_losses": test_losses['Online Continual'][0],
        "future_losses": future_losses,
        # Add more key-value pairs as needed
    }
    experiment_record["running_time"] = running_time

    # # Save the dictionary as JSON
    with open(filepath, "w") as file:
        json.dump(experiment_record, file, indent=4)

    print(f"Experiment record saved to {filepath}")

    # After the loop, save the results to a CSV file
    pd.DataFrame(prediction_results).to_csv(os.path.join(output_folder_results, "prediction_results.csv"), index=False)

    pd.DataFrame(batch_accuracy).to_csv(os.path.join(output_folder_results, "batch_accuracy.csv"), index=False)


def compute_batch_fixed_size_accuracy(prediction_results,fixed_size):
    # Convert the lists to numpy arrays
    actual_labels_np = np.array(prediction_results['actual_labels'])
    prediction_labels_np = np.array(prediction_results['prediction_labels'])

    # Create the modified dictionary
    prediction_results = np.concatenate((actual_labels_np.reshape(-1, 1), prediction_labels_np.reshape(-1, 1)), axis=1)
    numEvents = prediction_results.shape[0]
    numBatches = int(numEvents/fixed_size)
    print("Num batches is {}".format(numBatches))
    batch_accuracy = []
    for i in range(numBatches):
        batch_accuracy.append(np.sum(prediction_results[i*fixed_size:(i+1)*fixed_size,0]==prediction_results[i*fixed_size:(i+1)*fixed_size,1])/fixed_size)
    return batch_accuracy

def main(config):
    training_losses, future_losses, running_time, loss_window_means, update_tags, loss_window_variances, prediction_results = tfcl_main(config.get_dataset(),config.get_recent_buffer_size(),config.get_hard_buffer_size(),config.get_ratio())
    return training_losses, future_losses, running_time, loss_window_means, update_tags, loss_window_variances, prediction_results

if __name__ == "__main__":

    # configPath = Path("./configs")
    configPath=Path("/Users/meruozhu/Downloads/MP_data/MP_codes/MP/experiments/experiments_tfcl_sdl")/"configs"
    for configfile in configPath.glob("*.json"):
        # 通过这串代码，当一个配置文件的运行出现错误时会被跳过，执行except语句从而继续运行下一个配置文件
        try:
            config = ConfigParser.from_file(configfile)
            print(config.get_dataset(),config.get_recent_buffer_size(),config.get_hard_buffer_size())
            training_losses, future_losses, running_time, loss_window_means, update_tags, loss_window_variances, prediction_results = main(config)
            batch_accuracy = compute_batch_fixed_size_accuracy(prediction_results,config.get_recent_buffer_size())
            print(batch_accuracy)
            save_results(training_losses, future_losses, running_time, prediction_results, batch_accuracy,config)
            save_figure(training_losses, future_losses, config)
            print("Time taken is {}".format(running_time))
        except Exception as e:
            print(e)
            print("Error in {}".format(str(configfile)))
            continue