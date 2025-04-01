from imu_uwb_pose import config as c, data_extraction as de
import numpy as np
import os

import os
import numpy as np
import torch

def process_amass():
    config = c.config()
    processed = []

    processed_dir = os.path.join(config.processed_pose, "AMASS", "train")
    if os.path.exists(processed_dir):
        processed = os.listdir(processed_dir)

    for dataset in config.amass_datasets:
        if dataset in processed:
            continue
        
        dataset_path = os.path.join(config.root_dir, config.raw_amass, dataset)

        # in jupyter amass data is nested so add the first dir in the dataset path
        subdir = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))][0]
        dataset_path = os.path.join(dataset_path, subdir)
        
        if not os.path.exists(dataset_path):
            continue

        subject_names = [s for s in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, s))]
        subject_count = 0

        for subject in subject_names:
            data_location = 'train'
            if (subject_count % 10) == 0:
                data_location = 'test'
            subject_count += 1
            subject_path = os.path.join(dataset_path, subject)

            action_files = [f for f in os.listdir(subject_path) if f.endswith(".npz")]

            for action in action_files:
                action_path = os.path.join(subject_path, action)

                # Load the data
                print(f'processing {action}')
                try:
                    data = np.load(action_path)
                except:
                    print("unable to open file")
                    continue

                # Process the data
                output = de.extract_amass(data, config)

                if output is None:
                    print("output is none")
                    continue
                
                
                # Save the data
                save_dir = os.path.join(config.processed_pose, "AMASS", data_location, dataset, subject)
                os.makedirs(save_dir, exist_ok=True)

                # Save the processed output
                save_path = os.path.join(save_dir, action.split(".")[0] + ".pt")
                torch.save(output, save_path)




    
    
    

if __name__=="__main__":
    process_amass()