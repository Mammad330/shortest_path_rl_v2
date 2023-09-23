from utils import process_npy_files


def main():
    npy_file_path1 = 'training/DQL/2023-09-16_16-55/plots/train_reward.npy'
    npy_file_path2 = 'training/DQL/2023-09-16_16-55/plots/eval_reward.npy'
    window_size = 100  # Adjust the window size for the running average
    output_csv_file = 'training/DQL/2023-09-16_16-55/plots/Result_of_rewards.csv'  # Specify the output CSV file name

    process_npy_files(npy_file_path1, npy_file_path2, window_size, output_csv_file)

if __name__ == "__main__":
    main()