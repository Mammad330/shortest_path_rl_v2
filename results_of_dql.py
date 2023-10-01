from utils import save_results

def main():
    # List of date variables for different evaluation sessions
    date_variables = [
        '2023-09-24_22-56'
          # Add more date variables as needed
    ]

    # Process and print evaluation data for each date variable
    for date_variable in date_variables:
        print(f"Analysis for Evaluation Session: {date_variable}")
        save_results(date_variable)
        print("\n" + "=" * 50 + "\n")  # Separator

if __name__ == "__main__":
    main()
