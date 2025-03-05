import argparse

import pandas as pd


def txt_to_csv(input_txt: str, output_csv: str) -> None:
    """
    Convert a .txt file to a .csv file with a 'text' column using pandas.

    Args:
        input_txt (str): Path to the input .txt file.
        output_csv (str): Path to the output .csv file.

    Returns:
        None
    """
    try:
        # Read the .txt file, each line becomes an entry in a list
        with open(input_txt, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Strip newline characters from each line
        lines = [line.strip() for line in lines]

        # Create a DataFrame with a single column 'text'
        df = pd.DataFrame(lines, columns=["text"])

        # Write DataFrame to CSV
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"CSV file '{output_csv}' created successfully.")

    except FileNotFoundError:
        print(f"Error: The file {input_txt} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: The parsed arguments containing input_txt and output_csv.
    """
    parser = argparse.ArgumentParser(description="Convert a .txt file to a .csv file.")
    parser.add_argument("input_txt", type=str, help="Path to the input .txt file.")
    parser.add_argument("output_csv", type=str, help="Path to the output .csv file.")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args: argparse.Namespace = parse_arguments()

    # Call the conversion function with parsed arguments
    txt_to_csv(args.input_txt, args.output_csv)
