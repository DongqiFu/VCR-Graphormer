import os

def find_files_with_second_last_line(folder_path, target_string):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    if len(lines) >= 2 and target_string in lines[-2]:
                        print(file_path)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Replace '/path/to/your/folder' with the actual path to your folder
folder_path = 'log/twitch_gamer'

# Replace 'your_string' with the string you are looking for
target_string = 'accuracy= 0.62'

find_files_with_second_last_line(folder_path, target_string)
