def add_comma_and_braces(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        file.write('[\n')
        for i, line in enumerate(lines):
            # Add a comma at the end of each line except the last one
            comma = ',' if i < len(lines) - 1 else ''
            file.write(f'    {line.strip()}{comma}\n')
        file.write(']')

# Example usage:
input_file_path = 'News_Category_Dataset_IS_course.json'  # Replace with your input file path
output_file_path = 'dataset.json'  # Replace with your output file path

add_comma_and_braces(input_file_path, output_file_path)