import csv
import shutil
import os

from replace1 import replace1


def modify_file(template_file, new_file, row):
    with open(template_file, 'r') as file:
        lines = file.readlines()

    # Modify specific lines in the new file
    lines[522] = f"          Unit Vector Y Component = {row[4]}\n"
    lines[523] = f"          Unit Vector Z Component = {row[3]}\n"
    lines[534] = f"          Relative Pressure = {row[2]} [kPa]\n"

    with open(new_file, 'w') as file:
        file.writelines(lines)

def create_files_from_csv(csv_file, template_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            i_value = row[0].strip()
            ma_value = row[1].strip()
            pressure_value = row[2].strip()
            vector_value = row[3].strip()


            folder_name = f"A1-baseline-Ma{ma_value}-4-{i_value}"
            os.makedirs(folder_name, exist_ok=True)


            new_filename = os.path.join(folder_name, folder_name+".ccl")
            shutil.copy(template_file, new_filename)
            modify_file(template_file, new_filename, row)
            replace1(folder_name)





            

# Example usage
csv_file = 'data.csv'  # replace with your CSV file name
template_file = 'A1-baseline-Ma0.69-4-i0.ccl'  # replace with your template CCL file name
create_files_from_csv(csv_file, template_file)

