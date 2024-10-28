import csv

# Define the path to the CSV file
csv_file_path = "detected_times.csv"

# Open the CSV file
with open(csv_file_path, mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    
    previous_time = None
    hour_failures = 0
    minute_failures = 0
    second_failures = 0

    for row in csv_reader:
        current_time = row[1]  # Assuming the time is in the second column
        if previous_time:
            prev_h, prev_m, prev_s = map(int, previous_time.split(':'))
            curr_h, curr_m, curr_s = map(int, current_time.split(':'))

            if not (((curr_h == prev_h or curr_h == prev_h+1) and (curr_m == prev_m or curr_m == prev_m+1) and (curr_s == prev_s or curr_s == prev_s+1))):
                # print(f"Failure detected:\nPrevious: {previous_time}\n Current: {current_time}")
                
                if curr_h != prev_h or curr_h != prev_h+1:
                    hour_failures += 1
                elif curr_m != prev_m or curr_m != prev_m+1:
                    minute_failures += 1
                elif curr_s != prev_s or curr_s != prev_s+1:
                    second_failures += 1

        previous_time = current_time

print(f"Hour failures: {hour_failures}")
print(f"Minute failures: {minute_failures}")
print(f"Second failures: {second_failures}")