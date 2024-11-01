import csv

# Define the path to the CSV file
csv_file_path = "detected_times.csv"

# Open the CSV file
with open(csv_file_path, mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    
    previous_time = None
    hour_failures = 0
    hour_invalid_transitions = 0
    minute_failures = 0
    second_failures = 0

    for row in csv_reader:
        current_time = row[1]  # Time is in the second column
        if previous_time:
            prev_h, prev_m, prev_s = map(int, previous_time.split(':'))
            curr_h, curr_m, curr_s = map(int, current_time.split(':'))

            if not (((curr_h == prev_h or curr_h == prev_h+1) and (curr_m == prev_m or curr_m == prev_m+1) and (curr_s == prev_s or curr_s == prev_s+1))):
                # print(f"Failure detected:\nPrevious: {previous_time}\n Current: {current_time}")
                
                if curr_h != prev_h and curr_h != prev_h+1 and not (curr_h == 0 and prev_h == 11):
                    hour_failures += 1
                if curr_m != prev_m and curr_m != prev_m+1:
                    if (curr_m == 0 and prev_m != 59):
                        minute_failures += 1
                if curr_s != prev_s and  (curr_s >= prev_s+3 or curr_s < prev_s): # considering 3s jump tolerance
                    if not (curr_s < 3 and prev_s > 57):
                        second_failures += 1
                        print(f"Failure detected at row {csv_reader.line_num}:\nPrevious: {previous_time}\nCurrent: {current_time}")
                if curr_h==prev_h+1 and (curr_m!=0 or prev_m!=59):
                    hour_invalid_transitions += 1
                    print(csv_reader.line_num)
        previous_time = current_time

print(f"Hour invalid transitions: {hour_invalid_transitions}")
print(f"  Hour invalid increment: {hour_failures}")
print(f"Minute invalid increment: {minute_failures}")
print(f"Second invalid increment: {second_failures}")