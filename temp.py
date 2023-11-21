# Initialize an empty list to store the data
data = []

# Open the file in read mode
with open('opt.txt', 'r') as file:
    # Read lines one by one and add them to the data list
    for line in file:
        print(line.strip()[:3])
        data.append(1 if line.strip()[:3]=='opt' else 0)  # Convert the line to a float and remove any trailing newline characters

# Initialize an empty list to store the 2D vectors
vectors_2d = []

# Open the text file in read mode
with open('vec.txt', 'r') as file:
    # Iterate through the lines of the file
    for line in file:
        # Split the line based on the delimiter (e.g., space or comma)
        components = line.strip().split()  # Use .split(',') for comma-separated values
        if len(components) == 2:
            # Convert components to floats and create a 2D vector as a tuple
            vector_2d = (float(components[0]), float(components[1]))
            # Append the 2D vector to the list
            vectors_2d.append(vector_2d)

import matplotlib.pyplot as plt

for i in range(len(vectors_2d)):
    x, y = vectors_2d[i]
    color = 'red' if data[i] == 1 else 'blue'
    plt.scatter(x, y, color=color)

# Show the plot
plt.show()


