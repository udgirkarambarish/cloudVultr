import math

# Function to calculate the distance between two points (Euclidean distance)
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to calculate the area of a polygon using Shoelace Theorem
def calculate_area(vertices):
    n = len(vertices)
    area = 0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2

# Function to check if a closed polygon can be formed
def is_closed_polygon(sticks):
    points_count = {}
    
    # Count the occurrences of each point
    for x1, y1, x2, y2 in sticks:
        points_count[(x1, y1)] = points_count.get((x1, y1), 0) + 1
        points_count[(x2, y2)] = points_count.get((x2, y2), 0) + 1
    
    # For a closed polygon, each point must occur exactly twice (once at the start and end)
    for count in points_count.values():
        if count != 2:
            return False
    return True

# Function to process the problem
def solve_problem(sticks):
    # Step 1: Check if a closed figure is formed
    if not is_closed_polygon(sticks):
        print("No")
        return
    
    # Step 2: Find the vertices of the closed polygon
    vertices = []
    for x1, y1, x2, y2 in sticks:
        if (x1, y1) not in vertices:
            vertices.append((x1, y1))
        if (x2, y2) not in vertices:
            vertices.append((x2, y2))
    
    # Step 3: Calculate the area of the closed figure using Shoelace Theorem
    area = calculate_area(vertices)
    
    # Step 4: Check if Arjun can form the same figure with leftover sticks
    # Calculate the perimeter of the closed figure
    perimeter = 0
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        perimeter += distance(x1, y1, x2, y2)
    
    # Calculate the total length of the leftover sticks
    leftover_length = 0
    for x1, y1, x2, y2 in sticks:
        leftover_length += distance(x1, y1, x2, y2)
    
    if leftover_length == perimeter:
        can_form_same_shape = "Yes"
    else:
        can_form_same_shape = "No"
    
    # Step 5: Output the results
    print("Yes")
    print(can_form_same_shape)
    print(f"{area:.2f}", end="")

# Main function to handle input and execute the solution
def main():
    # Input processing
    N = int(input().strip())  # Number of sticks
    sticks = []
    
    for _ in range(N):
        x1, y1, x2, y2 = map(int, input().split())
        sticks.append((x1, y1, x2, y2))
    
    # Solve the problem
    solve_problem(sticks)

# Run the program
if __name__ == "__main__":
    main()
