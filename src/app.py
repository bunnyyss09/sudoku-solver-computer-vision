import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load your trained digit classification model
model = load_model("my_model.h5"    )

# Function to preprocess the Sudoku image
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

# Function to find the main outline of the Sudoku puzzle
def main_outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

# Function to reframe the corners of the Sudoku puzzle
def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new

# Function to split the Sudoku grid into cells
def splitcells(image):
    rows = np.vsplit(image, 9)
    cells = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for cell in cols:
            cells.append(cell)
    return cells

# Function to crop the cells
def CropCell(cells):
    Cells_croped = []
    for image in cells:
        img = np.array(image)
        img = img[4:46, 6:46]
        img = Image.fromarray(img)
        Cells_croped.append(img)
    return Cells_croped

# Function to read digits from cells using the model
def read_cells(cell, model):
    result = []
    for image in cell:
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (32, 32))
        img = img / 255
        img = img.reshape(1, 32, 32, 1)
        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        probabilityValue = np.amax(predictions)
        if probabilityValue > 0.65:
            result.append(classIndex)
        else:
            result.append(0)
    return result

# Sudoku solver functions
def next_box(quiz):
    for row in range(9):
        for col in range(9):
            if quiz[row][col] == 0:
                return (row, col)
    return False

def possible(quiz, row, col, n):
    for i in range(0, 9):
        if quiz[row][i] == n and row != i:
            return False
    for i in range(0, 9):
        if quiz[i][col] == n and col != i:
            return False
    row0 = (row) // 3
    col0 = (col) // 3
    for i in range(row0 * 3, row0 * 3 + 3):
        for j in range(col0 * 3, col0 * 3 + 3):
            if quiz[i][j] == n and (i, j) != (row, col):
                return False
    return True

def solve(quiz):
    val = next_box(quiz)
    if val is False:
        return True
    else:
        row, col = val
        for n in range(1, 10):
            if possible(quiz, row, col, n):
                quiz[row][col] = n
                if solve(quiz):
                    return True
                else:
                    quiz[row][col] = 0
        return

# Function to display Sudoku grid in a table format
def display_grid(grid):
    table = "<table style='border-collapse: collapse; margin: 20px auto;'>"
    for row in range(9):
        table += "<tr>"
        for col in range(9):
            if col % 3 == 0 and col != 0:
                table += "<td style='border-left: 3px solid black;'>"
            else:
                table += "<td>"
            table += f"<div style='text-align: center; padding: 10px; border: 1px solid black;'>{grid[row][col]}</div>"
            table += "</td>"
        table += "</tr>"
        if row % 3 == 2 and row != 8:
            table += "<tr><td colspan='9' style='border-bottom: 3px solid black;'></td></tr>"
    table += "</table>"
    st.markdown(table, unsafe_allow_html=True)

# Streamlit app
st.title("Sudoku Solver")
st.write("Choose an option to solve a Sudoku puzzle:")

# Toggle between image upload and manual input
option = st.radio("Select an option:", ("Upload an Image", "Enter Manually"))

if option == "Upload an Image":
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.write("### Uploaded Image")
        st.image(uploaded_file, caption="Your Sudoku Puzzle", use_container_width=True)

        # Process the image
        with st.spinner("Processing the image..."):
            # Read the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            puzzle = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Resize and preprocess the image
            puzzle = cv2.resize(puzzle, (450, 450))
            su_puzzle = preprocess(puzzle)
            
            # Find the outline of the Sudoku puzzle
            su_contour, hierarchy = cv2.findContours(su_puzzle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            su_biggest, su_maxArea = main_outline(su_contour)
            
            if su_biggest.size != 0:
                su_biggest = reframe(su_biggest)
                su_pts1 = np.float32(su_biggest)
                su_pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
                su_matrix = cv2.getPerspectiveTransform(su_pts1, su_pts2)
                su_imagewrap = cv2.warpPerspective(puzzle, su_matrix, (450, 450))
                su_imagewrap = cv2.cvtColor(su_imagewrap, cv2.COLOR_BGR2GRAY)
                
                # Split the Sudoku grid into cells
                sudoku_cell = splitcells(su_imagewrap)
                sudoku_cell_croped = CropCell(sudoku_cell)
                
                # Read digits from cells
                grid = read_cells(sudoku_cell_croped, model)
                grid = np.asarray(grid)
                grid = np.reshape(grid, (9, 9))
                
                # Display the extracted Sudoku grid
                st.write("### Extracted Sudoku Grid")
                display_grid(grid)
                
                # Solve the Sudoku puzzle
                if solve(grid):
                    st.write("### Solved Sudoku Grid")
                    display_grid(grid)
                else:
                    st.error("Solution doesn't exist. Model misread digits.")
            else:
                st.error("Could not detect Sudoku puzzle in the image.")

else:
    # Manual input option
    st.write("### Enter the Sudoku Puzzle")
    st.write("Fill in the numbers below (use 0 for empty cells):")

    # Create a 9x9 grid for manual input
    grid = np.zeros((9, 9), dtype=int)
    for row in range(9):
        cols = st.columns(9, gap="small")
        for col in range(9):
            with cols[col]:
                grid[row][col] = st.number_input(
                    "",
                    min_value=0,
                    max_value=9,
                    value=0,
                    key=f"cell_{row}_{col}",
                    label_visibility="collapsed",
                )

    # Solve the Sudoku puzzle
    if st.button("Solve"):
        st.write("### Solved Sudoku Grid")
        if solve(grid):
            display_grid(grid, "Solved Grid")
        else:
            st.error("No solution exists for the given puzzle.")