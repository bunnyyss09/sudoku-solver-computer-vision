# Sudoku Puzzle Solver

This project is designed to solve Sudoku puzzles from images or manual input. It utilizes neural networks for digit recognition and a backtracking algorithm to solve the puzzle.

---

## Sample Video

---
## Project Breakdown

### **Part 1: Digit Classification**
- Trains a neural network using the **Chars74K** dataset for digit classification.
- The model is used to recognize digits extracted from the Sudoku puzzle grid.

### **Part 2: Sudoku Image Detection and Reading**
- Identifies the Sudoku puzzle in an uploaded image using **OpenCV** for contour detection.
- Crops and classifies each cell in the grid, extracting the numbers to construct the initial puzzle array.

### **Part 3: Puzzle Solving**
- Uses recursion and a backtracking algorithm to solve the puzzle efficiently.
- Displays both the extracted and solved Sudoku grids in a user-friendly table format.

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/sudoku-puzzle-solver.git
   ```

2. **Install dependencies**:
   Ensure you have Python 3.x installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

---

## Application Features

1. **Upload Sudoku Puzzle Image**:
   - Upload an image of an unsolved Sudoku puzzle.
   - The app will detect the grid, classify the digits, and solve the puzzle.
   
2. **Manual Sudoku Input**:
   - Option to manually enter a Sudoku puzzle for solving.
   - Input a 9x9 grid using Streamlit's number input fields.

---

## Dependencies

- **Python 3.x**
- **Streamlit**: For building the web interface.
- **OpenCV**: For image processing and contour detection.
- **NumPy**: For array manipulation.
- **TensorFlow/Keras**: For digit classification using the trained model.
- **PIL**: For image handling and preprocessing.

---

## Model Training Details
- The digit classification model is trained on the **Chars74K** dataset, which contains various handwritten characters.
- The model predicts digits with high confidence to construct the initial Sudoku puzzle.

---

## Future Improvements
- **Enhanced Accuracy**: Improve the digit classification model with additional training data or more advanced architectures.
- **Mobile Compatibility**: Develop a mobile-friendly version of the app for easier access.
- **Additional Puzzle Types**: Extend support to other grid-based puzzles like KenKen or Kakuro.