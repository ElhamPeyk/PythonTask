from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox as msg
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Float, String, Table, MetaData
import os


class CSVLoader:
    """Base class for loading CSV files."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load(self):
        """Load CSV file into DataFrame."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"The file {self.filepath} does not exist.")
        self.df = pd.read_csv(self.filepath)

    def get_dataframe(self):
        """Return the loaded DataFrame."""
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call load() method first.")
        return self.df


class TrainingDataLoader(CSVLoader):
    """Class for loading training data CSV files."""

    def load(self):
        """Load CSV file and rename columns accordingly."""
        super().load()
        self.df.columns = ['x', 'y1', 'y2', 'y3', 'y4']


class IdealFunctionLoader(CSVLoader):
    """Class for loading ideal function CSV files."""

    def load(self):
        """Load CSV file and rename columns accordingly."""
        super().load()
        self.df.columns = ['x'] + [f'ideal_y{i}' for i in range(1, 51)]


class TestDataLoader(CSVLoader):
    """Class for loading test data CSV files."""

    def load(self):
        """Load CSV file and rename columns accordingly."""
        super().load()
        self.df.columns = ['x', 'y']


class DataProcessor:
    """Class for processing data and performing calculations."""

    def __init__(self, db_folder='DB', db_name='PythonTaskDataBase.db'):
        self.db_folder = db_folder
        self.db_name = db_name
        self._ensure_db_directory()
        self.db_path = os.path.join(self.db_folder, self.db_name)
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        self.metadata = MetaData()

    def _ensure_db_directory(self):
            """Ensure the database directory exists."""
            if not os.path.exists(self.db_folder):
                os.makedirs(self.db_folder)

    def create_tables(self):
        """Create necessary tables in the database."""
        self.training_data = Table('TrainingData', self.metadata,
                                   Column('x', Float, primary_key=True),
                                   Column('y1', Float),
                                   Column('y2', Float),
                                   Column('y3', Float),
                                   Column('y4', Float))

        columns = [Column('x', Float, primary_key=True)]
        columns += [Column(f'ideal_y{i}', Float) for i in range(1, 51)]
        self.ideal_functions = Table('IdealFunctions', self.metadata, *columns)

        self.test_results = Table('TestResults', self.metadata,
                                  Column('x', Float),
                                  Column('y', Float),
                                  Column('ideal_function', String),
                                  Column('deviation', Float))

        self.metadata.create_all(self.engine)

    def save_to_db(self, df, table_name):
        """Save DataFrame to the specified table in the database."""
        df.to_sql(table_name, self.engine, if_exists='replace', index=False)

    def find_best_fit(self, training_df, ideal_df):
        """Find the best fit functions for each training data column."""
        best_fit_functions = {}
        for col in training_df.columns[1:]:
            min_error = float('inf')
            best_fit = None
            for ideal_col in ideal_df.columns[1:]:
                """ It is often used in regression analysis and curve fitting to determine how well 
                a model or function fits a set of data. The goal is usually to minimize the SSE to find the
                 best-fitting model or function.."""
                error = np.sum((training_df[col] - ideal_df[ideal_col]) ** 2)
                if error < min_error:
                    min_error = error
                    best_fit = ideal_col
            best_fit_functions[col] = best_fit
        return best_fit_functions

    def calculate_deviation(self, test_df, ideal_df, best_fit_functions):
        """Calculate deviations for test data based on best fit functions."""
        results = []
        for _, row in test_df.iterrows():
            x_val = row['x']
            y_val = row['y']
            best_fit_col = None
            min_deviation = float('inf')
            for train_col, ideal_col in best_fit_functions.items():
                ideal_values = ideal_df.loc[ideal_df['x'] == x_val, ideal_col].values
                if ideal_values.size > 0:
                    ideal_val = ideal_values[0]
                    deviation = abs(y_val - ideal_val)
                    if deviation < min_deviation:
                        min_deviation = deviation
                        best_fit_col = ideal_col
            results.append([x_val, y_val, best_fit_col, min_deviation])
        return results


def open_file_dialog(entry):
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if file_path:
        entry.delete(0, END)
        entry.insert(END, file_path)


def register():
    if entryPath1.get() == ""  or entryFunction.get() == "" or entryTest.get() == "":
        msg.showinfo("Warning", "Please select all CSV files.")
        return

    filePath1 = entryPath1.get()
    fileFunction = entryFunction.get()
    fileTest = entryTest.get()

    try:
        # Create a SQLite database and tables
        processor = DataProcessor(db_folder='Database', db_name='PythonTaskDataBase.db')
        processor.create_tables()
        # Proceed with other operations...

        # Load training data into the database
        training_loader = TrainingDataLoader(filePath1)
        training_loader.load()
        training_df = training_loader.get_dataframe()
        processor.save_to_db(training_df, 'TrainingData')


        # Load ideal functions data into the database
        ideal_loader = IdealFunctionLoader(fileFunction)
        ideal_loader.load()
        ideal_df = ideal_loader.get_dataframe()
        processor.save_to_db(ideal_df, 'IdealFunctions')

        # Load test data
        test_loader = TestDataLoader(fileTest)
        test_loader.load()
        test_df = test_loader.get_dataframe()

        # Find the best fit functions for each training data column
        best_fit_functions = processor.find_best_fit(training_df, ideal_df)

        # Calculate deviations for test data
        deviations = processor.calculate_deviation(test_df, ideal_df, best_fit_functions)
        deviation_df = pd.DataFrame(deviations, columns=['x', 'y', 'ideal_function', 'deviation'])
        processor.save_to_db(deviation_df, 'TestResults')

        # Print the results
        result_str = "Best fit functions:\n"
        for key, value in best_fit_functions.items():
            result_str += f"The best fit for {key} is {value}\n"
        msg.showinfo("Best Fit Functions", result_str)

        msg.showinfo("Success", "Data successfully loaded into the database and best fit functions identified.")
    except Exception as e:
        msg.showinfo("Error", f"Failed to process CSV files: {str(e)}")


mainObject = Tk()
mainObject.geometry("600x600")
mainObject.resizable(0, 0)
mainObject.title("TaskPython")
mainObject.configure(background="white")

# Training Data Set 1
lblPath1 = Label(mainObject, text="Training DS 1", bg="white")
lblPath1.grid(row=0, column=0, padx=(10, 10), pady=10)
button1 = Button(mainObject, text="...", command=lambda: open_file_dialog(entryPath1))
button1.grid(row=0, column=1, pady=10)
entryPath1 = ttk.Entry(mainObject, width=50)
entryPath1.grid(row=0, column=2, padx=(10, 10), pady=10)

# Function Data Set
lblFunctionPath = Label(mainObject, text="Function DS", bg="white")
lblFunctionPath.grid(row=4, column=0, padx=(10, 10), pady=10)
buttonFunction = Button(mainObject, text="...", command=lambda: open_file_dialog(entryFunction))
buttonFunction.grid(row=4, column=1, pady=10)
entryFunction = ttk.Entry(mainObject, width=50)
entryFunction.grid(row=4, column=2, padx=(10, 10), pady=10)

# Test file
lblTestPath = Label(mainObject, text="Test file", bg="white")
lblTestPath.grid(row=5, column=0, padx=(10, 10), pady=10)
buttonTest = Button(mainObject, text="...", command=lambda: open_file_dialog(entryTest))
buttonTest.grid(row=5, column=1, pady=10)
entryTest = ttk.Entry(mainObject, width=50)
entryTest.grid(row=5, column=2, padx=(10, 10), pady=10)

btnRegister = Button(mainObject, text="Confirm", command=register)
btnRegister.grid(row=6, column=1, columnspan=2, padx=(10, 10), pady=20)

if __name__ == "__main__":
    mainObject.mainloop()

