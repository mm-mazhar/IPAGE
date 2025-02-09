## IPAGE | Omdena | Soil Nutrient Prediction for Enhanced Fertilizer Recommendations


### Project Description

This project aimed to develop a predictive model system for soil organic carbon (SOC), boron, and zinc levels using existing macronutrient and physicochemical data from soil sensors. The goal was to optimize fertilizer recommendations, improve agricultural productivity, and promote economic efficiency for smallholder farmers. We developed and tested various machine learning models, ultimately selecting Ridge Regression for its balance between performance and generalizability. Our results demonstrated an improvement in soil nutrient predictions, supporting sustainable and precision agriculture.

## `EDA`

<table style="width:100%" align="center">
  <tr>
    <td><img src="https://i.imgur.com/apY8TQV.png" width="200px" height=100px/></td>
    <td><img src="https://i.imgur.com/t9nS6qy.png" width="200px" height=100px/></td>
  </tr>
</table>

<table style="width:100%" align="center">
  <tr>
    <td><img src="https://i.imgur.com/Nkobwci.png" width="200px" height=100px/></td>
    <td><img src="https://i.imgur.com/yPc4tlO.png" width="200px" height=100px/></td>
  </tr>
</table>

## `Data Preparation`

<table style="width:100%" align="center">
  <tr>
    <td><img src="https://i.imgur.com/vrZsMGm.png" width="200px" height=100px/></td>
    <td><img src="https://i.imgur.com/YHbGNFY.png" width="200px" height=100px/></td>
    <td><img src="https://i.imgur.com/z6VOFK5.png" width="200px" height=100px/></td>
    <td><img src="https://i.imgur.com/fWfEBqm.png" width="200px" height=100px/></td>
  </tr>
</table>

## `Modeling`

<table style="width:100%" align="center">
  <tr>
    <td><img src="https://i.imgur.com/N3ZZH83.png" width="200px" height=100px/></td>
    <td><img src="https://i.imgur.com/AuG7PwW.png" width="200px" height=100px/></td>
  </tr>
</table>

<table style="width:100%" align="center">
  <tr>
    <td><img src="https://i.imgur.com/zjBOnFp.png" width="200px" height=100px/></td>
  </tr>
</table>

### Base Models Considered

A variety of machine learning models were explored to predict soil properties:
- Random Forest Regressor
- CatBoost Regressor
- Ridge Regression
- TPOT Classifier


The modeling process involved:
- Identifying overfitting in complex models and addressing it through regularization techniques.
- Selecting Ridge Regression as the final model for its balance between interpretability and accuracy.


### Overfitting Analysis

#### Observations:

<table style="width:100%" align="center">
  <tr>
    <td><img src="https://i.imgur.com/zjBOnFp.png" width="200px" height=100px/></td>
  </tr>
</table>

-   Complex models (e.g., Random Forest, CatBoost) exhibited high R² on training data but lower R² on test data, indicating overfitting.
- Ridge Regression provided better generalization on unseen data.

#### Possible Causes:
- Limited training data from specific regions resulted in poor generalization.
- Overly complex models captured noise rather than true patterns.
- Dataset imbalance led to biased predictions favoring dominant regions.

#### Mitigation Strategies
- Implemented cross-validation and regularization techniques (Ridge Regression) to prevent overfitting.
- Applied feature selection to remove redundant variables and improve model efficiency.
- Preferred linear models for their interpretability and resilience to overfitting.

<table style="width:100%" align="center">
  <tr>
    <td><img src="https://i.imgur.com/G6DeUjA.png" width="200px" height=100px/></td>
    <td><img src="https://i.imgur.com/x61ucP9.png" width="200px" height=100px/></td>
    <td><img src="https://i.imgur.com/iHmcjXr.png" width="200px" height=100px/></td>
  </tr>
</table>


## `API`

FastAPI-based API designed to intelligently analyze and manage soil data. It provides endpoints for data creation, retrieval, model training, and inference to predict soil properties like SOC, Boron, and Zinc. The project uses a combination of data preprocessing techniques, machine learning models (Ridge regression), and database storage (SQLite) to provide valuable insights for agricultural applications.

### Key Features of API

*   **Data Management:** Endpoints for creating and retrieving soil data records.
*   **Model Training:** Functionality to train machine learning models (Ridge Regression) on soil data.
*   **Prediction (Inference):** API endpoints for predicting soil properties based on input data.
*   **Data Preprocessing:** Implements data cleaning and transformation for optimized model training.
*   **SQLite Database:** Uses an SQLite database for data persistence and management.

### API Endpoints

The API provides the following endpoints:

*   **GET `/`**: Returns a basic HTML response indicating the API is running.
*   **POST `/data`**: Creates a new soil data record. Takes a JSON payload conforming to the `RawData` schema.
    *   **Example Request:**
        ```json
        {
          "Area": "Sample Area",
          "pH": 6.5,
          "Nitrogen": 0.1,
          "Phosphorus": 0.2,
          "Potassium": 0.3,
          "Sulfur": 0.05,
          "Boron": 0.01,
          "Zinc": 0.005,
          "Sand": 40,
          "Silt": 30,
          "Clay": 30
        }
        ```
    *   **Returns:** A JSON response indicating success or failure and the created data.

*   **GET `/data?limit={limit}`**: Retrieves soil data records.  Optionally takes a `limit` query parameter to specify the number of records to retrieve.
    *   **Example Request:**
        ```
        GET /data?limit=10
        ```
    *   **Returns:** A JSON array of soil data objects.

*   **POST `/train?target={target}`**:  Trains a machine learning model (Ridge Regression) to predict a specific soil property.  The `target` query parameter specifies the property to predict (e.g., `SOC`, `Boron`, `Zinc`).
    *   **Example Request:**
        ```
        POST /train?target=SOC
        ```
    *   **Returns:** A JSON response indicating success and model metrics.

*   **POST `/inference/point/`**: Predict soil properties from a JSON payload.  The body needs to conform to `PredictionInput`. The `targets` Query parameter specifies the output field (e.g., `SOC`, `Boron`, `Zinc`).
    *   **Example Request:**
        ```
        POST /inference/point/?target=SOC&target=Boron
        {
          "Area": "Sample Area",
          "pH": 6.5,
          "Nitrogen": 0.1,
          "Phosphorus": 0.2,
          "Potassium": 0.3,
          "Sulfur": 0.05,
          "Boron": 0.01,
          "Zinc": 0.005,
          "Sand": 40,
          "Silt": 30,
          "Clay": 30
        }
        ```
    *   **Returns:** A JSON payload with the prediction value.
* **POST `/inference/batch/`**: Predict soil properties from a CSV file (batch)
    *   **Example request**
    ```
    curl -X POST -H "Content-Type: multipart/form-data" \
  -F file=@test_version_merged_v3.csv \
  "http://localhost:8080/inference/batch/?targets=SOC&targets=Boron"
    ```

<table style="width:100%" align="center">
  <tr>
    <td><img src="https://i.imgur.com/xe3LHrO.png" width="200px" height=100px/></td>
  </tr>
</table>

## `Streamlit APP`

This Streamlit application provides a user-friendly interface to visualize and analyze soil data. It complements the API by offering interactive dashboards and data exploration tools for agricultural insights.

### Key Features of Streamlit APP

*   **Data Overview:** Displays a Sweetviz report for a comprehensive overview of the dataset.
*   **Interactive Dashboards:**  Enables users to create dynamic visualizations and explore data patterns through selectable numeric and categorical features.
*   **Geospatial Visualization:**  Showcases data geographically on an interactive map (built with Folium), allowing users to identify spatial trends.
*   **Feature Analysis:** Offers a detailed feature analysis with distribution plots, scatter plots, and other comparison visuals.
*   **Interactive Filtering:**  Includes customizable sidebar panels with various data, style and model controls.
*   **Single and Batch Predictions:**  Supports both single datapoint model prediction for immediate feedback and batch prediction for larger scale analysis of the datasets

<table style="width:100%" align="center">
  <tr>
    <td><img src="https://i.imgur.com/fqFQPN7.png" width="200px" height=100px/></td>
    <td><img src="https://i.imgur.com/wFV6JJC.png" width="200px" height=100px/></td>
  </tr>
</table>

## `Local Execution`

If every thing is setup, then you can run the project via
- open CLI
- cd into [PATH: where you stored the project]
- `poetry run uvicorn src.api.app:app --reload --port 8002`
- Open Another window of CLI and run
- `poetry run streamlit run ./src/app/app.py`

For first time setup please follow the instruction below


## `Setup and Local Execution`

1.  **Prerequisites:**
    *   Python 3.7+
        *   **Download Python:** Download the specific Python 3.11.9 version from the official [Python website](https://www.python.org/downloads/).

        *   **Run the Installer:**
            *   **Windows:** Run the downloaded `.exe` installer. **Important:** During installation, make sure to check the box that says "Add Python to PATH". This will allow you to run Python commands from your command prompt or PowerShell.
            *   **macOS:** Run the downloaded `.pkg` installer. Python may already be installed on your system, but this will ensure you have the latest version and all necessary components.
            *   **Linux:** Python is typically pre-installed on Linux systems. You can update or install Python using your distribution's package manager (e.g., `apt`, `yum`, `dnf`). For example, on Debian/Ubuntu:

                ```bash
                sudo apt update
                sudo apt install python3 python3-dev
                ```

        *   **Verify Installation:** Open a new command prompt or terminal and type:

            ```bash
            python --version
            ```

        This should print the Python version you installed (e.g., `Python 3.11.9`).

    *   [Poetry](https://python-poetry.org/) (for dependency management)

        *   **Installing Poetry:** Poetry is a tool for dependency management and packaging in Python. It allows you to declare the libraries your project depends on and will manage the installation/updating of them for you.

          *   **Windows:**

              1.  **Open PowerShell (as Administrator):** Right-click on the Start button and select "Windows PowerShell (Admin)" or "Terminal (Admin)".

              2.  **Run the Installer:** Use the following command in PowerShell:

                  ```powershell
                  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
                  ```

                  This command downloads and executes the official Poetry installer.  The `-UseBasicParsing` argument is important for compatibility with PowerShell.

              3.  **Add Poetry to Your PATH (if not automatic):** The installer *should* automatically add Poetry to your system's PATH. If it doesn't, you'll need to do it manually:

                  *   Search for "Edit the system environment variables" in the Start menu and open it.
                  *   Click "Environment Variables".
                  *   In the "System variables" section, find the `Path` variable and select it, and then click "Edit".
                  *   Click "New" and add the path to Poetry's `bin` directory.  The default location is typically:
                      ```
                      %APPDATA%\Python\Scripts
                      ```
                  *   Click "OK" on all dialogs to save the changes.

              4.  **Verify Installation:** Close and reopen your PowerShell, and then type:

                  ```powershell
                  poetry --version
                  ```

                  This should print the Poetry version you installed.

          *   **macOS:**

              1.  **Open Terminal:** Open the Terminal application.

              2.  **Run the Installer:**

                  ```bash
                  curl -sSL https://install.python-poetry.org | python3 -
                  ```

                  This command downloads and executes the official Poetry installer.

              3.  **Add Poetry to Your PATH (if not automatic):** The installer *should* automatically add Poetry to your `~/.zshrc` or `~/.bashrc` file. If not, you'll need to add it manually. To locate where Poetry has installed it, you may run:

                  ```bash
                  poetry env info
                  ```
                  This should output the base python version for your poetry install, after finding a base location for you to install.

              4.   **Open your shell configuration file:** This will typically either be `~/.zshrc` (if you're using Zsh) or `~/.bashrc` (if you're using Bash). You can edit it with a text editor like `nano` or `vim`. For example:

                      ```
                      nano ~/.zshrc
                      ```

              5.  **Add the Poetry Path:** Add the following line to the end of the file, replacing `[POETRY_HOME]` with the actual path you found, and add the following to the bottom of the file:
                  ```
                  export PATH="$PATH:[POETRY_HOME]/bin"
                  ```
              6.  **Save the file and reload your shell:** Run the following command to reload your shell configuration:
                  ```
                  source ~/.zshrc   # If you're using Zsh
                  ```
                  or
                  ```
                  source ~/.bashrc   # If you're using Bash
                  ```

              7.  **Verify Installation:** Open a new Terminal window and type:

                  ```bash
                  poetry --version
                  ```

                  This should print the Poetry version you installed.

          *   **Linux:**

              1.  **Open Terminal:** Open your terminal application.

              2.  **Run the Installer:**

                  ```bash
                  curl -sSL https://install.python-poetry.org | python3 -
                  ```

                  This command downloads and executes the official Poetry installer.

              3.  **Add Poetry to Your PATH (if not automatic):**  Similar to macOS, the installer *should* automatically add Poetry to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`). If you can't run `poetry` after the installation, follow these steps:

                    To locate where Poetry has installed it, you may run:

                  ```bash
                  poetry env info
                  ```

                  This should output the base python version for your poetry install, after finding a base location for you to install.

              4.  **Open your shell configuration file:** Typically `~/.bashrc` or `~/.zshrc`.
                  ```bash
                  nano ~/.bashrc   # If you're using Bash
                  ```
                  or
                  ```bash
                  nano ~/.zshrc   # If you're using Zsh
                  ```
              5.  **Add the Poetry Path:** Add the following line to the end of the file, replacing `[POETRY_HOME]` with the actual path you found. Make sure to put in the location of the `.poetry` directory on your system:
                  ```
                  export PATH="$PATH:[POETRY_HOME]/bin"
                  ```
              6.  **Save the file and reload your shell:**
                  ```bash
                  source ~/.bashrc   # If you're using Bash
                  ```
                  or
                  ```bash
                  source ~/.zshrc   # If you're using Zsh
                  ```

              7.  **Verify Installation:** Open a new Terminal window and type:

                  ```bash
                  poetry --version
                  ```

                  This should print the Poetry version you installed.

              **General Notes:**

              *   **Restart Terminal/Command Prompt:** Always close and reopen your terminal or command prompt after installing Python or modifying PATH variables. This ensures that the changes are recognized by your system.
              *   **Permissions:** Some installation steps (especially on Linux/macOS) may require administrator privileges (`sudo`).
              *   **Alternative Installation Methods:**  There are alternative ways to install Poetry, such as using `pip` (although this is generally not recommended) or using package managers like `conda`. Consult the Poetry documentation for details: [https://python-poetry.org/docs/](https://python-poetry.org/docs/)

              These updated instructions should provide clear and platform-specific guidance for installing Poetry.


2.  **Clone the Repository (if applicable):**
    ```bash
    - git clone https://github.com/OmdenaAI/ipage
    ```

3.  **Install Dependencies:**

    *   First, install the project dependencies
    ```bash
    - poetry install
    ```

   if there any specific system dependencies for utils and other dependencies refer to the readme files for those specific utils

4.  **Configure Environment Variables:**
    
    * Ensure that you can access database . the default is using sqlLite

        *   You can do the following:
            *   make [.env file](src/api/.env)

        *   The following variables needs to be defined in .env file

        ```
        DATABASE = "./src/api/ipage.db"
        DRIVERNAME = "sqlite"
        ```

5.  **Run the API:**

    * cd into ./src/api and run the API:

    ```bash
    - Windows (Command Prompt - cmd)
        - poetry run uvicorn src.api.main:app --reload --port 8080
    - Windows (PowerShell)
        - poetry run uvicorn src.api.main:app --reload --port 8080
    - Linux / macOS (Bash, Zsh)
        - poetry run uvicorn src.api.main:app --reload --port 8080
    ```

    *   This will start the FastAPI server.  The `--reload` flag enables automatic reloading upon code changes.

    * Select the local server
      <table style="width:100%" align="center">
        <tr>
          <td><img src="https://i.imgur.com/zJ6B1Uk.png" width="200px" height=100px/></td>
        </tr>
      </table>
6.  **Access the API Documentation:**

    *   Open your web browser and navigate to [http://localhost:8003/docs](http://localhost:8003/docs) 
    or 
    [http://127.0.0.1:8003/docs](http://127.0.0.1:8003/docs) 
    
        (or the appropriate address based on your `FASTAPI_PORT` configuration).
    
    *   You should see the Swagger UI documentation, which allows you to interact with the API endpoints.


7.  **Run Streamlit APP:**

    ```
    poetry run streamlit run ./src/app/app.py
    ```




## `Contribution Guidelines`

- Have a Look at the [project structure](#project-structure) and [folder overview](#folder-overview) below to understand where to store/upload your contribution
- If you're creating a task, Go to the task folder and create a new folder with the below naming convention and add a README.md with task details and goals to help other contributors understand
  - Task Folder Naming Convention : _task-n-taskname.(n is the task number)_ ex: task-1-data-analysis, task-2-model-deployment etc.
  - Create a README.md with a table containing information table about all contributions for the task.
- If you're contributing for a task, please make sure to store in relavant location and update the README.md information table with your contribution details.
- Make sure your File names(jupyter notebooks, python files, data sheet file names etc) has proper naming to help others in easily identifing them.
- Please restrict yourself from creating unnessesary folders other than in 'tasks' folder (as above mentioned naming convention) to avoid confusion.



## `Folders Overview`

- Original - Folder Containing old/completed Omdena challenge code.
- Reports - Folder to store all Final Reports of this project
- API - Folder to store FastAPI
- App - Folder to store Streamlit App code
- Data - Folder to Store all the data collected and used for this project
- Docs - Folder for Task documentations, Meeting Presentations and task Workflow Documents and Diagrams.
- References - Folder to store any referneced code/research papers and other useful documents used for this project
- Tasks - Master folder for all tasks
  - All Task Folder names should follow specific naming convention
  - All Task folder names should be in chronologial order (from 1 to n)
  - All Task folders should have a README.md file with task Details and task goals along with an info table containing all code/notebook files with their links and information
  - Update the task-table whenever a task is created and explain the purpose and goals of the task to others.
- Visualization - Folder to store dashboards, analysis and visualization reports
- Results - Folder to store final analysis modelling results for the project.
