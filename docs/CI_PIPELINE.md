# Practical 9: Continuous Integration Using GitHub Actions

## 1. Aim
To implement Continuous Integration (CI) using GitHub Actions for an Iris Classification MLOps project.

## 2. Objectives
- Automate code compilation and linting.
- Run unit tests automatically whenever code is pushed or a pull request is created.
- Generate test and code coverage reports.
- Verify model quality through automated tests.
- Detect errors early in the development process.

## 3. Tools and Technologies
- Python 3.11
- Git and GitHub
- GitHub Actions
- Pytest
- Flake8
- Pytest-Cov
- Scikit-learn

## 4. Workflow Configuration

The CI workflow is stored at:

`.github/workflows/ci.yml`

The workflow runs automatically on:
- Push events to the `main` and `develop` branches.
- Pull requests targeting the `main` and `develop` branches.

## 5. CI Pipeline Jobs

### Job 1: Build and Lint
- Sets up Python 3.11.
- Installs project dependencies.
- Compiles Python source and test files.
- Checks code quality using Flake8.

### Job 2: Test Suite
- Prepares the Iris dataset.
- Runs data collection, preprocessing, and feature engineering.
- Executes unit tests using Pytest.
- Generates test and coverage reports.
- Uploads reports as GitHub Actions artifacts.

### Job 3: Model Quality Gate
- Runs automated model quality tests.
- Checks model accuracy, F1 score, valid predictions, and prediction count.
- Runs only when the test suite succeeds.

## 6. Test Results

The complete Practical 8 test suite was executed successfully:

**20 tests passed.**

The CI workflow also completed successfully on the main branch, with all three jobs passing.

## 7. CI Failure Demonstration

A separate branch named `ci-failure-demo` was created to demonstrate automatic failure detection.

An intentional failing test was temporarily added:

`assert 1 == 0`

The Pull Request triggered GitHub Actions and produced these results:

| Job | Result |
|---|---|
| Build and Lint | Passed |
| Test Suite | Failed |
| Model Quality Gate | Skipped |

This demonstrated that GitHub Actions detects failing tests and prevents dependent jobs from running.

The temporary branch was deleted after the demonstration. The main branch was not affected.

## 8. Conclusion

Continuous Integration was successfully implemented using GitHub Actions. The pipeline automatically checks code quality, executes unit tests, generates reports, and verifies model quality. The failure demonstration confirmed that the pipeline can detect errors automatically.