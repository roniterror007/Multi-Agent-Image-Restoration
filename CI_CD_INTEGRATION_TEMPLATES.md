# CI/CD Integration Templates

## GitHub Actions (.github/workflows/test.yml)

```yaml
name: QA Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
    
    - name: Run pytest - Math Edge Cases
      run: pytest tests/test_math_edge_cases.py -v --tb=short
    
    - name: Run pytest - Segmentation/VLM
      run: pytest tests/test_segmentation_vlm_robustness.py -v --tb=short
    
    - name: Run pytest - E2E Integration
      run: pytest tests/test_pipeline_e2e.py -v --tb=short
    
    - name: Run pytest - Full Suite with Coverage
      run: pytest tests/ -v --cov=. --cov-report=xml --cov-report=term
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  memory-test:
    runs-on: ubuntu-latest
    # Run memory tests on separate job (they're slower)
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
    
    - name: Run memory stress tests
      run: pytest tests/test_memory_stress.py -v --tb=short -m memory

  lint:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install linters
      run: |
        pip install pylint flake8 pytest
    
    - name: Lint with flake8
      run: |
        flake8 batch_process.py agent_controller.py feedback_store.py --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 batch_process.py agent_controller.py feedback_store.py --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

---

## GitLab CI (.gitlab-ci.yml)

```yaml
stages:
  - test
  - coverage
  - performance

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip
    - venv/

before_script:
  - python -V
  - pip install virtualenv
  - virtualenv venv
  - source venv/bin/activate
  - pip install -r requirements-test.txt

test:edge_cases:
  stage: test
  script:
    - pytest tests/test_math_edge_cases.py -v --tb=short
  artifacts:
    reports:
      junit: report.xml
    paths:
      - htmlcov/
    expire_in: 1 week

test:segmentation_vlm:
  stage: test
  script:
    - pytest tests/test_segmentation_vlm_robustness.py -v --tb=short
  artifacts:
    reports:
      junit: report_seg.xml

test:e2e:
  stage: test
  script:
    - pytest tests/test_pipeline_e2e.py -v --tb=short
  artifacts:
    reports:
      junit: report_e2e.xml

test:memory:
  stage: test
  script:
    - pytest tests/test_memory_stress.py -v --tb=short
  artifacts:
    reports:
      junit: report_mem.xml
  only:
    - merge_requests
    - main

test:full:
  stage: test
  script:
    - pytest tests/ -v --cov=. --cov-report=html --cov-report=term --cov-report=xml
  artifacts:
    paths:
      - htmlcov/
      - coverage.xml
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    expire_in: 30 days

performance:
  stage: performance
  script:
    - pytest tests/test_memory_stress.py -v --benchmark-only
  artifacts:
    paths:
      - .benchmarks/
    expire_in: 30 days
  only:
    - merge_requests
```

---

## Jenkins Pipeline (Jenkinsfile)

```groovy
pipeline {
    agent any
    
    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 30, unit: 'MINUTES')
        timestamps()
    }
    
    stages {
        stage('Setup') {
            steps {
                script {
                    sh '''
                        python -m venv venv
                        . venv/bin/activate
                        pip install -r requirements-test.txt
                    '''
                }
            }
        }
        
        stage('Lint') {
            steps {
                script {
                    sh '''
                        . venv/bin/activate
                        flake8 batch_process.py agent_controller.py || true
                    '''
                }
            }
        }
        
        stage('Test - Edge Cases') {
            steps {
                script {
                    sh '''
                        . venv/bin/activate
                        pytest tests/test_math_edge_cases.py -v --junitxml=results/edge_cases.xml
                    '''
                }
            }
        }
        
        stage('Test - Segmentation/VLM') {
            steps {
                script {
                    sh '''
                        . venv/bin/activate
                        pytest tests/test_segmentation_vlm_robustness.py -v --junitxml=results/seg_vlm.xml
                    '''
                }
            }
        }
        
        stage('Test - E2E') {
            steps {
                script {
                    sh '''
                        . venv/bin/activate
                        pytest tests/test_pipeline_e2e.py -v --junitxml=results/e2e.xml
                    '''
                }
            }
        }
        
        stage('Test - Memory (Optional)') {
            when {
                branch 'main'
            }
            steps {
                script {
                    sh '''
                        . venv/bin/activate
                        pytest tests/test_memory_stress.py -v --junitxml=results/memory.xml
                    '''
                }
            }
        }
        
        stage('Coverage') {
            steps {
                script {
                    sh '''
                        . venv/bin/activate
                        pytest tests/ --cov=. --cov-report=html --cov-report=xml
                    '''
                }
                publishHTML(target: [
                    reportDir: 'htmlcov',
                    reportFiles: 'index.html',
                    reportName: 'Coverage Report'
                ])
            }
        }
    }
    
    post {
        always {
            junit 'results/*.xml'
            step([$class: 'CoberturaPublisher',
                coberturaReportFile: 'coverage.xml'
            ])
        }
        success {
            echo 'All tests passed!'
        }
        failure {
            echo 'Tests failed - review logs'
        }
    }
}
```

---

## Local Pre-Commit Hook (.git/hooks/pre-commit)

```bash
#!/bin/bash
# Pre-commit hook: Run quick tests before committing

set -e

echo "Running pre-commit checks..."

# Math edge cases only
echo "Running edge case tests..."
pytest tests/test_math_edge_cases.py -q

if [ $? -ne 0 ]; then
    echo "❌ Tests failed - commit blocked"
    exit 1
fi

echo "✅ Pre-commit checks passed"
exit 0
```

**Install:**
```bash
chmod +x .git/hooks/pre-commit
```

---

## Makefile (Local Development)

```makefile
.PHONY: test test-fast test-full test-coverage test-memory clean install lint

install:
	pip install -r requirements-test.txt

lint:
	flake8 batch_process.py agent_controller.py feedback_store.py
	pylint batch_process.py agent_controller.py feedback_store.py || true

test-fast:
	pytest tests/test_math_edge_cases.py -v --tb=short

test-seg-vlm:
	pytest tests/test_segmentation_vlm_robustness.py -v --tb=short

test-e2e:
	pytest tests/test_pipeline_e2e.py -v --tb=short

test-full:
	pytest tests/ -v --tb=short

test-memory:
	pytest tests/test_memory_stress.py -v --tb=short -m memory

test-coverage:
	pytest tests/ --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

test-benchmark:
	pytest tests/test_memory_stress.py -v --benchmark-only

clean:
	rm -rf .pytest_cache __pycache__ htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "Available targets:"
	@echo "  make install         - Install test dependencies"
	@echo "  make lint            - Run linters"
	@echo "  make test-fast       - Run edge case tests only"
	@echo "  make test-seg-vlm    - Run segmentation/VLM tests"
	@echo "  make test-e2e        - Run E2E tests"
	@echo "  make test-full       - Run all tests"
	@echo "  make test-memory     - Run memory stress tests"
	@echo "  make test-coverage   - Run with coverage report"
	@echo "  make test-benchmark  - Run performance benchmarks"
	@echo "  make clean           - Clean build artifacts"
```

**Usage:**
```bash
make install
make test-fast
make test-coverage
```

---

## requirements-test.txt

```
pytest>=7.0.0
pytest-benchmark>=4.0.0
pytest-timeout>=2.1.0
pytest-cov>=4.0.0
pytest-xdist>=3.0.0
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
Pillow>=9.0.0
opencv-python>=4.5.0
imagehash>=4.2.0
psutil>=5.9.0
```

---

## Docker (docker/Dockerfile)

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

COPY requirements-test.txt .
RUN apt-get update && apt-get install -y python3.10 python3-pip
RUN pip install -r requirements-test.txt

COPY . .

CMD ["pytest", "tests/", "-v"]
```

**Usage:**
```bash
docker build -f docker/Dockerfile -t qa-tests .
docker run --gpus all qa-tests pytest tests/ -v
```

---

## Azure Pipelines (azure-pipelines.yml)

```yaml
trigger:
  - main
  - develop

pr:
  - main

pool:
  vmImage: 'ubuntu-latest'

strategy:
  matrix:
    Python39:
      python.version: '3.9'
    Python310:
      python.version: '3.10'
    Python311:
      python.version: '3.11'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '$(python.version)'
  displayName: 'Use Python $(python.version)'

- script: |
    python -m pip install --upgrade pip
    pip install -r requirements-test.txt
  displayName: 'Install dependencies'

- script: |
    pytest tests/test_math_edge_cases.py -v --junitxml=junit/edge_cases.xml
  displayName: 'Test - Edge Cases'

- script: |
    pytest tests/test_segmentation_vlm_robustness.py -v --junitxml=junit/seg_vlm.xml
  displayName: 'Test - Segmentation/VLM'

- script: |
    pytest tests/test_pipeline_e2e.py -v --junitxml=junit/e2e.xml
  displayName: 'Test - E2E'

- script: |
    pytest tests/ --cov=. --cov-report=html --cov-report=xml --junitxml=junit/all.xml
  displayName: 'Test - Full Coverage'

- task: PublishTestResults@2
  condition: succeededOrFailed()
  inputs:
    testResultsFiles: '**/junit/*.xml'
    testRunTitle: 'Python $(python.version) Tests'

- task: PublishCodeCoverageResults@1
  inputs:
    codeCoverageTool: Cobertura
    summaryFileLocation: '$(System.DefaultWorkingDirectory)/coverage.xml'
    reportDirectory: '$(System.DefaultWorkingDirectory)/htmlcov'
```

---

## Recommended CI/CD Strategy

### For Pull Requests
```bash
# Fast path - skip memory tests
pytest tests/ -m "not memory" -v --tb=short
# Time: ~15 seconds
# Cost: Low
```

### For Main Branch
```bash
# Full validation
pytest tests/ -v --cov=. --cov-report=xml
# Time: ~45-60 seconds
# Cost: Medium
```

### Nightly/Weekly
```bash
# Complete with benchmarks
pytest tests/ -v --cov=. --benchmark-only
# Time: ~2-3 minutes
# Cost: High
```

### Deployment (Production)
```bash
# Minimal check on deployment machine
pytest tests/test_pipeline_e2e.py -v --tb=short
# Time: ~20 seconds
# Cost: Low
# Location: Production RTX A2000
```

---

## Environment Variables

### GitHub Actions
```yaml
env:
  CUDA_VISIBLE_DEVICES: 0
  PYTHONUNBUFFERED: 1
  PYTEST_TIMEOUT: 300
```

### GitLab CI
```yaml
variables:
  CUDA_VISIBLE_DEVICES: "0"
  PYTHONUNBUFFERED: "1"
```

### Jenkins
```groovy
environment {
    CUDA_VISIBLE_DEVICES = '0'
    PYTHONUNBUFFERED = '1'
}
```

---

## Health Check Scripts

### test-status.sh
```bash
#!/bin/bash
echo "QA Test Suite Status Check"
echo "============================"

cd "$(dirname "$0")"

# Count tests
TOTAL=$(grep -r "def test_" tests/ | wc -l)
echo "Total tests: $TOTAL"

# Run quick check
pytest tests/test_math_edge_cases.py -q --tb=no

if [ $? -eq 0 ]; then
    echo "✅ Quick check passed"
else
    echo "❌ Quick check failed"
    exit 1
fi
```

---

**Choose the CI/CD system that best fits your workflow and deployment strategy.**

