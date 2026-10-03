# Remember Me

A hackathon prototype for a dementia-assistance application.

# What problem am I solving with the idea?

People living with dementia may struggle to recognize familiar faces, which can cause confusion, anxiety, and loss of independence.

# What is my solution?

My solution is a face recognition assistant designed to work with smart glasses, something like Meta Ray-Ban Display smart glasses. When the user looks at someone and activates the system, it scans the person’s face, checks whether they are a registered familiar person, and displays their name and relationship to the user. The goal is to provide quick, discreet memory cues during everyday interactions.

## Requirements

- Python 3.11
- uv

## Installation

Install the project dependencies:

```bash
uv sync
```

## Running the application

```bash
uv run uvicorn remember_me.main:app --reload
```
