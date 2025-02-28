---
title: Multi-Agentic SQL Generator
sdk: gradio
app_file: app.py
license: apache-2.0
tags:
- sql
- openai
- multi-agent
- demo
sdk_version: 5.19.0
---

# Multi-Agentic SQL Generator

The Multi-Agentic SQL Generator is a modular, multi-step system that translates natural language queries into SQL queries, validates and optimizes them, and then executes them against a SQLite database. The system leverages OpenAI's language models (via a LangGraph workflow) for query understanding, validation, and optimization. It also provides evaluation capabilities using RAGAS metrics (such as context precision and context recall) to assess performance and quality.

## Capabilities

- **Natural Language Query Understanding:**  
  Converts natural language queries into structured SQL metadata according to a predefined database schema.

- **Query Validation:**  
  Checks the generated SQL for syntax errors and security risks (e.g., harmful operations like `DROP`, `DELETE`).

- **Query Optimization:**  
  Optimizes SQL queries for performance, ensuring only the necessary columns, joins, and filtering conditions are included.

- **SQL Execution:**  
  Executes the optimized SQL query against a SQLite database and returns the results.

- **Evaluation with RAGAS Metrics:**  
  Evaluates the generated output using metrics like:
  - **Faithfulness:** (0.7500) Faithfulness measures how well the generated SQL execution results match the expected (ground truth) results.
  - **Answer Relevancy:** (0.2280) Answer relevancy measures how well the generated SQL results match the intent of the user's query.
  - **Context Precision:** (0.3000) Context Precision measures how much of the generated SQL execution result is actually relevant to the query.
  - **Context Recall:** (0.3500) Context Recall measures how much of the expected result was actually retrieved.

- **Extensibility and Deployment:**  
  Easily integrable with front-end frameworks (e.g., Chainlit) and deployable on platforms like Hugging Face Spaces.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/multi-agent-sql-generator.git
   cd multi-agent-sql-generator
