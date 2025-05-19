# Snowflake Retail Recommender

A complete product recommendation system built on Snowflake using retail transaction data and machine learning.

## Overview

This project implements an end-to-end recommendation system for retail products using Snowflake's data platform and machine learning capabilities. It analyzes customer purchase patterns to generate personalized product recommendations using Bayesian Personalized Ranking (BPR).

## Features

- **Data Ingestion**: Load retail transaction data from CSV or Excel into Snowflake
- **Data Preprocessing**: Clean and transform retail data for ML readability
- **Exploratory Analysis**: Visualize purchase patterns and customer segments
- **Feature Engineering**: Create user-item matrices for recommendation algorithms
- **Model Training**: Implement BPR algorithm with hyperparameter tuning
- **Evaluation Metrics**: Calculate precision, recall, and catalog coverage
- **Visualization**: Interactive dashboards for performance insights
- **Recommendation API**: Generate personalized product recommendations for customers

## Architecture

1. **Data Storage**: Snowflake data warehouse
2. **Processing**: Snowpark Python for data transformation
3. **Machine Learning**: Snowflake ML for model training and inference
4. **Visualization**: Matplotlib and Seaborn for insights

## Getting Started

### Prerequisites
- Snowflake account with Snowpark enabled
- Python 3.8+ with required packages

### Installation

1. Clone the repository:
```bash
git clone https://github.com/username/snowflake-retail-recommender.git
cd snowflake-retail-recommender
```

2. Configure your Snowflake connection in the notebook or environment variables.

3. Run the setup script to create necessary Snowflake resources:
```python
# Run the setup cells in the notebook
```

4. Load sample data or connect your own retail dataset.

## Usage

### Building a Recommendation Model

```python
# Run the full pipeline
recommendation_system = build_recommendation_system(online_retail_data)
```

### Generating Recommendations

```python
# Get recommendations for a specific customer
customer_id = '13085'
recommendations = recommend_for_customer(customer_id, recommendation_system, online_retail_data)
```

## Dataset

This project uses the [Online Retail II dataset](http://archive.ics.uci.edu/ml/datasets/Online+Retail+II) from the UCI Machine Learning Repository, containing transaction data from a UK online retailer between 2009 and 2011.

## Performance

- Precision@10: 0.081
- Recall@10: 0.053
- Catalog Coverage: 60.7%

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the dataset
- Snowflake for the data platform and ML capabilities
