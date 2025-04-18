# Urdu Marsiya NER Annotator

A Streamlit application designed for annotating Urdu Marsiya poetry with Named Entity Recognition (NER).

## Overview

This application helps researchers, particularly in digital humanities, to identify, annotate, and verify named entities in Urdu Marsiya poetry. The tool combines AI-powered entity recognition with human verification to ensure accuracy in identifying entities such as people, locations, dates, and more within historical and religious texts.

## Features

### 1. Text Upload and Automated NER Tagging

- **Upload Files**: Submit Urdu text files for analysis
- **Paste Text**: Directly paste Urdu text for immediate processing
- **Browse Existing Files**: Navigate through a collection of Marsiya texts
- **LLM-based Tagging**: Automated entity recognition using AI language models
- **Configurable Settings**: Select different language models and adjust processing parameters

### 2. Manual Review and Correction

- **Interactive Tagged View**: View tagged entities with color highlighting
- **Line-by-Line Review**: Navigate through the text one line at a time
- **Tag Verification**: Confirm or reject AI-suggested entity tags
- **Tag Correction**: Change incorrect entity types
- **Manual Tagging**: Add missing entity tags to words
- **Remove Tags**: Delete incorrectly added tags
- **English Translation**: View English translations of Urdu text for context

### 3. LLM-as-a-Judge Evaluation

- **Multiple Model Comparison**: Compare how different AI models perform on NER tasks
- **Detailed Metrics**: View accuracy, precision, recall, and F1 scores
- **Entity-Type Analysis**: See performance broken down by entity type (PERSON, LOCATION, etc.)
- **Visual Reporting**: Bar charts comparing model performance
- **Contextual Evaluation**: LLMs judge each entity with surrounding sentence context

### 4. Data Export and Statistics

- **Excel Export**: Download annotated data as Excel spreadsheets
- **Stats Dashboard**: View statistics about entity types and verification status
- **Email Results**: Send the annotated data directly via email

## Entity Types

The application recognizes the following entity types:

| Entity Type | Description | Color Code |
|-------------|-------------|------------|
| PERSON | Names of people, prophets, Imams, and historical figures | Light Blue |
| LOCATION | Places, landmarks, and geographical features | Light Green |
| DATE | Dates, Islamic months, and specific days | Light Yellow |
| TIME | Time references and periods | Light Pink |
| ORGANIZATION | Groups, tribes, armies, and institutions | Light Orange |
| DESIGNATION | Titles, honorifics, and roles | Light Gray |
| NUMBER | Numerically significant values | Light Purple |

## How to Use

### Installation and Setup

1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`
4. Log in with your credentials

### Workflow

1. **Upload and Tag Texts**:
   - Select an LLM model for tagging
   - Upload a text file or paste Urdu text
   - Submit for automated NER tagging

2. **Review and Correct Tags**:
   - Navigate through the text line by line
   - Review each tagged entity
   - Mark correct tags as verified
   - Fix incorrect entity types
   - Add missing entity tags

3. **Evaluate Model Performance**:
   - Select models to compare
   - Run the LLM-as-a-Judge evaluation
   - View detailed performance metrics
   - Identify which models perform best for different entity types

4. **Export Results**:
   - Download annotated data as Excel files
   - View statistics for the current file or entire corpus
   - Email results to colleagues

## For Digital Humanities Researchers

This tool is designed specifically for scholars working with historical Urdu texts, particularly Marsiya poetry from the Karbala tradition. You don't need advanced technical knowledge to use this application. The interface guides you through the process of identifying important entities like historical figures, sacred locations, and significant dates.

The AI assistance helps speed up the annotation process, while the human review capabilities ensure scholarly accuracy. This combination makes it possible to efficiently process large volumes of text without sacrificing quality in your research.

## Authentication

The application uses secure authentication to protect research data. Users are assigned specific roles that determine their access level within the application.

## Technical Details

- Built with Streamlit for a simple, interactive interface
- Uses advanced AI language models for entity recognition
- Supports concurrent processing for faster analysis
- Implements a three-stage workflow: automated tagging → human verification → model evaluation
- Data is stored in structured JSON format for easy analysis