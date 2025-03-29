# UAS Last-Mile Delivery Simulator

A Streamlit web application for simulating last-mile delivery demand for Unmanned Aerial Systems (UAS).

## Overview

This application provides a user-friendly interface for running simulations of last-mile delivery demand. It allows users to:

- Define a geographic area of interest
- Classify buildings based on OpenStreetMap data
- Estimate building heights
- Allocate population to residential buildings
- Model delivery demand
- Generate origin-destination matches for deliveries
- Visualize results with maps, charts, and animations

## Installation

### Local Development

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Usage

1. Configure your simulation parameters in the sidebar:
   - Area Selection: Define the geographic area using coordinates
   - Data Acquisition: Select Census data sources
   - Building Classification: Set parameters for identifying residential buildings
   - Height Estimation: Configure building height estimation
   - Demand Model: Adjust delivery demand parameters
   - Simulation Parameters: Set duration and random seed
   - Origin-Destination Matching: Configure how deliveries are matched to stores

2. Click "Run Simulation" to start the process
3. View results in the tabs for each module
4. Download the routing dataset for further analysis

## Modules

The simulation is divided into several modules:

- **Module A**: Area Selection & Data Acquisition - Fetches geographic data and Census information
- **Module B**: Building Classification & Store Identification - Classifies buildings and identifies stores
- **Module C**: Building Height Estimation - Estimates building heights
- **Module D**: Population Allocation - Allocates population to residential buildings
- **Module E**: Demand Modeling - Simulates delivery demand based on population and parameters
- **Module F**: Origin-Destination Matching - Matches deliveries to stores

## Deployment

### Streamlit Community Cloud

You can deploy this application for free on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push your code to a GitHub repository
2. Sign in to Streamlit Community Cloud
3. Create a new app and connect it to your repository
4. Set the main file path to `streamlit_app.py`

### Other Deployment Options

- **Heroku**: Deploy using a Procfile with `web: streamlit run streamlit_app.py --server.port=$PORT`
- **AWS/Azure/GCP**: Deploy using container services with the provided Dockerfile
- **Private Server**: Run behind Nginx with SSL for production environments

## Data Sources

- **OpenStreetMap**: Building footprints and points of interest
- **US Census Bureau**: Demographic data (population, income, etc.)

## License

MIT License

## Credits

Based on UAS Last-Mile Delivery Demand Modeling research. 