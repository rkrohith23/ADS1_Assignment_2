# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:54:12 2023

@author: rkkro
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis


def read_world_bank_data(filePath):
    """
    Alter the columns labeled "Country Name" and "Time" after reading
    World Bank data from a CSV file.

    Parameters:
    - filePath: the path of the CSV file holding data from the World Bank.

    Returns:
    - df (pd.DataFrame): The original DataFrame read from the CSV file.
    - transposed (pd.DataFrame): The transposed DataFrame with 'Country Name'
      and 'Time' columns interchanged and cleaned.
    """
    # Read the CSV file into a DataFrame
    year_df = pd.read_csv(filePath)

    # Create a copy of the DataFrame for transposition
    country_df = year_df.copy() #Assume Country_df as transposed dataframe

    # Swap 'Country Name' and 'Time' columns
    country_df[['Country Name', 'Time']] = country_df[['Time', 'Country Name']]

    # Rename the columns for clarity
    country_df = country_df.rename(columns={'Country Name': 'Time', 'Time': 'Country Name'})

    # Additional datacleaning steps 
    # dropping unnecessary columns and handling missing values
    country_df = country_df.drop(columns=['Time Code','Country Code'])  

    # Handle missing values if needed
    country_df = country_df.dropna()

    return country_df, year_df   



def heatmap(correlation_matrix):
    if not correlation_matrix.empty:
        # Create a heatmap for the correlation matrix
        plt.figure(figsize = (12 , 8))
        heatmap = sns.heatmap(correlation_matrix , square=True, annot = True , cmap = 'rocket' ,
                              fmt = ".2f" , linewidths = .5)
        plt.xlabel('Indicators')
        plt.ylabel('Indicators')
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Correlation Coefficient')
        plt.title('Correlation Matrix for Selected World Bank Indicators' ,
                  fontsize = 18)
        plt.show()
    else:
        print("Correlation matrix is empty.")  





def histogram_plot(data , kurtosis_value):
    
    # Create a histogram
    sns.set_palette("viridis")

    # Create a histogram using Seaborn with kernel density estimate (KDE)
    sns.histplot(data, kde=True, bins=20)
    plt.title(f'Distribution with Kurtosis {kurtosis_value:.2f} '
              f'(Renewable energy consumption)' , fontsize = 15)
    
    #Adding labels
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    
    # Adding legend
    plt.legend()
    plt.show()
    






def BarGraph(data, columnName, countries=['India','Canada','Germany', 'United Kingdom',
                'Afghanistan', 'United States','Saudi Arabia']):
    
    
    # Validate dataset
    if 'Country Name' not in data.columns or 'Time' not in data.columns or columnName not in data.columns:
        raise ValueError("Invalid data format. Required columns: 'Country Name', 'Time', {}".format(columnName))

    # Set color palette
    sns.set_palette("tab10")

    # Plot data for each country
    bar_width = 0.1
    positions = np.arange(len(data[data['Country Name'] == countries[0]]['Time']))

    for i, country in enumerate(countries):
        specific_data = data[data['Country Name'] == country]
        position_variable = positions + (i - 1) * bar_width
        plt.bar(position_variable, specific_data[columnName], width=bar_width,
                label=country, alpha=0.7, edgecolor='black')

    # Adding labels and title
    plt.xlabel('Year')
    plt.xticks(positions, data[data['Country Name'] == countries[0]]['Time'])
    plt.ylabel('Percentage')
    plt.title(columnName, fontsize=18)

    # Adding legend
    plt.legend()

    # Display a grid
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()







def lineGraph(data):
    for country in ['India','Canada','Germany', 'United Kingdom',
                'Afghanistan', 'United States','Saudi Arabia']:
        country_data = data_Year[data_Year['Country Name'] == country]
        # print(country_data['Fossil fuel energy consumption (% of total)'])
        plt.plot(country_data['Time'] , country_data['Fossil fuel energy consumption (% of total)'] ,
                 label = country)

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('Fossil fuel energy consumption (% of total)')
    plt.title('Fossil fuel energy consumption' , fontsize = 18)
   

    # Adding legend
    plt.legend()

    # Display a grid
    plt.grid(True)
    plt.show()


def lineGraph_two(data):
    for country in ['India','Canada','Germany', 'United Kingdom',
                'Afghanistan', 'United States','Saudi Arabia']:
        country_data = data_Year[data_Year['Country Name'] == country]
        # print(country_data['CO2 emissions (kt)'])
        plt.plot(country_data['Time'] , country_data['CO2 emissions (kt)'] ,
                 label = country)

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('CO2 emissions (kt)')
    plt.title('CO2 emissions (kt)' , fontsize = 18)

    # Adding legend
    plt.legend()

    # Display a grid
    plt.grid(True)
    plt.show()


    
def scatterPlot_three(data):
    for country in ['India', 'Canada', 'Germany', 'United Kingdom', 'Afghanistan', 'United States', 'Saudi Arabia']:
        country_data = data_Year[data_Year['Country Name'] == country]
        plt.scatter(country_data['Time'], country_data['Total greenhouse gas emissions (kt of CO2 equivalent)'], label=country, marker='o')

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('Total greenhouse gas emissions (kt of CO2 equivalent)')
    plt.title('Total greenhouse gas emissions', fontsize=18)

    # Adding legend
    plt.legend()

    # Display a grid
    plt.grid(True)
    plt.show()




def pieChart(data):
    # Assuming 'Country Name' and 'Population growth (annual %)' columns are present in the DataFrame
    # You may need to adjust column names based on your actual data structure

    # Filter data for the year 2020
    piedata = data[data['Time'] == 2020]

    # Define the list of countries to include in the pie chart
    countries = ['India','Germany', 'United Kingdom', 'Afghanistan','Canada', 'United States', 'Saudi Arabia']

    # Filter data for the selected countries
    country_data = piedata[piedata['Country Name'].isin(countries)]

    # Check if the country_data is not empty before creating the pie chart
    if not country_data.empty:
        # Define explode values (fraction of the radius with which to offset each wedge)
        explode = (0.1, 0, 0, 0.1, 0, 0, 0.1)  # Adjust the values as needed

        # Create a pie chart with explode
        plt.pie(country_data['Population growth (annual %)'], labels=country_data['Country Name'],
                autopct='%1.f%%', startangle=95, explode=explode)

        # Customize the plot as needed
        plt.title('Population growth (annual %) of year 2020', fontsize=18)

        # Show the plot
        plt.show()
    










country_data , year_data = read_world_bank_data("world_bank_dataset.csv")
print("             -------------- COUNTRY DATA ------------------             ")
print(country_data.head())
print("             ------------------ YEAR DATA  ------------------              ")
print(year_data.head())

#statistical analysis
print("            ------------------ STATISTICAL ANALYSIS ------------------         ")

"""
METHOD 1 :
DESCRIBE:
"""
country_data['Population growth (annual %)'] = pd.to_numeric(country_data['Population growth (annual %)'] ,
                                             errors = 'coerce')
describes = country_data['Population growth (annual %)'].describe()

print("------------------ METHOD 1 : Describes ")
print(describes)

"""
METHOD 2 :
SKEWNESS
"""
print("------------------ METHOD 2 : Skewness  ")
skew_column_name = 'Fossil fuel energy consumption (% of total)'
# Use apply with a lambda function to convert the column to numeric and calculate skewness
country_data[skew_column_name] = country_data[skew_column_name]\
    .apply(lambda x: pd.to_numeric(x , errors = 'coerce'))
# Now, you can calculate skewness
skewness_value = country_data[skew_column_name].skew()
print("The Skewness value of Fossil fuel energy consumption (% of total) is" , skewness_value)
"""
KURTOSIS
"""

print("------------------ METHOD 3: Kurtosis ")
kurtosis_column = 'Renewable energy consumption (% of total final energy consumption)'

country_data[kurtosis_column] = pd.to_numeric(country_data[kurtosis_column] ,
                                              errors = 'coerce')
# Calculate kurtosis
kurtosis_value = country_data[kurtosis_column].kurtosis()
print('Kurtosis for Renewable energy consumption (% of total final energy consumption) is ' ,
      kurtosis_value)
histogram_plot(country_data[kurtosis_column] , kurtosis_value)

"""CORELATION"""
print(country_data.columns)


#Heat map
# Select a few indicators for analysis
selected_indicators = ['Population growth (annual %)' , 'Total greenhouse gas emissions (kt of CO2 equivalent)','CO2 emissions (kt)' ,
                       'Fossil fuel energy consumption (% of total)','Renewable energy consumption (% of total final energy consumption)']
# Extract the relevant data for the selected indicators
df_selected_indicators = country_data[selected_indicators]
# Calculate the correlation matrix
correlation_matrix = df_selected_indicators.corr()
column_names = ['Population Growth', 'Total Greenhouse gas Emissions', 'CO2 Emissions', 'Fossil Fuel Consumption', 'Renewable Energy Consumption']

# Rename the columns in the correlation matrix
correlation_matrix.columns = column_names
correlation_matrix.index = column_names

heatmap(correlation_matrix)

#pieGraph
pieChart(country_data)

#bar graph
country_data['Time'] = pd.to_numeric(country_data['Time'], errors='coerce')

data_Year = country_data[(country_data['Time'] >= 1990) &
                         (country_data['Time'] <= 2020)]
BarGraph(data_Year , 'Renewable energy consumption (% of total final energy consumption)')

#Line Graph
data_Year = country_data[(country_data['Time'] >= 1990) &
                         (country_data['Time'] <= 2020)]
lineGraph(data_Year)
lineGraph_two(data_Year)

#Scatter Graph
scatterPlot_three(data_Year)


#bargraph
data_Year = country_data[(country_data['Time'] >= 1990) & (country_data['Time'] <= 2021)]
BarGraph(data_Year , 'GDP growth (annual %)')


#pieGraph
pieChart(country_data)