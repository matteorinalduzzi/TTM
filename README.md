# Predicting Venetian Lagoon Tide Levels with Multivariate Time Series Modeling

## Abstract: 
This article explores the application of multivariate time series models for predicting tide levels in the Venice Lagoon. We leverage Time Series Foundation Models (TSFM) from the [TTM library](https://huggingface.co/ibm/TTM) to construct a robust model that incorporates various relevant factors. Furthermore we demonstrate how this time series model developed by [IBM Research](https://arxiv.org/pdf/2401.03955.pdf) can be deployed and run on the [IBM watsonx platform](https://www.ibm.com/watsonx) in order to streamline the process and unlock additional benefits such as model governance. The model is trained on datasets retrieved from the official [portal of the City of Venice](https://www.comune.venezia.it/it/content/centro-previsioni-è-segnalazioni-maree) and from [ISPRA Servizio Laguna di Venezia](https://www.venezia.isprambiente.it/rete-meteo-mareografica). 
The article details the process of data acquisition, feature selection, and model development using TSFM. It then evaluates the model's performance in predicting tide levels and discusses the implications for flood forecasting and improved water management strategies in the Venice Lagoon. The code for implementing this approach is provided in a [Jupyter notebook](tsfm/ttm_venice_levels.ipynb). 
**TODO: Add findings**

## Introduction

The Venice Lagoon is a unique ecosystem where the city of Venice itself is built on a network of islands and canals. This beautiful setting comes with a challenge: fluctuating tide levels. Particularly high tides, known as "acqua alta," can flood parts of the city, impacting residents and infrastructure.

Predicting these high tides accurately, but also giving indication on times for hifgh and low tides in the day, is crucial for flood forecasting and water management. By knowing the expected water level, authorities can activate flood barriers, issue warnings, and adjust transportation routes to minimize disruption.

Such study is a two-step process, according to ISPRA [https://www.venezia.isprambiente.it/](https://www.venezia.isprambiente.it/).  

First, scientists separate out the predictable influence of the moon and sun using established methods. Then, they tackle the more chaotic effect of weather. Here, two approaches come into play. Statistical models analyze past data on tides, weather patterns, and even forecasts to find reliable connections and predict future surges. Deterministic models, on the other hand, use powerful computer simulations to mimic how the ocean responds to wind and pressure, calculating the surge across the entire Mediterranean.  

By combining these methods, scientists can provide accurate short-term forecasts (12-48 hours) and a general idea of the tide's direction over longer periods (3-5 days). 

In this context, we would liek to explore multivariate time series modeling. This approach considers multiple factors that influence tides, not just astronomical cycles but also meteorological factors such as:

- Wind speed and direction
- Atmospheric pressure
- Temperature
- Rain (?)

By incorporating this additional data, the model can paint a more complete picture of the forces affecting the lagoon's water level. This potentially leads to more accurate forecasts and better preparedness for high tides in Venice.

## Time Series: a Powerful Tools for Forecasting

A time series is a collection of data points indexed in chronological order. Imagine temperature readings taken every hour, or stock prices recorded daily - these are all examples of time series data. The power of time series analysis lies in its ability to exploit the inherent temporal relationships within the data for forecasting purposes. 

Time series can be applied in various forecasting problems:

* **Demand forecasting:** Businesses can predict future sales trends based on historical sales data, seasonality, and marketing campaigns. 
* **Financial forecasting:** Investment firms can analyze past stock market movements to predict future trends and make informed investment decisions.
* **Weather forecasting:** Meteorological agencies use time series models incorporating temperature, pressure, and humidity data to predict future weather patterns.
* **Inventory management:** Retailers can optimize stock levels by forecasting future demand using historical sales data and lead times.

Traditionally, moving averages and exponential smoothing were used for time series forecasting. However, the field has seen significant advancements with the rise of machine learning. Here are some cutting-edge models:

* **ARIMA (Autoregressive Integrated Moving Average):** This model captures trends and seasonality by analyzing past values and residuals from the data.

  * **Autoregressive Integrated Moving Average (ARIMA) Model** by Hyndman, R.J. & Athanasopoulos, G. (2013) [https://otexts.com/fpp3/](https://otexts.com/fpp3/). This is a comprehensive text  covering the theory and application of ARIMA models for time series forecasting.
  * **Box-Jenkins methodology** by Wikipedia [https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method](https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method). This Wikipedia entry provides a concise overview of the Box-Jenkins methodology, a statistical framework for fitting ARIMA models to time series data.

* **Prophet:** Developed by Facebook, Prophet is a powerful tool that considers holidays, seasonality, and other user-defined regressors for improved forecasting accuracy.

  * **Forecasting at Scale with Prophet** by Sean J. Taylor et al. (2017) [https://arxiv.org/pdf/2005.07575](https://arxiv.org/pdf/2005.07575). This is the original research paper by Facebook introducing Prophet, detailing its architecture and implementation.
  * **Prophet: Forecasting Prophet** by Facebook [https://github.com/facebook/prophet](https://github.com/facebook/prophet). This is the official documentation for Prophet from Facebook, providing a user guide and tutorials.

* **Deep Learning Techniques:** Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks are particularly adept at capturing complex temporal patterns in data, leading to highly accurate forecasts, especially for non-linear data.

  * **Long Short-Term Memory Networks with Recurrent Neural Networks** by Sepp Hochreiter & Jürgen Schmidhuber (1997) [https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf). This is the seminal paper introducing Long Short-Term Memory (LSTM) networks, a specific type of RNN well-suited for time series forecasting.
  * **Recurrent neural networks** by Wikipedia [https://en.wikipedia.org/wiki/Recurrent_neural_network](https://en.wikipedia.org/wiki/Recurrent_neural_network). This Wikipedia entry provides a general introduction to RNNs, including their architecture, training process, and applications beyond time series forecasting.

When building a time series model, it's crucial to distinguish between two types of data: endogenous and exogenous variables.

* **Endogenous Variables:** These are the core time series data points you aim to predict. In our Venice Lagoon example, the endogenous variable would be the tide levels at a specific location. 
* **Exogenous Variables:** These are external factors that can influence the endogenous series. For instance, when predicting tide levels, exogenous variables might include wind speed, atmospheric pressure, or moon phase. 

Time series models can leverage the relationship between these endogenous and exogenous variables. By incorporating these external factors, models can capture a more comprehensive picture and improve prediction accuracy.

Predicting tide levels in the Venice Lagoon is a prime example perfectly suited for time series analysis, especially when considering exogenous variables. Here's why:

* **Historical Data Availability:** Long-standing records of tide levels exist, providing a rich dataset for training time series models.
* **Cyclical Nature:** Tides exhibit predictable daily and seasonal cycles that time series models can effectively capture.
* **External Influences:** Meteorological factors like wind and atmospheric pressure significantly impact tide levels. Time series models can incorporate these external factors (exogenous variables) for more comprehensive forecasting.

By analyzing historical data on tide levels, wind patterns, and atmospheric pressure (exogenous variables), time series models can provide accurate predictions of future lagoon water levels (endogenous variable). This information is crucial for flood forecasting, allowing authorities to implement preventative measures and ensure the safety of Venice and its inhabitants.  


## Data Acquisition and Preprocessing

* Explain the source of the data used for model training. In this case, mention the official portal of the City of Venice for tide level data. 
* Briefly describe the datasets retrieved from that will be used for the additional factors influencing tides (wind speed, atmospheric pressure, etc.). 
* Explain any necessary preprocessing steps like handling missing values, scaling, and time-series transformation (if needed).

## Leveraging watson ML environments for running the model

* Explain how to go from the cloning the code repository to creating a service to run the IBM Cloud
* Discuss benefits of such environment such as governance, dataset assets...

## Model Development with TSFM

* Briefly introduce the Time Series Foundation Models (TSFM) from the TTM library. 
* Explain the chosen TSFM architecture for the model (e.g., Informer, Transformer). 
* Describe the process of feature selection: which factors from the retrieved datasets will be included as input features besides tide levels.
* Explain the model training process, including hyperparameter tuning (if applicable).

## Model Evaluation and Results

* Describe the chosen metrics for evaluating the model's performance in predicting tide levels (e.g., Mean Squared Error, Root Mean Squared Error).
* Present the achieved prediction accuracy and discuss the model's effectiveness.
* Consider including visualizations like scatter plots or time series plots comparing predicted vs. actual tide levels.

## Discussion and Future Work

* Discuss the implications of this approach for flood forecasting and water management strategies in the Venice Lagoon.
* Highlight the limitations of the model and potential areas for improvement (e.g., incorporating additional factors or using more advanced TSFM architectures).
* Briefly discuss potential future work, such as exploring real-time forecasting applications or integrating the model with existing decision support systems.

## Conclusion

* Summarize the key findings of the article, emphasizing the effectiveness of the TSFM-based model for predicting tide levels in the Venice Lagoon.
* Restate the importance of accurate tide level prediction for the future of Venice.

## Appendix (Optional)

* Include the Jupyter notebook code (or a link to the code repository) for implementing the model.
* Provide any additional technical details or supplementary information relevant to the model development process.
