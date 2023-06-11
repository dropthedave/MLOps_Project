# MLOps_Project

Use Kedro organization and modular code. You can keept your exploration notebooks in your
appropriate folder.
1. Unit data tests: You can use one of the tools from the class or your own solutions, but it is important to have several asserts for the data quality.
2. MLflow for experimentation and model versioning.
3. Data drift evaluation: If you build a pipeline to test a sample of data of your strongest model, include this component as well. 
   You can play with your sample if you want to generate drift, or see how the mtetrics would change if drift happens.
5. Try to build tests for your relevant functions and pipelines.

# Datasets

   ## Smoke Detection
   - Binary Classification Task
   - Predict Fire Alarm


   ## Dataset Source
   https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset
   
   ## Column Description

   **UTC**<br>
   Timestamp UTC seconds

   **Temperature[C]**<br>
   Air Temperature

   **Humidity[%]**<br>
   Air Humidity

   **TVOC[ppb]**<br>
   otal Volatile Organic Compounds; measured in parts per billion

   **eCO2[ppm]**<br>
   co2 equivalent concentration; calculated from different values like TVCO

   **Raw H2**<br>
   raw molecular hydrogen; not compensated (Bias, temperature, etc.)

   **Raw Ethanol**<br>
   raw ethanol gas

   **Pressure[hPa]**<br>
   Air Pressure

   **PM1.0**<br>
   particulate matter size < 1.0 µm (PM1.0). 1.0 µm < 2.5 µm (PM2.5)

   **PM2.5**<br>
   particulate matter size < 1.0 µm (PM1.0). 1.0 µm < 2.5 µm (PM2.5)

   **NC0.5**<br>
   Number concentration of particulate matter. This differs from PM because NC gives the actual number of particles in the air. The raw NC is also classified by the particle size: < 0.5 µm (NC0.5); 0.5 µm < 1.0 µm (NC1.0); 1.0 µm < 2.5 µm (NC2.5);

   **NC1.0**<br>
   Number concentration of particulate matter. This differs from PM because NC gives the actual number of particles in the air. The raw NC is also classified by the particle size: < 0.5 µm (NC0.5); 0.5 µm < 1.0 µm (NC1.0); 1.0 µm < 2.5 µm (NC2.5);

   **NC2.5**<br>
   Number concentration of particulate matter. This differs from PM because NC gives the actual number of particles in the air. The raw NC is also classified by the particle size: < 0.5 µm (NC0.5); 0.5 µm < 1.0 µm (NC1.0); 1.0 µm < 2.5 µm (NC2.5);

   **CNT**<br>
   Sample counter

   **Fire Alarm (Target)**<br>
   ground truth is "1" if a fire is there
