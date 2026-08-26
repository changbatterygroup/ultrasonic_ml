gathering code and notebooks for ultrasonic analysis

# Restructured to SQlite to faster analysis and visualization

### maintainance:
* periodically update readme as the functions change
* find and write TODO tests
* test heritability of base classes as more functions added on


## SQlite Database structure:
1. indent is table
2. indents are columns
* original waveform is referenced with collection idx
* analysis waveforms are recognized with any combination of PRIMARY KEY (collection_index,waveform,analysis_name,result_name,reference_id)

```text
DATABASE.sqlite3
|
├── parameters (originally here)
|   |-- gui TEXT
|   |-- experiment TEXT
|   ...
|   |-- 'omega_scaled' REAL (*method* for calculating last 3)
|   |-- 'dt' REAL
|   |-- 'f_s' REAL
│
├── acoustics (originally here)
│   ├── voltage BLOB (*method* for getting key names)
|   ... multiple scans from multiplexer
│   ├── time REAL
|   |-- frequency REAL (*method* for calculating)
|   |-- time_collected REAL (*method* need to retireve and translate to datetime format)
│   ├── collection_index INTEGER 
│   ├── X REAL
│   ├── Z REAL
│
|── analysis (*method* to write table)
|   ├── collection_index INTEGER  (same as acoustics table)
|   ├── waveform TEXT (select from column name in acoustics. i.e. voltage, voltage_echo_forward, etc )
|   ├── analysis_name TEXT (*method* for each. i.e. filter, ungain, hilbert, fft, cwt, correlation )
|   |-- result_name TEXT (ie raw, magnitude, phase, group_velocity) (*method)
|   ├── value BLOB (array)
|   ├── x_axis TEXT (ie time, freq) TODO: are these better in the reference section? need to test flow later
|   |── x_unit TEXT (ie ns, Hz)
|   ├── y_axis TEXT (ie time, freq)
|   |── y_unit TEXT (ie mV, Hz)
|   |-- reference_id INTEGER 
|   PRIMARY KEY (collection_index,waveform,analysis_name,result_name,reference_id)
|
|-- analysis_reference (*method* for constructing)
|   |-- reference_id INTEGER (use as idx)
|   |-- analysis_name TEXT (ie, cwt, correlation, fft)
|   |-- reference_name TEXT (ie filename, 'cmor0.5-1.0') 
|   |-- value BLOB (*method* to read in a water waveform, scale, interp), (*method* upload from cwt)
|   |-- metadata TEXT (ie kwargs to calculate in function)
|
|-- controller (*method* translate merging and filtering methods from csv. Reference gradient_helpers.py)
|   |-- collection_index
|   ...
...
```

## SQlite_Database python class and methods
```
DATABASE
    |
    V
AcousticsDatabase (base)
│
├── database connection
│   ├── connect()
│   ├── close()
│   ├── __enter__()
│   ├── __exit__()
│   ├── get_tables()
│   └── get_columns()
│
├── parameters
│   ├── get_parameters()
│   ├── initialize_frequency_parameters()
│   └── get_datetime()
│
├── waveform handling
│   ├── get_waveform_columns()
│   ├── has_waveform()
│   ├── fetch_column()
│   ├── fetch_value()
│   ├── fetch_waveform()
│   ├── fetch_waveform_batch()
│   ├── fetch_time()
│   ├── fetch_waveform_value()
│   ├── fetch_index_across_acquisitions()
│   ├── get_acquisition_count()
│   └── get_acquisition_index()
│
├── preprocessing
│   ├── preprocess()
│   ├── fetch_preprocessed_waveform()
│   ├── _undo_gain()
│   ├── _butterworth_filter()
│   └── _absolute_max()
│
├── analysis results
│   ├── create_analysis_table()
│   ├── store_analysis_result()
│   ├── fetch_analysis_result()
│   └── list_analysis_results()
│
├── analysis references
│   ├── create_reference_table()
│   ├── create_reference()
│   ├── fetch_reference()
│   └── list_references()
│
├── serialization
│   ├── serialize_array()
│   └── deserialize_array()
│
└── subclass hooks # TODO: figure out when we have children
    ...
        |
        V
    child databases # TODO: translate existing thermal gradient, FT, and CWT work done in pandas
            │
            ├── FrequencyDomainDatabase
            ├── ThermalGradientsDatabase
```


## Viewer
* run in jupyter notebook with
    * `%matplotlib tk` for external window
    * `%matplotlib widget` for inline viewer
* run as script
    * `matplotlib.use("TkAgg")`

```
DATABASE
    |
    V
AcousticsDatabase
    |
    V
AcousticsViewer (base)
│
├── Initialization
│   └── __init__(db, waveforms, analyses, figsize, x_limits, y_limits)
│
├── Limits
│   ├── _get_x_limits()
│   └── _get_y_limits()
│
├── Data
│   ├── fetch_raw(waveform, row)
│   ├── fetch_analyzed(waveform, analysis_name, result_name, row, reference_id)
│   ├── fetch_time(row)
│   └── fetch_analyzed_labels(row)
│
├── Figure management
│   ├── build()
│   └── _setup_axes()
│
├── Waveform visualization
│   ├── plot_raw()
│   ├── plot_analyzed()
│   └── update()
│
├── Collection navigation
│   ├── _create_slider()
│   ├── _on_slider()
│   ├── goto(row)
│   ├── next()
│   └── previous()
│
├── Interaction
│   ├── _connect_events()
│   ├── _on_click()
│   ├── on_click()
│   └── _on_key()
│
└── Display (User should only use these)
    ├── refresh()
    ├── show_raw()
    └── show_preprocessed()
          |
          V
     child viewers # TODO: translate existing thermal gradient, FT, and CWT work done in pandas
          │
          ├── CWTViewer
          ├── SpatialViewer
          ├── ThermalGradientsViewer
```