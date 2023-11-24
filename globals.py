INDEX_TYPE = "uint32[pyarrow]"
ECG_TYPE = "float32[pyarrow]"
ANN_TYPE = "string[pyarrow]"
BOOL_TYPE = "bool"


# tsipouras2005

CATEGORY_1_BEATS = ['N', '/', 'f', 'F', 'L', 'R', 'Q', 'V']

CATEGORY_2_BEATS = ['V']


LIST_BEATS_1 = CATEGORY_1_BEATS + CATEGORY_2_BEATS


mb_artm_directory = './databases/mit-bih-arrhythmia-database-1.0.0/'
mb_nsrdb_directory = './databases/mit-bih-normal-sinus-rhythm-database-1.0.0/'
dataframes_directory = './Results/dataframes/'
figures_directory = './Results/figures/'
