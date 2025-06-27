# Drift config can be specified using indexes if INTS
# or portion of the DF in case of not knowing the size of the dataset


# More on electricity here: https://api.openml.org/d/44156

electricity_columns = [
    "date",
    "day",
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
]

# The column is a column present in the dataset determined by the user on the left side of the drift_config
drift_config = {
    "electricity": {
        "abrupt_gradual": {
            "column": "nswdemand",
            "drifts": {
                "abrupt": [(1000, 2000), (4000, 5000)],
                "gradual": [(6000, 7000)],
            },
        },
        "just_abrupt": {
            "column": "nswdemand",
            "drifts": {"abrupt": [(1000, 8000), (8000, 15000)]},
        },
        "incremental": {
            "column": "nswdemand",
            "drifts": {"incremental": [(0.4, 0.8)]},
        },
    },
    # "MULTISTAGGER": {
    #     "just_abrupt": {
    #         "column": "size",
    #         "drifts": {"abrupt": [(0.4, 0.8)]},
    #     },
    # },
}
