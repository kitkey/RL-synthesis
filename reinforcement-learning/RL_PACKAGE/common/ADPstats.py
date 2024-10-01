from pandas import read_csv
import os.path
from __init__ import PATH_FOR_GETTING_RANDOM_SCHEMES


def get_adp_stats(name: str, test_number = 1) -> dict:
    if test_number == 0:
        path = os.path.join("ADP_stats", f"ad_{name}.csv")
        a = read_csv(path)
        a["ad"] = a["area"] * a["delay"]

        stat_dict = {
                "best area-delay": a[a['ad'] == a['ad'].min()].to_dict(orient='records'),
                "best area": a[a['area'] == a['area'].min()].to_dict(orient='records'),
                "best delay": a[a['delay'] == a['delay'].min()].to_dict(orient='records'),
                "mean area-delay": a["ad"].mean(),
                "mean area": a["area"].mean(),
                "mean delay": a["delay"].mean()
                }
    else:
        pd = read_csv(PATH_FOR_GETTING_RANDOM_SCHEMES)
        stat_dict = {"best area-delay": pd[name].min(),
                       "mean area-delay": pd[name].mean()}
    # print(json.dumps(stat_dict, indent=4))
    return stat_dict
