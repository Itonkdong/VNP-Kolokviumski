import datetime
import pandas as pd
import requests as req
import urllib3
from bs4 import BeautifulSoup
from IPython.display import HTML
import warnings

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")


def get_soup(snapshot_url):
    response = req.get(snapshot_url)

    if response.status_code != 200:
        raise Exception("Something went wrong")

    raw_html = response.text
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup


def print_elements(collection):
    for element in collection:
        print(element)


def format_white_space(string: str):
    return " ".join(string.split())



def parse_date(date: str) -> datetime.datetime:
    """
        Parses a date string in the format "Sat, Oct 26th 2024" to a datetime object.

        This function removes ordinal suffixes (e.g., "st", "nd", "rd", "th") from the day
        in the date string, then parses it into a datetime object.

        Parameters:
        - date (str): A date string in the format "Day, Month Day_suffix Year",
                      for example, "Sat, Oct 26th 2024".

        Returns:
        - datetime: A datetime object representing the parsed date.

        Example:
        >>> parse_date("Sat, Oct 26th 2024")
        datetime.datetime(2024, 10, 26, 0, 0)
        """

    date_parts = date.split()
    date_parts[0] = date_parts[0][:-1]
    date_parts[2] = date_parts[2][:-2]
    clear_date_string = f"{date_parts[0]}, {date_parts[1]} {date_parts[2]} {date_parts[3]}"
    parsed_date = datetime.datetime.strptime(clear_date_string, "%a, %b %d %Y")
    return parsed_date
