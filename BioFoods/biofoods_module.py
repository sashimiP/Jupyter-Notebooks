import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits.basemap import Basemap

import seaborn as sns


# Функции за анализиране на пазара


def load_table(file_path):
    """
    Loads a dataframe from an excel file.
    CLeans the columns.
    Returns a pandas DataFrame.
    """
    dataframe = pd.read_excel(file_path, skiprows=range(0, 8), header=[0, 1])
    dataframe = dataframe.drop(columns=dataframe.columns[2])
    dataframe.columns = [dataframe.columns[0][1].lower(), dataframe.columns[1][1].lower(),
                         dataframe.columns[2][0].lower()]

    return dataframe


def get_leading_countries(dataframe, feature_name, size_y=5):
    """
    Get the top leading countries in the specified feature.
    Returns a pandas DataFrame.
    size_y: number of countries.
    """
    leading_countries = dataframe.nlargest(size_y, feature_name)

    return leading_countries


def plot_world_organic_subplots():
    bar_colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:cyan']
    world_sales = load_table("data/fibl_world_retail_sales_2019.xlsx")
    world_consumption_per_capita = load_table("data/fibl_world_consumption_per_capita_2019.xlsx")
    world_farmland_ha = load_table("data/fibl_world_farmaland_2019_ha.xlsx")
    world_farmland_per_cent = load_table("data/fibl_world_farmaland_2019_per_cent.xlsx")

    top_5_sales = get_leading_countries(world_sales, world_sales.columns[2])
    top_5_per_capita = get_leading_countries(world_consumption_per_capita, world_consumption_per_capita.columns[2])
    top_5_farmland_ha = get_leading_countries(world_farmland_ha, world_farmland_ha.columns[2])
    top_5_farmland_per_cent = get_leading_countries(world_farmland_per_cent, world_farmland_per_cent.columns[2])

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].bar(top_5_sales[top_5_sales.columns[0]], top_5_sales[top_5_sales.columns[2]], color=bar_colors[0])
    axs[0, 0].set_title("Top 5 Countries in Organic Retail Sales 2017")
    axs[0, 0].set_ylabel("Organic retail sales [Million €]")
    axs[0, 0].tick_params('x', labelrotation=45)
    axs[1, 0].bar(top_5_per_capita[top_5_per_capita.columns[0]], top_5_per_capita[top_5_per_capita.columns[2]],
                  color=bar_colors[1])
    axs[1, 0].set_title("Top 5 Countries in Organic Consumption Per Capita 2017")
    axs[1, 0].set_ylabel("Organic per capita consumption [€/person]")
    axs[1, 0].tick_params('x', labelrotation=45)
    axs[0, 1].bar(top_5_farmland_ha[top_5_farmland_ha.columns[0]], top_5_farmland_ha[top_5_farmland_ha.columns[2]],
                  color=bar_colors[2])
    axs[0, 1].set_title("Top 5 Countries in Organic Farmland in Hectars 2017")
    axs[0, 1].set_ylabel("Organic area (farmland) [ha]")
    axs[0, 1].tick_params('x', labelrotation=45)
    axs[1, 1].bar(top_5_farmland_per_cent[top_5_farmland_per_cent.columns[0]],
                  top_5_farmland_per_cent[top_5_farmland_per_cent.columns[2]], color=bar_colors[3])
    axs[1, 1].set_title("Top 5 Countries in Organic Farmland Share of Total [%] 2017")
    axs[1, 1].set_ylabel("Organic area share of total farmland [%]")
    axs[1, 1].tick_params('x', labelrotation=45)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=1.5,
                        top=1.5,
                        wspace=0.4,
                        hspace=0.6)
    plt.show()


def plot_bar_chart(dataframe):
    plt.bar(dataframe[dataframe.columns[0]], dataframe[dataframe.columns[2]],
            color=['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:cyan'])
    plt.xlabel(dataframe.columns[0], fontsize=14)
    plt.ylabel(dataframe.columns[2], fontsize=14)
    plt.xticks(fontsize=12, rotation='vertical')
    plt.yticks(fontsize=12)
    plt.show()


# Функции за почистване на таблицата

def remove_nan_observations(dataframe, feature_to_clean):
    """
    Removes observations, if observations contain NaN values in the specified feature.
    Returns a pandas DataFrame.
    """
    clean_df = dataframe[dataframe[feature_to_clean].notna()]

    return clean_df


def remove_more_digits(dataframe, limit):
    """
    Removes the telephone numbers with more than the specified number of symbols.
    Returns a pandas DataFrame.
    """
    indexes_to_drop = dataframe[dataframe["Телефон за връзка с респондента:"].str.len() > limit].index
    clean_df = dataframe.drop(indexes_to_drop)

    return clean_df


def remove_less_digits(dataframe, limit):
    """
    Removes the telephone numbers with less than the specified number of elements.
    Returns a pandas DataFrame.
    """
    indexes_to_drop = dataframe[dataframe["Телефон за връзка с респондента:"].str.len() < limit].index
    clean_df = dataframe.drop(indexes_to_drop)

    return clean_df


def standartise_phones(dataframe):
    """
    Replaces "+359" with "0".
    Returns a pandas DataFrame.
    """
    clean_feature = dataframe["Телефон за връзка с респондента:"].str.replace("359", "0", 1).str.strip("+")
    clean_df = dataframe.copy()
    clean_df["Телефон за връзка с респондента:"] = clean_feature

    return clean_df


def remove_different_first_digit(dataframe):
    """
    Removes phone numbers that begin with a different than "0" digit.
    Returns a pandas DataFrame.
    """
    indexes_to_drop = dataframe[dataframe["Телефон за връзка с респондента:"].str[0] != "0"].index
    clean_df = dataframe.drop(indexes_to_drop)

    return clean_df


def remove_white_space(dataframe, feature_to_clean):
    """
    Removes white space from string.
    Returns a pandas DataFrame.
    """
    clean_df = dataframe.copy()
    clean_df[feature_to_clean] = clean_df[feature_to_clean].str.replace(" ", "")

    return clean_df


# Функция, която извиква всички горепосочени функции.

def clean_phone_feature(dataframe, phone_feature):
    """
    Removes observations without a telephone number or with a wrong number
    based on the Bulgarian telecommunication operator standards.
    """
    tidy_phone = dataframe
    tidy_phone = remove_nan_observations(tidy_phone, phone_feature)
    tidy_phone = remove_less_digits(tidy_phone, 10)
    tidy_phone = remove_more_digits(tidy_phone, 13)
    tidy_phone = standartise_phones(tidy_phone)
    tidy_phone = remove_different_first_digit(tidy_phone)
    tidy_phone = remove_white_space(tidy_phone, phone_feature)

    return tidy_phone


def drop_feature(dataframe, feature_to_drop):
    """
    Removes the specified feature from the Data Frame.
    Returns a pandas DataFrame.
    """
    clean_df = dataframe.drop(columns=feature_to_drop)

    return clean_df


def divide_table(dataframe, features):
    """
    Divides a Data Frame in two.
    One containing the said features,
    and the other without them.
    Returns two pandas Data Frames.
    """
    df_contains = dataframe[features]
    df_without = dataframe[dataframe.columns[~dataframe.columns.isin(df_contains)]]

    return df_contains, df_without


# Индексите трябва да се рестартират, за да може тези,
# които отговарят на индексите от таблицата participants_data,
# да са в отделна колона.

def melt_dataframe(dataframe, id_vars, var_name, value_name):
    """
    Resets the indices,
    and melts the Data Frame.
    Returns a pandas DataFrame.

    id_vars : Column(s) to use as identifier variables.
    var_name : Name to use for the ‘variable’ column. If None it uses frame.columns.name or ‘variable’.
    value_vars : Name to use for the ‘value’ column.
    """
    dataframe_melt = dataframe.reset_index()
    dataframe_melt = dataframe_melt.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)

    return dataframe_melt


def get_assumption_with_number(dataframe, feature_name):
    """
    Seperates the assumption's number and the assumption from the rest of the "question" feature.
    Returns a pandas Series.
    """
    flag = dataframe['question'].str.find("[") != -1
    assumption_with_number = dataframe["question"][flag].str.partition("[")[2].str.strip("]").str.strip()
    assumption_with_number = pd.concat(
        [assumption_with_number, dataframe["question"][~flag].str.partition("(")[0].str.strip()])
    assumption_with_number.sort_index(inplace=True)

    return assumption_with_number


# Деклариране на функции, които ще разделят съставните части на въпроса на отделни колони.

def get_question_number(dataframe, feature_name):
    """
    Seperates the question's number from the rest of the "question" feature.
    Returns a pandas Series.
    """
    number = dataframe['question'].str.partition(".")[0]

    return number


def get_the_question(dataframe, feature_name):
    """
    Seperates the question from the rest of the "question" feature.
    Returns a pandas Series.
    """
    question = dataframe['question'].str.partition("?")[0].str.partition(":")[0].str.partition(".")[2].str.strip()

    return question


def get_note(dataframe, feature_name):
    """
    Seperates the participant's clarification note from the rest of the "question" feature.
    Returns a pandas Series.
    """
    note = dataframe[feature_name].str.partition('(')[1] + \
           dataframe[feature_name].str.partition('(')[2].str.partition('[')[0].str.strip("]")
    note = note.str.strip()

    return note


def get_assumption_number(assumption_and_numbers):
    """
    Seperates the assumption's number from the rest of the assumption_and_numbers Series.
    Returns a pandas Series.
    """
    assumption_number = assumption_and_numbers.str.strip(".").str.rpartition(".")[0]

    return assumption_number


def get_assumption(assumption_and_numbers):
    """
    Seperates the assumption from the assumption_and_numbers Series.
    Returns a pandas Series.
    """
    assumption = assumption_and_numbers.str.strip(".").str.rpartition(".")[2].str.strip()

    return assumption


# Automating the process of plotting a matplotlib barh chart

def select_question(dataframe, question_number):
    select_question_df = dataframe[dataframe["question_num"] == question_number]

    return select_question_df


def get_mean_answers(dataframe, question_number):
    selected_df = select_question(dataframe, question_number)

    assumption_numbers = selected_df["assumption_num"].unique()
    answer_means_lst = []
    for number in assumption_numbers:
        answer_means_lst.append(round(selected_df[selected_df["assumption_num"] == number].answer.mean(), 2))

    answer_means_series = pd.Series(answer_means_lst, index=selected_df.assumption.unique())

    return answer_means_series


# FIX IN FUTURE
def create_family_single_df(family_dataframe, single_dataframe, question_number):
    question_family_means = get_mean_answers(family_dataframe, question_number).sort_values()
    question_single_means = get_mean_answers(single_dataframe, question_number).sort_values()

    data = {"family": question_family_means.values,
            "single": question_single_means.values}

    family_single_df = pd.DataFrame(data, columns=["family", "single"], index=question_family_means.index)

    return family_single_df


def get_plot_title(dataframe, question_number):
    note = dataframe[dataframe["question_num"] == question_number]["note"].unique()[0]
    title = dataframe['the_question'].unique()[question_number - 1]

    return note, title


# FIX IN FUTURE
def plot_barh_means(family_dataframe, single_dataframe, question_number):
    plot_note, plot_title = get_plot_title(family_dataframe, question_number)

    family_single_df = create_family_single_df(family_dataframe, single_dataframe, question_number)

    family_single_df.plot.barh()
    plt.title(f"{plot_title}\n{plot_note}", fontsize=16, x=-0.1, y=1.05, bbox={'facecolor': '0.8', 'pad': 5})
    plt.xlabel("Answer means", fontsize=14)
    plt.ylabel("Assumptions", fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(title="Segment", prop={"size": 12})
    plt.show()


# Automating the process of plotting a matplotlib pie chart

def get_value_counts_percentage(dataframe, question_number):
    selected_df = select_question(dataframe, question_number)

    values = selected_df.value_counts("answer")
    total_number = selected_df.answer.size
    part_of_whole = values / total_number
    value_percentages = part_of_whole * 100

    return round(value_percentages, 2)


def get_colours(value_counts):
    number_of_colors = np.arange(value_counts.values.size, dtype=float)
    colors = cm.Set1(number_of_colors)

    return colors


def plot_pie_chart(dataframe, question_number, family_status=""):
    plot_note, plot_title = get_plot_title(dataframe, question_number)

    value_counts = get_value_counts_percentage(dataframe, question_number)
    colors = color_palette = sns.color_palette()

    my_explode = np.zeros(len(value_counts))
    my_explode[0] = 0.1

    plt.pie(value_counts, labels=value_counts.index, autopct='%.2f%%', textprops={"size": 13}, pctdistance=0.8,
            shadow=True, explode=my_explode, startangle=75, radius=1.1, wedgeprops={"edgecolor": "black"})
    if family_status.lower() == "family segment":
        plt.title(f"{plot_title}\n {plot_note}\n\n{family_status}",
                  fontsize=15)  # bbox={'facecolor':'0.8', 'pad':5})
    elif family_status.lower() == "single segment":
        plt.title(family_status, fontsize=15)  # bbox={'facecolor':'0.8', 'pad':5})
    plt.show()


def plot_family_single_pie_chart(family_dataframe, single_dataframe, question_number, ):
    plot_pie_chart(family_dataframe, question_number, "Family Segment")
    print("\n")
    plot_pie_chart(single_dataframe, question_number, "Single Segment")


def string_to_float(dataframe, column, question_num):
    """
    Replaces the string variables in the desired column and question number with a floating point variable.
    :param dataframe: pandas DataFrame.
    :param column: column containing the string variables.
    :param question_num: number of the question you want strings replaced with floats.
    :return: pandas DataFrame.
    """
    new_df = dataframe

def tidy_place_col(dataframe):
    """
    Cleans the place of residence column.

    :param dataframe: pandas DataFrame
    :return: a pandas DataFrame.
    """
    tidy_df = dataframe.copy(deep=True)
    tidy_df["Местоживеене (населено място):"] = tidy_df["Местоживеене (населено място):"].str.strip()
    tidy_df["Местоживеене (населено място):"].replace(["Виктор Цонев", "Ruse"], "Русе", inplace=True)

    return tidy_df


def add_lat_col(dataframe, latitude):
    """
     Adds latitude and longitude columns to the dataframe.
    The coordinates are based on the place of residence column.

    :param dataframe: pandas DataFrame to add the latitude and longitude columns to.
    :param latitude: a dict that holds places and their respective latitude coordinates.
    :param longitude: a dict that holds places and their respective longitude coordinates.
    :return: pandas DataFrame.
    """
    lat_df = dataframe
    lat_dict = latitude

    for key, value in lat_dict.items():
        indices = lat_df.loc[lat_df["Местоживеене (населено място):"] == key].index
        lat_df.loc[indices, "latitude"] = value

    return lat_df


def add_lng_col(dataframe, longitude):
    """
     Adds latitude and longitude columns to the dataframe.
    The coordinates are based on the place of residence column.

    :param dataframe: pandas DataFrame to add the latitude and longitude columns to.
    :param longitude: a dict that holds places and their respective longitude coordinates.
    :return: pandas DataFrame.
    """
    lng_df = dataframe
    lng_dict = longitude

    for key, value in lng_dict.items():
        indices =  lng_df.loc[ lng_df["Местоживеене (населено място):"] == key].index
        lng_df.loc[indices, "longitude"] = value

    return lng_df

def plot_basemap_places(dataframe, latitude, longitude):
    """
    Plots a basemap of Bulgaria with the places where our participants are from.
    :param dataframe: pandas DataFrame
    :param latitude: dictionary with the latitude coordinates of the places.
    :param longitude: dictionary with long
    :return: void
    """
    df_to_plot = tidy_place_col(dataframe)
    df_to_plot = add_lng_col(df_to_plot, longitude)
    df_to_plot = add_lat_col(df_to_plot, latitude)

    plt.figure(figsize=(12, 8))
    m = Basemap(
        projection="merc",
        llcrnrlat=41,
        urcrnrlat=45,
        llcrnrlon=22,
        urcrnrlon=30)
    x, y = m(df_to_plot.longitude, df_to_plot.latitude)
    m.fillcontinents(color="lightgreen", lake_color="aqua")
    m.drawmapboundary(fill_color='aqua')
    m.drawcountries(linewidth=1)
    m.drawcoastlines()

    plt.plot(x, y, "o", markersize=10, color="crimson", alpha=0.2)
    plt.title("Republic of Bulgaria", fontsize=20)
    plt.xlabel("Latitude", fontsize=14)
    plt.ylabel("Longitude", fontsize=14)
    fig, ax = plt.subplots(figsize=(12, 1))
    fig.subplots_adjust()

    norm = mpl.colors.Normalize(vmin=0, vmax=100)

    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="Reds"),
                      cax=ax, orientation='horizontal')
    cb.set_label(label='Respondents Density', size='x-large')
    plt.show()


def plot_basemap_places_labels(dataframe, latitude, longitude):
    """
    Plots a basemap of Bulgaria with the places where our participants are from.
    :param dataframe: pandas DataFrame
    :param latitude: dictionary with the latitude coordinates of the places.
    :param longitude: dictionary with long
    :return: void
    """
    df_to_plot = tidy_place_col(dataframe)
    df_to_plot = add_lng_col(df_to_plot, longitude)
    df_to_plot = add_lat_col(df_to_plot, latitude)

    places = df_to_plot["Местоживеене (населено място):"].value_counts().index
    # places_values = df_to_plot["Местоживеене (населено място):"].value_counts().values

    plt.figure(figsize=(12, 10))
    m = Basemap(
        projection="merc",
        llcrnrlat=41,
        urcrnrlat=45,
        llcrnrlon=22,
        urcrnrlon=30)
    x, y = m(df_to_plot.longitude.value_counts().index, df_to_plot.latitude.value_counts().index)
    m.fillcontinents(color="lightgreen", lake_color="aqua")
    m.drawmapboundary(fill_color='aqua')
    m.drawcountries(linewidth=1)
    m.drawcoastlines()
    plt.scatter(x, y, s=100, color="red")  # alpha=0.2)
    for i, label in enumerate(places):
        plt.annotate(f"{label}", (x[i], y[i]), fontsize="large", weight="bold")

    plt.show()