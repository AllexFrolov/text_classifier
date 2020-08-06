import os
import re
import time

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC

russian_stemmer = SnowballStemmer('russian')
nltk.download('stopwords')


def read_file(path: str) -> list:
    """
    Считывает данные из файла
    :param path: (str) Путь к файлу
    :return:
        result - (list) содержит списки [class, title, text]
    """
    result = []
    # Список для хранения Класса, Заголовка и Текста
    row = []
    with open(path, 'r') as f:
        for line in f.readlines():
            # Найдем имена классов
            if 'ClNames' in line:
                # Заменим ненужное обозначение
                names = line.replace('ClNames:', '', 1)
                # Разделим сроку и удалим лишние пробелы
                class_names = list(map(str.strip, names.split(';')))[:-1]
            # Найдем строчку с классом
            if 'CLASS' in line:
                # Извлечем номер класса из строчки
                index = int(line.replace('CLASS:', '', 1)) - 1
                # Добавим в лист имя класса
                row.append(class_names[index])
            # Найдем строчку с заголовком
            if 'Title' in line:
                row.append(line.replace('Title:', '', 1))
            # Найдем строчку с текстом
            if 'Abstract' in line:
                row.append(line.replace('Abstract:', '', 1))
                # Добавим сроку к результату и очистим ее
                result.append(row)
                row = []
    return result


def open_data(data_folder: str) -> pd.DataFrame:
    """
    Пытается прочитать данные из всех файлов в папке data_folder
    :param data_folder: (str) Путь к папке
    :return: (pd.DataFrame) датафрейм,
            состоящий из 'class_name', 'title', 'text' столбцов
    """
    # Получим список всех файлов в папке
    files = os.listdir(data_folder)
    all_data = []
    # Прочитаем каждый файл и добавим данные в общий список
    for file_name in files:
        data = read_file(data_folder + file_name)
        all_data += data
    return pd.DataFrame(all_data, columns=['class_name', 'title', 'text'])


def clearing(text: str) -> str:
    """
    Удаляет все кроме русских символов
    :param text: (str) Необработанный текст
    :return: (str) Чистый текст
    """
    # Воспользуемся регулярными выражениями для того чтобы отчистить текст
    result = ' '.join(re.sub(r'[^а-яА-ЯёЁ]', ' ', text).split())
    # Если в итоге получилась пустая строка, то выведем None
    if len(result) == 0:
        result = None
    return result


def stemming(text: str) -> str:
    """
        Стемминг каждого слова в строке
        :param text: (str) Необработанный текст
        :return: stemming_text - (str) Текст после стемминга
    """
    stemming_text = ' '.join(list(map(russian_stemmer.stem, text.split())))
    return stemming_text

# pd.set_option('display.max_columns', 3)
# pd.set_option('display.max_colwidth', 100)


def preprocess_data(data: pd.DataFrame):
    """
    Подготовка данных и разбиение на тренировочную и тестовую выборки
    :param data: (pandas.DataFrame) Необработанные данные
    :return: ftr_train, trg_train - (X, y) Тренировочные
            ftr_test, trg_test - (X, y) Тестовые
    """
    # Удалим дубликаты
    data.drop_duplicates(inplace=True)
    # Обновим индексы
    data.reset_index(inplace=True, drop=True)
    # Отчистим текст
    data['clear_text'] = data['text'].apply(clearing)
    np.random.seed(42)
    # Удалим строки с пропусками, если есть
    data = data.dropna().reset_index(drop=True)
    # Используем стемминг
    data['stem_text'] = data['clear_text'].apply(stemming)
    # Создадим столбец class
    data['class'] = OrdinalEncoder().fit_transform(data[['class_name']])
    # Разделим данные на target и features
    target = data['class']
    features = data['stem_text']
    # Разделим данные на тренировочную и тестовую выборки
    # Используем параметр stratify для равномерного распределения классов
    ftr_train, ftr_test, trg_train, trg_test = \
        train_test_split(features, target, test_size=0.2,
                         random_state=42, shuffle=True, stratify=target)
    # Векторизуем признаки, одновременно удалим стоп-слова
    stop_words = set(stopwords.words('russian'))
    vec = CountVectorizer(stop_words=stop_words)
    ftr_train = vec.fit_transform(ftr_train)
    ftr_test = vec.transform(ftr_test)
    return ftr_train, trg_train, ftr_test, trg_test


def find_parameters(estimator: object, parameters: dict,
                    train_data: tuple) -> dict:
    """
    Подбор параметров с помощью GridSearchCV
    :param estimator: (object) Модель
    :param parameters: (dict) Cловарь параметров с их значениями
    :param train_data: (tuple(X, y)) Данные для поиска параметров
    :return: (dict) Лучшие параметры
    """
    results = GridSearchCV(estimator, parameters,
                           scoring='accuracy', n_jobs=-1, cv=3)
    results.fit(train_data[0], train_data[1])
    print(f'Лучшая точность: {results.best_score_:.4f}')
    print(f'Лучшие параметры: {results.best_params_}')
    return results.best_params_


if __name__ == '__main__':
    # Считаем данные
    dataset = open_data('Data/')
    # Обработаем данные и раздедим их на выборки
    ftr_train, trg_train, ftr_test, trg_test = \
        preprocess_data(dataset)
    # Создадим и обучим модель k-nearest neighbors
    # Определим параметры и их границы для оптимизации модели
    print('K-nearest neighbors')
    params = {'n_neighbors': np.arange(1, 10),
              'weights': ['uniform', 'distance']}
    knn_model = KNeighborsClassifier(n_jobs=-1)
    # Получим лучшие параметры
    best_params = find_parameters(knn_model, params, (ftr_train, trg_train))
    # Установим параметры в модель
    knn_model.set_params(**best_params)
    # Обучим модель
    knn_model.fit(ftr_train, trg_train)
    # Получим значение точности на тестовых данных
    start_time = time.perf_counter()
    score = knn_model.score(ftr_test, trg_test)
    end_time = time.perf_counter()
    print(f'Время предсказания: {end_time - start_time:.4f}')
    print(f'Точность на тестовой выборке: {score:.4f}')
    print()

    # Создадим и обучим модель C-Support Vector Classification
    print('С-support vector')
    svc_model = SVC(random_state=42)
    params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    best_params = find_parameters(svc_model, params, (ftr_train, trg_train))
    svc_model.set_params(**best_params)
    svc_model.fit(ftr_train, trg_train)
    start_time = time.perf_counter()
    score = svc_model.score(ftr_test, trg_test)
    end_time = time.perf_counter()
    print(f'Время предсказания: {end_time - start_time:.4f}')
    print(f'Точность на тестовой выборке: {score:.4f}')
