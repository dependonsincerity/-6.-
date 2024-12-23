import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

file_path = "for 6(dataset).csv"
data = pd.read_csv(file_path)

# 2. Анализ набора данных
def analyze_memory(data, label):
    memory_usage = data.memory_usage(deep=True)
    total_memory = memory_usage.sum() / (1024 * 1024)  
    column_stats = []
    for col in data.columns:
        col_memory = memory_usage[col] / (1024 * 1024)  
        col_fraction = col_memory / total_memory
        col_type = data[col].dtype
        column_stats.append({
            "Колонка": col,
            "Объем памяти (МБ)": col_memory, 
            "Доля от общего объема": col_fraction,
            "Тип данных": str(col_type)
        })
    analysis = {
        "Метка": label,
        "Общий объем памяти (МБ)": total_memory,  
        "Статистика по колонкам": sorted(column_stats, key=lambda x: x["Объем памяти (МБ)"], reverse=True)
    }
    return analysis

# 3. Сохраняем данные в JSON с учетом преобразования типов
def serialize_to_json(data, filename):
    def convert_types(obj):
        if isinstance(obj, (np.int64, np.float64)):
            return obj.item()
        raise TypeError(f"Тип {type(obj)} не сериализуется в JSON")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=convert_types)

original_analysis = analyze_memory(data, "Оригинальный набор данных")
serialize_to_json(original_analysis, "memory_stats(изначальный).json")

# 4. Оптимизация данных: уменьшаем количество преобразований
def optimize_dataframe(df):
    df_optimized = df.copy()
    
    # Преобразуем object в category, если уникальных значений < 50% от количества строк
    for col in df_optimized.select_dtypes(include=["object"]):
        if df_optimized[col].nunique() / len(df_optimized) < 0.5:
            df_optimized[col] = df_optimized[col].astype("category")
    
    # Оптимизируем только те числовые столбцы, которые реально можно уменьшить
    for col in df_optimized.select_dtypes(include=["int", "float"]):
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="integer" if col in df_optimized.select_dtypes(include=["int"]).columns else "float")
    
    return df_optimized

optimized_data = optimize_dataframe(data)
optimized_analysis = analyze_memory(optimized_data, "Оптимизированный набор данных")
serialize_to_json(optimized_analysis, "memory_stats(оптимизированный).json")

# 5. Сравнение объемов памяти
original_memory_mb = original_analysis['Общий объем памяти (МБ)']
optimized_memory_mb = optimized_analysis['Общий объем памяти (МБ)']

print(f"Снижение объема памяти: {original_memory_mb:.2f} МБ -> {optimized_memory_mb:.2f} МБ")

# 6. Работа с поднабором данных:
chunk_size = 5000
selected_columns = list(data.columns[:10])
new_file = "reworked_file(6).csv"

# Сохраняем поднабор данных.
chunk_iterator = pd.read_csv(file_path, usecols=selected_columns, chunksize=chunk_size)
for i, chunk in enumerate(chunk_iterator):
    chunk.to_csv(new_file, mode="a", index=False, header=(i == 0))

# 7. Построение графиков
reworked_data = pd.read_csv(new_file)
subset_data = reworked_data.head(1000)

# Линейный график
plt.figure(figsize=(10, 6))
plt.plot(subset_data.index[:100], subset_data.iloc[:100, 0], marker="o")
plt.title("Линейный график. Отображение результата по первой колонке.")
plt.xlabel("Индекс")
plt.ylabel("Значение")
plt.savefig("Линейный.png")

# Столбчатый график
plt.figure(figsize=(10, 6))
subset_data.iloc[:10, 1].value_counts().plot(kind="bar")
plt.title("Столбчатый график. Частота категорий во второй колонке.")
plt.xlabel("Категории")
plt.ylabel("Частота")
plt.savefig("Столбчатый.png")

# Круговая диаграмма
plt.figure(figsize=(10, 6))
subset_data.iloc[:10, 1].value_counts().plot(kind="pie", autopct='%1.1f%%')
plt.title("Круговая диаграмма. Распределение категорий.")
plt.ylabel('') 
plt.savefig("Круговая.png")

# Гистограмма
plt.figure(figsize=(10, 6))
subset_data.iloc[:10, 4].hist(bins=20)
plt.title("Гистограмма. Распределение значений по пятой колонке")
plt.xlabel("Значения")
plt.ylabel("Частота")
plt.savefig("Гистограмма.png")

# Pairplot (Парный график для визуализации зависимостей между признаками)
plt.figure(figsize=(10, 6))
sns.pairplot(subset_data.iloc[:, :4])
plt.suptitle("Парный график. Визуализация зависимостей между признаками", y=1.001)
plt.savefig("Парный график.png")

print("Анализ данных завершен. Все файлы сохранены.")
