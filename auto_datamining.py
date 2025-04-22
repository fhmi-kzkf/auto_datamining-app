import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, silhouette_score, mean_absolute_error, r2_score, 
                             confusion_matrix, precision_score, recall_score, f1_score,
                             davies_bouldin_score, calinski_harabasz_score, mean_squared_error)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- Fungsi Pra-pemrosesan Otomatis ---
def preprocess_data(df):
    df = df.copy()
    # Isi nilai kosong numerik dengan mean
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)
    # Isi nilai kosong kategorikal dengan modus
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def encode_features(X):
    # One-hot encoding fitur kategorikal
    X_encoded = pd.get_dummies(X)
    return X_encoded

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# --- Fungsi Algoritma ---
def run_classification(X, y, tune=False):
    results = {}
    X = encode_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [3, 5, 10, None]}),
        'Naive Bayes': (GaussianNB(), {}),
        'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [3,5,7]}),
        'Support Vector Machine': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']})
    }

    for name, (model, params) in models.items():
        if tune and params:
            grid = GridSearchCV(model, params, cv=3, scoring='accuracy')
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            best_model = model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results[name] = {
            'model': best_model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'y_test': y_test,
            'y_pred': y_pred
        }
    return results

def run_clustering(X, n_clusters, tune=False):
    X_num = X.select_dtypes(include=['int64', 'float64'])
    X_scaled = scale_features(X_num)
    results = {}

    if tune:
        best_k = n_clusters
        best_score = -1
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k
        n_clusters = best_k

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)

    results['K-Means'] = {
        'silhouette_score': sil_score,
        'davies_bouldin_score': db_score,
        'calinski_harabasz_score': ch_score,
        'X_scaled': X_scaled,
        'labels': labels,
        'n_clusters': n_clusters
    }

    return results

def run_estimation(X, y, tune=False):
    X = encode_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    results = {}

    lr = LinearRegression().fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
    mape_lr = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100

    results['Linear Regression'] = {
        'MAE': mae_lr, 'R2': r2_lr, 'RMSE': rmse_lr, 'MAPE': mape_lr,
        'y_test': y_test, 'y_pred': y_pred_lr
    }

    svr = SVR()
    if tune:
        params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        grid = GridSearchCV(svr, params, cv=3, scoring='neg_mean_absolute_error')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
    else:
        best_model = svr.fit(X_train, y_train)

    y_pred_svr = best_model.predict(X_test)
    mae_svr = mean_absolute_error(y_test, y_pred_svr)
    r2_svr = r2_score(y_test, y_pred_svr)
    rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
    mape_svr = np.mean(np.abs((y_test - y_pred_svr) / y_test)) * 100

    results['Support Vector Regression'] = {
        'MAE': mae_svr, 'R2': r2_svr, 'RMSE': rmse_svr, 'MAPE': mape_svr,
        'y_test': y_test, 'y_pred': y_pred_svr
    }

    return results

def run_association(df, min_support, min_confidence):
    if df.dtypes.unique().tolist() == [bool]:
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        return rules
    else:
        return None

# --- Fungsi Visualisasi ---
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    st.pyplot(fig)

def plot_clusters(X_scaled, labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels

    fig, ax = plt.subplots()
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='Set2', data=df_pca, ax=ax)
    ax.set_title('Cluster Visualization (PCA 2D)')
    st.pyplot(fig)

def plot_regression_results(y_test, y_pred, title="Actual vs Predicted"):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    st.pyplot(fig)

# --- Sidebar Info dan Tutorial ---
def sidebar_info():
    st.sidebar.title("Instruksi / Tentang / About")
    menu = st.sidebar.radio("Pilih menu info:", ["Instruksi Penggunaan", "Tentang Algoritma/Metode", "Tutorial", "About"])

    if menu == "Instruksi Penggunaan":
        st.sidebar.markdown("""
        ### Cara Menggunakan Aplikasi Data Mining Ini

        1. **Upload File CSV atau Excel (.xlsx)**  
           Unggah dataset Anda dalam format CSV atau Excel.

        2. **Pilih Dataset Contoh (Opsional)**  
           Anda bisa pilih dataset contoh untuk mencoba aplikasi tanpa upload file.

        3. **Pilih Jenis Task**  
           - *Klasifikasi*: Mengkategorikan data ke kelas tertentu.  
           - *Klastering*: Mengelompokkan data tanpa label.  
           - *Estimasi/Forecasting*: Memprediksi nilai numerik.  
           - *Association Rule*: Menemukan aturan asosiasi dari data transaksi (format one-hot encoded).

        4. **Pilih Kolom Target (Jika Perlu)**  
           Untuk klasifikasi dan estimasi, pilih kolom target yang ingin diprediksi.

        5. **Atur Parameter (Jika Ada)**  
           Contoh: jumlah cluster untuk klastering, nilai minimum support dan confidence untuk association rule.

        6. **Aktifkan Tuning Hyperparameter (Opsional)**  
           Centang untuk mencari parameter terbaik (akan memakan waktu lebih lama).

        7. **Lihat Hasil dan Rekomendasi Algoritma Terbaik**  
           Aplikasi akan menjalankan beberapa algoritma dan menampilkan hasil evaluasi serta rekomendasi.

        8. **Lihat Visualisasi Hasil dan Statistik Dataset**  
           Visualisasi membantu memahami data dan performa model.

        **Catatan:**  
        - Pastikan data sudah bersih dan sesuai format.  
        - Untuk association rule mining, data harus dalam format one-hot encoded (kolom boolean).  
        """)

    elif menu == "Tentang Algoritma/Metode":
        st.sidebar.markdown("""
        ### Penjelasan Algoritma/Metode yang Digunakan

        **Klasifikasi:**  
        - *Decision Tree*: Pohon keputusan yang membagi data berdasarkan atribut.  
        - *Naive Bayes*: Probabilistik berdasarkan Teorema Bayes dengan asumsi independensi.  
        - *K-Nearest Neighbors (k-NN)*: Mengklasifikasikan berdasarkan kedekatan dengan data tetangga.  
        - *Support Vector Machine (SVM)*: Mencari hyperplane terbaik untuk memisahkan kelas.

        **Klastering:**  
        - *K-Means*: Mengelompokkan data ke dalam k cluster berdasarkan jarak ke centroid.

        **Estimasi/Forecasting:**  
        - *Linear Regression*: Model linier untuk memprediksi nilai numerik.  
        - *Support Vector Regression (SVR)*: Versi regresi dari SVM untuk prediksi nilai kontinu.

        **Association Rule Mining:**  
        - *Apriori*: Menemukan itemset yang sering muncul bersama dan aturan asosiasi dengan support dan confidence.

        """)

    elif menu == "Tutorial":
        st.sidebar.markdown("""
        ### Tutorial Singkat

        1. **Upload dataset Anda** atau pilih dataset contoh.  
        2. Pilih jenis analisis yang ingin dilakukan.  
        3. Jika perlu, pilih kolom target (untuk klasifikasi/estimasi).  
        4. Atur parameter seperti jumlah cluster atau support/confidence.  
        5. Aktifkan tuning hyperparameter jika ingin optimasi model.  
        6. Jalankan analisis dan lihat hasil serta visualisasi.  
        7. Gunakan sidebar untuk membaca instruksi dan penjelasan algoritma.

        **Dataset Contoh yang Tersedia:**  
        - Iris (klasifikasi)  
        - Mall Customers (klastering)  
        - Boston Housing (estimasi)  
        """)

    else:
        st.sidebar.markdown("""
        ### About

        Aplikasi ini dibuat menggunakan Python dan Streamlit untuk memudahkan eksplorasi data mining dengan berbagai algoritma populer.  
        Dikembangkan oleh AI Assistant.  
        Silakan gunakan dan kembangkan sesuai kebutuhan Anda!  
        """)

# --- Fungsi untuk load dataset contoh ---
@st.cache_data
def load_example_dataset(name):
    if name == "Iris (Klasifikasi)":
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        df = pd.read_csv(url)
    elif name == "Mall Customers (Klastering)":
        url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/cluster/mall_customers.csv"
        df = pd.read_csv(url, index_col=0)
    elif name == "Boston Housing (Estimasi)":
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        df = pd.read_csv(url)
    else:
        df = pd.DataFrame()
    return df

def main():
    sidebar_info()

    st.title("Tools Data Mining Otomatis dengan Python & Streamlit")

    # Pilihan dataset contoh
    example_datasets = ["-- Pilih Dataset Contoh --", "Iris (Klasifikasi)", "Mall Customers (Klastering)", "Boston Housing (Estimasi)"]
    example_choice = st.selectbox("Pilih dataset contoh (atau upload file Anda sendiri)", example_datasets)

    if example_choice != "-- Pilih Dataset Contoh --":
        df = load_example_dataset(example_choice)
        st.success(f"Dataset contoh '{example_choice}' berhasil dimuat.")
    else:
        uploaded_file = st.file_uploader("Upload file CSV atau Excel (.xlsx)", type=["csv", "xlsx"])
        df = None
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"File '{uploaded_file.name}' berhasil diupload.")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    if df is not None and not df.empty:
        df = preprocess_data(df)
        st.write("Preview dataset:")
        st.dataframe(df.head())

        st.subheader("Statistik Deskriptif Dataset")
        st.write(df.describe(include='all'))

        task = st.selectbox("Pilih jenis task", ["Klasifikasi", "Klastering", "Estimasi/Forecasting", "Association Rule"])

        tune = False
        if task in ["Klasifikasi", "Klastering", "Estimasi/Forecasting"]:
            tune = st.checkbox("Aktifkan Tuning Hyperparameter (akan memakan waktu lebih lama)")

        if task == "Klasifikasi":
            target = st.selectbox("Pilih kolom target (kelas)", df.columns)
            X = df.drop(columns=[target])
            y = df[target]
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            results = run_classification(X, y, tune=tune)
            st.write("Evaluasi tiap algoritma klasifikasi:")
            for algo, res in results.items():
                st.write(f"### {algo}")
                st.write(f"- Accuracy: {res['accuracy']:.4f}")
                st.write(f"- Precision: {res['precision']:.4f}")
                st.write(f"- Recall: {res['recall']:.4f}")
                st.write(f"- F1-Score: {res['f1_score']:.4f}")
                plot_confusion_matrix(res['y_test'], res['y_pred'], title=f"Confusion Matrix - {algo}")

            best_algo = max(results, key=lambda x: results[x]['accuracy'])
            st.success(f"Algoritma terbaik: {best_algo} (Accuracy: {results[best_algo]['accuracy']:.4f})")

            st.subheader("Distribusi Kelas Target")
            fig, ax = plt.subplots()
            sns.countplot(x=y, ax=ax)
            ax.set_xlabel("Kelas")
            ax.set_ylabel("Jumlah")
            st.pyplot(fig)

        elif task == "Klastering":
            X = df.select_dtypes(include=['int64', 'float64'])
            if X.shape[1] == 0:
                st.error("Dataset tidak memiliki fitur numerik untuk klastering.")
                return
            n_clusters = st.slider("Jumlah cluster (k)", 2, 10, 3)
            results = run_clustering(X, n_clusters, tune=tune)
            for algo, res in results.items():
                st.write(f"### {algo}")
                st.write(f"- Silhouette Score: {res['silhouette_score']:.4f}")
                st.write(f"- Davies-Bouldin Index: {res['davies_bouldin_score']:.4f}")
                st.write(f"- Calinski-Harabasz Index: {res['calinski_harabasz_score']:.4f}")
                st.write(f"- Jumlah Cluster Terbaik: {res['n_clusters']}")
                plot_clusters(res['X_scaled'], res['labels'])

        elif task == "Estimasi/Forecasting":
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) < 2:
                st.error("Dataset harus memiliki minimal 2 kolom numerik untuk estimasi/forecasting.")
                return
            target = st.selectbox("Pilih kolom target (numerik)", numeric_cols)
            X = df.drop(columns=[target])
            y = df[target]

            results = run_estimation(X, y, tune=tune)
            for algo, res in results.items():
                st.write(f"### {algo}")
                st.write(f"- MAE: {res['MAE']:.4f}")
                st.write(f"- RMSE: {res['RMSE']:.4f}")
                st.write(f"- MAPE: {res['MAPE']:.2f}%")
                st.write(f"- R2 Score: {res['R2']:.4f}")
                plot_regression_results(res['y_test'], res['y_pred'], title=f"Actual vs Predicted - {algo}")

            best_algo = min(results, key=lambda x: results[x]['MAE'])
            st.success(f"Algoritma terbaik (MAE terkecil): {best_algo} (MAE: {results[best_algo]['MAE']:.4f})")

        elif task == "Association Rule":
            st.info("Pastikan data sudah dalam format one-hot encoded (kolom boolean).")
            min_support = st.slider("Min Support", 0.01, 0.5, 0.1)
            min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.5)
            rules = run_association(df, min_support, min_confidence)
            if rules is not None and not rules.empty:
                st.write("Aturan asosiasi ditemukan:")
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
            else:
                st.warning("Tidak ditemukan aturan asosiasi atau data tidak sesuai format one-hot encoded.")

if __name__ == "__main__":
    main()
