#Import libraries:
import os
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Coder 1:
def load_and_preprocess_data(mat_path=r'D:\Linear Algebra\Project 7/users_movies.mat'):
    """
    Task 1, 2, 3: Load dữ liệu MATLAB, xử lý 0-indexing và lọc người dùng hợp lệ.
    """
    # Task 1: Load file .mat
    data = sio.loadmat(mat_path)
    movies = data['movies'].flatten()                     # 3952 tên phim
    users_movies = data['users_movies']                   # Ma trận (6040, 3952)
    users_movies_sort = data['users_movies_sort']         # Ma trận (6040, 20)
    index_small = data['index_small'].flatten() - 1       # Chuyển về 0-indexed cho Python
    trial_user = data['trial_user'].flatten()             # Vectơ 20 phần tử

    # Task 2: In danh sách 20 bộ phim phổ biến nhất
    print("=" * 50)
    print("DANH SÁCH 20 BỘ PHIM PHỔ BIẾN NHẤT (index_small):")
    print("=" * 50)
    for j, idx in enumerate(index_small):
        title = movies[idx][0] if isinstance(movies[idx], np.ndarray) else str(movies[idx])
        print(f"{j+1:2d}. {title}")
    print("\n")

    # Task 3: Lọc người dùng đã đánh giá đủ 20 phim (không chứa số 0)
    # Lệnh ratings = [] tương đương với việc chọn lọc các dòng thỏa mãn tích các phần tử != 0
    valid_mask = np.all(users_movies_sort != 0, axis=1)
    ratings = users_movies_sort[valid_mask]  # Ma trận ratings chỉ giữ lại các dòng hợp lệ

    print(f"Tổng số người dùng ban đầu: {users_movies_sort.shape[0]}")
    print(f"Số người dùng đã đánh giá đủ 20 phim (kích thước ratings): {ratings.shape[0]}")

    return {
        'movies': movies,
        'users_movies': users_movies,
        'users_movies_sort': users_movies_sort,
        'index_small': index_small,
        'trial_user': trial_user,
        'ratings': ratings,
        'valid_user_indices': np.where(valid_mask)[0]  # Lưu vị trí dòng gốc trong users_movies
    }

    #Coder 2: 
def calculate_euclidean_distance(ratings, user_vector):
    """
    Task 4 & 5: Tính khoảng cách Euclidean giữa user_vector và các dòng trong ratings.
    """
    # Task 4: Tính l2-norm ||ratings[i] - user_vector||
    eucl = np.linalg.norm(ratings - user_vector, axis=1)
    
    # Task 5: Sắp xếp tăng dần tìm người có khoảng cách nhỏ nhất
    dist_index = np.argsort(eucl)
    closest_user_dist = dist_index[0]  # Vị trí trong ma trận ratings
    
    return eucl, dist_index, closest_user_dist


def calculate_pearson_correlation(ratings, user_vector):
    """
    Task 6, 7 & 8: Trừ trung bình (Centering Data) và tính Hệ số tương quan Pearson.
    """
    # Task 6: Chuẩn hóa về tâm (ratings_cent và trial_user_cent)
    ratings_mean = ratings.mean(axis=1, keepdims=True)
    ratings_cent = ratings - ratings_mean
    
    user_mean = user_vector.mean()
    user_cent = user_vector - user_mean
    
    # Task 7: Tính Pearson r = cos(theta) thông qua tích vô hướng
    numerator = np.dot(ratings_cent, user_cent)
    denominator = np.linalg.norm(ratings_cent, axis=1) * np.linalg.norm(user_cent)
    
    # Chống lỗi chia cho 0 nếu có
    pearson = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    
    # Task 8: Sắp xếp giảm dần tìm người có tương quan cao nhất
    pearson_index = np.argsort(pearson)[::-1]
    closest_user_pearson = pearson_index[0]
    
    return pearson, pearson_index, closest_user_pearson

#Coder 3:
def get_movie_recommendations(user_vector, data_dict, closest_dist_idx, closest_pearson_idx):
    """
    Task 10 & 11: Lọc ra các bộ phim thích (5 sao) và gợi ý phim từ 2 phương pháp.
    """
    users_movies = data_dict['users_movies']
    movies = data_dict['movies']
    index_small = data_dict['index_small']
    valid_user_indices = data_dict['valid_user_indices']

    # Quy đổi chỉ số trong 'ratings' về chỉ số dòng gốc trong 'users_movies'
    orig_dist_user = valid_user_indices[closest_dist_idx]
    orig_pearson_user = valid_user_indices[closest_pearson_idx]

    # Phim được 5 sao từ người dùng gần nhất theo Euclidean
    rec_dist_indices = np.where(users_movies[orig_dist_user, :] == 5)[0]
    
    # Phim được 5 sao từ người dùng tương đồng nhất theo Pearson
    rec_pearson_indices = np.where(users_movies[orig_pearson_user, :] == 5)[0]

    # Phim mà user hiện tại đã cho 5 sao trong 20 phim phổ biến
    liked_local_indices = np.where(user_vector == 5)[0]
    liked_movie_indices = index_small[liked_local_indices]

    def extract_titles(indices):
        titles = []
        for idx in indices:
            t = movies[idx]
            titles.append(t[0] if isinstance(t, np.ndarray) else str(t))
        return titles

    return {
        'liked_movies': extract_titles(liked_movie_indices),
        'recommend_dist': extract_titles(rec_dist_indices),
        'recommend_pearson': extract_titles(rec_pearson_indices)
    }


def run_team_pipeline(team_ratings_dict, data_dict, eucl_func, pearson_func):
    """
    Task 12 & 13: Thu thập myratings của 7 bạn, chạy pipeline tự động và xuất file CSV số liệu cho Nhóm Report.
    """
    records = []
    for member_name, user_vec in team_ratings_dict.items():
        eucl, _, c_dist = eucl_func(data_dict['ratings'], user_vec)
        pearson, _, c_pearson = pearson_func(data_dict['ratings'], user_vec)
        
        recs = get_movie_recommendations(user_vec, data_dict, c_dist, c_pearson)
        
        records.append({
            'Member': member_name,
            'Closest_Dist_User_ID': c_dist,
            'Min_Euclidean_Distance': round(float(eucl[c_dist]), 4),
            'Closest_Pearson_User_ID': c_pearson,
            'Max_Pearson_Corr': round(float(pearson[c_pearson]), 4),
            'Liked_Movies_Count': len(recs['liked_movies']),
            'Rec_Dist_Count': len(recs['recommend_dist']),
            'Rec_Pearson_Count': len(recs['recommend_pearson'])
        })
    
    df_results = pd.DataFrame(records)
    df_results.to_csv('experiment_results.csv', index=False)
    print("Đã xuất thành công file số liệu 'experiment_results.csv' cho Nhóm Report LaTeX!")
    return df_results

    #Coder 4: 
    ### Vẽ biểu đồ bằng Matplotlib & Seaborn:

def plot_and_save_distributions(eucl, pearson, save_path='distribution_plots.png'):
    """
    Vẽ đồ thị biểu diễn phân bố khoảng cách Euclidean và tương quan Pearson.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    sns.histplot(eucl, kde=True, ax=axes[0], color='navy', bins=20)
    axes[0].set_title('Euclidean Distance Distribution')
    axes[0].set_xlabel('Distance (d)')
    axes[0].set_ylabel('Frequency')
    
    sns.histplot(pearson, kde=True, ax=axes[1], color='crimson', bins=20)
    axes[1].set_title('Pearson Correlation Distribution')
    axes[1].set_xlabel('Correlation Coefficient (r)')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Đã lưu đồ thị phân bố tại file: {save_path}")
    plt.show()

#Run All Tasks:

if __name__ == '__main__':
    # 1. Chạy Coder 1
    data = load_and_preprocess_data()

    # 2. Chạy Coder 2 trên trial_user
    eucl, dist_idx, c_dist = calculate_euclidean_distance(data['ratings'], data['trial_user'])
    pearson, pearson_idx, c_pearson = calculate_pearson_correlation(data['ratings'], data['trial_user'])

    print("\n--- KẾT QUẢ THỬ NGHIỆM TÍNH TOÁN (TRIAL USER) ---")
    print(f"-> User gần nhất theo Euclidean: ID #{c_dist} (Khoảng cách d = {eucl[c_dist]:.4f})")
    print(f"-> User tương đồng nhất theo Pearson: ID #{c_pearson} (Hệ số r = {pearson[c_pearson]:.4f})")

    # 3. Chạy Coder 3
    recs = get_movie_recommendations(data['trial_user'], data, c_dist, c_pearson)
    print("\nCác phim trial_user đã đánh giá 5 sao:", recs['liked_movies'])
    print("Top 3 phim gợi ý theo Euclidean:", recs['recommend_dist'][:3])
    print("Top 3 phim gợi ý theo Pearson:", recs['recommend_pearson'][:3])

    # 4. Chạy Coder 4 (Vẽ đồ thị)
    plot_and_save_distributions(eucl, pearson)

    # 5. Chạy pipeline mẫu cho 7 thành viên
    dummy_team_ratings = {
        f'Thành viên {i+1}': np.random.randint(1, 6, 20) for i in range(7)
    }
    run_team_pipeline(dummy_team_ratings, data)