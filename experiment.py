from __future__ import annotations

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD


SEED = 42
N_USERS = 10_000
TRAIN_NEGATIVES = 10
EVAL_NEGATIVES = 99
HARD_NEGATIVE_POOL = 500
WINDOWS = [(1, "1d"), (3, "3d"), (7, "7d"), (14, "14d"), (30, "30d"),
           (60, "60d"), (90, "90d"), (180, "180d"), (365, "365d"),
           (None, "all")]


def make_prefix(day_idx: np.ndarray, entity_idx: np.ndarray, values: np.ndarray,
                n_days: int, n_entities: int) -> np.ndarray:
    daily = np.zeros((n_days, n_entities), dtype=np.float32)
    np.add.at(daily, (day_idx, entity_idx), values.astype(np.float32, copy=False))
    prefix = np.zeros((n_days + 1, n_entities), dtype=np.float32)
    np.cumsum(daily, axis=0, dtype=np.float32, out=prefix[1:])
    return prefix


def make_one_hot(values: pd.Series, prefix: str):
    values = values.fillna("unknown").astype(str)
    cats = sorted(values.unique())
    arr = np.column_stack([(values.to_numpy() == cat).astype(np.float32) for cat in cats])
    names = [f"{prefix}_{cat}" for cat in cats]
    return arr, names


def ndcg_recall_mrr(qidx, item_idx, labels, scores, k=10):
    order = np.lexsort((item_idx, -scores, qidx))
    sorted_q = qidx[order]
    sorted_y = labels[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_q[1:] != sorted_q[:-1]) + 1, len(order)]
    recalls, ndcgs, reciprocal_ranks, first_ranks = [], [], [], []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        pos = np.flatnonzero(sorted_y[a:b] == 1) + 1
        if len(pos):
            relevant_in_topk = pos[pos <= k]
            recalls.append(float(len(relevant_in_topk) / len(pos)))
            dcg = float(np.sum(1 / np.log2(relevant_in_topk + 1)))
            ideal_n = min(len(pos), k)
            idcg = float(np.sum(1 / np.log2(np.arange(1, ideal_n + 1) + 1)))
            ndcgs.append(dcg / idcg if idcg else 0.0)
            reciprocal_ranks.append(1.0 / int(pos[0]))
            first_ranks.append(int(pos[0]))
    ranks = np.asarray(first_ranks, dtype=np.int32)
    return {
        "n_queries": int(len(recalls)),
        "recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "median_positive_rank": float(np.median(ranks)) if len(ranks) else 0.0,
    }


def main():
    started = time.time()
    parser = argparse.ArgumentParser(description="Run the time-safe KION ranking experiment.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory containing interactions.csv.zip, items_en.csv and users_en.csv.")
    parser.add_argument("--output", type=Path, default=None, help="Where to write the JSON results.")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or (script_dir / "data")

    raw = pd.read_csv(data_dir / "interactions.csv.zip", compression="zip")
    raw["date"] = pd.to_datetime(raw["last_watch_dt"], errors="coerce")
    invalid_rows = int(raw["date"].isna().sum())
    full = raw.dropna(subset=["date"]).copy()
    full["watched_pct"] = pd.to_numeric(full["watched_pct"], errors="coerce")

    item_meta = pd.read_csv(
        data_dir / "items_en.csv",
        usecols=["item_id", "content_type", "release_year", "genres", "for_kids", "age_rating"],
    ).drop_duplicates("item_id").set_index("item_id")
    user_meta = pd.read_csv(
        data_dir / "users_en.csv", usecols=["user_id", "age", "income", "sex", "kids_flg"]
    ).drop_duplicates("user_id").set_index("user_id")

    dates = np.array(sorted(full["date"].unique()), dtype="datetime64[ns]")
    n_dates = len(dates)
    train_cut = int(0.70 * n_dates)
    val_cut = int(0.85 * n_dates)
    full["day_idx"] = pd.Index(dates).get_indexer(full["date"].to_numpy(dtype="datetime64[ns]")).astype(np.int32)
    item_ids = np.sort(full["item_id"].unique()).astype(np.int64)
    item_map = {int(x): i for i, x in enumerate(item_ids)}
    full["item_idx"] = full["item_id"].map(item_map).astype(np.int32)

    all_users = np.sort(full["user_id"].unique()).astype(np.int64)
    rng = np.random.default_rng(SEED)
    selected_users = np.sort(rng.choice(all_users, size=min(N_USERS, len(all_users)), replace=False))
    sample = full[full["user_id"].isin(selected_users)].copy().reset_index(drop=True)
    user_ids = np.sort(sample["user_id"].unique()).astype(np.int64)
    user_map = {int(x): i for i, x in enumerate(user_ids)}
    sample["user_idx"] = sample["user_id"].map(user_map).astype(np.int32)
    sample["item_idx"] = sample["item_id"].map(item_map).astype(np.int32)
    n_users, n_items = len(user_ids), len(item_ids)
    sample["pair_key"] = sample["user_idx"].astype(np.int64) * n_items + sample["item_idx"].astype(np.int64)
    pair_keys, pair_col = np.unique(sample["pair_key"].to_numpy(), return_inverse=True)


    full_days = full["day_idx"].to_numpy(np.int32)
    full_items = full["item_idx"].to_numpy(np.int32)
    valid_watch = full["watched_pct"].notna().to_numpy(np.float32)
    watch_values = full["watched_pct"].fillna(0).to_numpy(np.float32)
    item_prefix = make_prefix(full_days, full_items, np.ones(len(full), np.float32), n_dates, n_items)
    item_watch_sum_prefix = make_prefix(full_days, full_items, watch_values, n_dates, n_items)
    item_watch_n_prefix = make_prefix(full_days, full_items, valid_watch, n_dates, n_items)

    sample_days = sample["day_idx"].to_numpy(np.int32)
    sample_users = sample["user_idx"].to_numpy(np.int32)
    sample_items = sample["item_idx"].to_numpy(np.int32)
    sample_watch = sample["watched_pct"].fillna(0).to_numpy(np.float32)
    sample_watch_n = sample["watched_pct"].notna().to_numpy(np.float32)
    user_prefix = make_prefix(sample_days, sample_users, np.ones(len(sample), np.float32), n_dates, n_users)
    user_watch_sum_prefix = make_prefix(sample_days, sample_users, sample_watch, n_dates, n_users)
    user_watch_n_prefix = make_prefix(sample_days, sample_users, sample_watch_n, n_dates, n_users)
    pair_prefix = make_prefix(sample_days, pair_col.astype(np.int32), np.ones(len(sample), np.float32), n_dates, len(pair_keys))


    genre_lookup = {}
    for item_id in item_ids:
        value = item_meta["genres"].get(item_id, "")
        if pd.isna(value):
            genres = []
        else:
            genres = [g.strip().lower() for g in str(value).split(",") if g.strip()]
        genre_lookup[int(item_id)] = genres
    genres = sorted({g for values in genre_lookup.values() for g in values})
    genre_to_id = {g: i for i, g in enumerate(genres)}
    item_genres = []
    for item_id in item_ids:
        item_genres.append([genre_to_id[g] for g in genre_lookup[int(item_id)]])
    max_genres = max((len(g) for g in item_genres), default=1)
    item_genre_ids = np.full((n_items, max_genres), -1, dtype=np.int32)
    for i, values in enumerate(item_genres):
        item_genre_ids[i, :len(values)] = values
    n_genres = max(len(genres), 1)

    lengths = np.asarray([len(item_genres[i]) for i in sample_items], dtype=np.int32)
    expanded_rows = np.repeat(np.arange(len(sample), dtype=np.int32), lengths)
    expanded_genres = np.concatenate([np.asarray(item_genres[i], dtype=np.int32) for i in sample_items if item_genres[i]]) if lengths.sum() else np.empty(0, np.int32)
    genre_pair_keys = sample_users[expanded_rows].astype(np.int64) * n_genres + expanded_genres.astype(np.int64)
    unique_genre_keys, genre_col = np.unique(genre_pair_keys, return_inverse=True)
    genre_prefix = make_prefix(sample_days[expanded_rows], genre_col.astype(np.int32),
                               np.ones(len(expanded_rows), np.float32), n_dates, len(unique_genre_keys))


    item_static_parts, item_static_names = [], []
    content = item_meta["content_type"].reindex(item_ids).fillna("unknown").astype(str)
    arr, names = make_one_hot(content, "item_type")
    item_static_parts.append(arr); item_static_names += names
    for g, gid in genre_to_id.items():
        item_static_parts.append((item_genre_ids[:, :].__eq__(gid).any(axis=1)).astype(np.float32)[:, None])
        item_static_names.append(f"item_genre_{g.replace(' ', '_')}")
    year = pd.to_numeric(item_meta["release_year"].reindex(item_ids), errors="coerce").fillna(0).to_numpy(np.float32)
    rating = pd.to_numeric(item_meta["age_rating"].reindex(item_ids), errors="coerce").fillna(-1).to_numpy(np.float32)
    for name, values in [("item_release_year", year), ("item_age_rating", rating)]:
        item_static_parts.append(values[:, None]); item_static_names.append(name)
    item_kids = pd.to_numeric(item_meta["for_kids"].reindex(item_ids), errors="coerce").fillna(-1).to_numpy(np.float32)
    item_static_parts.append(item_kids[:, None]); item_static_names.append("item_for_kids")
    item_static = np.column_stack(item_static_parts).astype(np.float32)

    selected_meta = user_meta.reindex(user_ids)
    user_static_parts, user_static_names = [], []
    for col in ["age", "income", "sex"]:
        arr, names = make_one_hot(selected_meta[col], f"user_{col}")
        user_static_parts.append(arr); user_static_names += names
    kids = pd.to_numeric(selected_meta["kids_flg"], errors="coerce").fillna(-1).to_numpy(np.float32)
    user_static_parts.append(kids[:, None]); user_static_names.append("user_kids_flg")
    user_static_parts.append(selected_meta["age"].isna().to_numpy(np.float32)[:, None]); user_static_names.append("user_metadata_missing")
    user_static = np.column_stack(user_static_parts).astype(np.float32)


    svd_k = 16
    snapshots = []
    for snap_day in range(21, n_dates, 21):
        hist = sample[sample["day_idx"] < snap_day]
        if len(hist) < svd_k:
            continue
        matrix = sparse.coo_matrix(
            (np.ones(len(hist), np.float32),
             (hist["user_idx"].to_numpy(np.int32), hist["item_idx"].to_numpy(np.int32))),
            shape=(n_users, n_items), dtype=np.float32,
        ).tocsr()
        matrix.sum_duplicates()
        svd = TruncatedSVD(n_components=svd_k, n_iter=5, random_state=SEED)
        u_factors = svd.fit_transform(matrix).astype(np.float32)
        i_factors = svd.components_.T.astype(np.float32)
        snapshots.append((snap_day, u_factors, i_factors))


    target_events = sample[sample["watched_pct"].ge(50)]
    target = target_events.groupby(["user_idx", "day_idx"], as_index=False).agg(
        date=("date", "first"), positive_items=("item_idx", lambda x: tuple(sorted(set(x.tolist()))))
    ).reset_index(drop=True)
    target["split"] = np.where(target["day_idx"] < train_cut, "train",
                       np.where(target["day_idx"] < val_cut, "val", "test"))
    target["qid"] = np.arange(len(target), dtype=np.int32)
    q_user = target["user_idx"].to_numpy(np.int32)
    q_day = target["day_idx"].to_numpy(np.int32)
    q_dates = target["date"].to_numpy()
    q_split = target["split"].to_numpy()
    q_repeated = np.zeros(len(target), dtype=bool)
    q_repeated_positive_count = np.zeros(len(target), dtype=np.int32)
    for qi, positives in enumerate(target["positive_items"]):
        keys = q_user[qi] * n_items + np.asarray(positives, dtype=np.int64)
        positions = np.searchsorted(pair_keys, keys)
        repeated = pair_prefix[q_day[qi], positions] > 0
        q_repeated_positive_count[qi] = int(np.sum(repeated))
        q_repeated[qi] = bool(np.any(repeated))

    user_history = {}
    for user, group in sample.groupby("user_idx", sort=False):
        group = group.sort_values("day_idx")
        user_history[int(user)] = (group["day_idx"].to_numpy(np.int32), group["item_idx"].to_numpy(np.int32))
    candidate_q, candidate_item, candidate_y = [], [], []
    for qi in range(len(target)):
        user = int(q_user[qi])
        pos_items = np.asarray(target.at[qi, "positive_items"], dtype=np.int32)
        user_days, user_items = user_history[user]
        n_past = np.searchsorted(user_days, q_day[qi], side="left")
        seen_before = np.unique(user_items[:n_past])
        pool = np.setdiff1d(np.arange(n_items, dtype=np.int32), seen_before)
        pool = pool[~np.isin(pool, pos_items)]


        if len(pool) > HARD_NEGATIVE_POOL:
            popularity_before = item_prefix[int(q_day[qi]), pool]
            top = np.argpartition(popularity_before, -HARD_NEGATIVE_POOL)[-HARD_NEGATIVE_POOL:]
            pool = pool[top]
        n_neg = TRAIN_NEGATIVES if q_split[qi] == "train" else EVAL_NEGATIVES
        chosen = rng.choice(pool, size=min(n_neg, len(pool)), replace=False)
        candidate_q.extend([qi] * len(pos_items)); candidate_item.extend(pos_items.tolist()); candidate_y.extend([1] * len(pos_items))
        candidate_q.extend([qi] * len(chosen)); candidate_item.extend(chosen.tolist()); candidate_y.extend([0] * len(chosen))

    candidate_q = np.asarray(candidate_q, dtype=np.int32)
    candidate_item = np.asarray(candidate_item, dtype=np.int32)
    labels = np.asarray(candidate_y, dtype=np.int8)
    c_user = q_user[candidate_q]
    c_day = q_day[candidate_q]
    n_cand = len(candidate_q)
    start_idx_by_window = {}
    date_np = dates.astype("datetime64[D]")
    for days, label in WINDOWS:
        if days is None:
            start_idx_by_window[label] = np.zeros(n_cand, dtype=np.int32)
        else:
            boundary = date_np[c_day] - np.timedelta64(days, "D")
            start_idx_by_window[label] = np.searchsorted(date_np, boundary, side="left").astype(np.int32)

    features = {}
    def window_delta(prefix, entity, start, end=c_day):
        return prefix[end, entity] - prefix[start, entity]



    for days, label in WINDOWS:
        start = start_idx_by_window[label]
        item_count = window_delta(item_prefix, candidate_item, start)
        item_watch_sum = window_delta(item_watch_sum_prefix, candidate_item, start)
        item_watch_n = window_delta(item_watch_n_prefix, candidate_item, start)
        user_count = window_delta(user_prefix, c_user, start)
        user_watch_sum = window_delta(user_watch_sum_prefix, c_user, start)
        user_watch_n = window_delta(user_watch_n_prefix, c_user, start)
        features[f"item_pop_{label}"] = np.log1p(item_count).astype(np.float32)
        features[f"item_watch_mean_{label}"] = np.divide(item_watch_sum, item_watch_n,
            out=np.zeros_like(item_watch_sum), where=item_watch_n > 0).astype(np.float32)
        features[f"user_hist_{label}"] = np.log1p(user_count).astype(np.float32)
        features[f"user_watch_mean_{label}"] = np.divide(user_watch_sum, user_watch_n,
            out=np.zeros_like(user_watch_sum), where=user_watch_n > 0).astype(np.float32)

        pair_key = c_user.astype(np.int64) * n_items + candidate_item.astype(np.int64)
        ppos = np.searchsorted(pair_keys, pair_key)
        pvalid = (ppos < len(pair_keys))
        pvalid &= pair_keys[np.minimum(ppos, len(pair_keys) - 1)] == pair_key
        pair_count = np.zeros(n_cand, dtype=np.float32)
        pair_count[pvalid] = window_delta(pair_prefix, ppos[pvalid].astype(np.int32), start[pvalid], c_day[pvalid])
        features[f"pair_hist_{label}"] = np.log1p(pair_count).astype(np.float32)


    c_genres = item_genre_ids[candidate_item]
    gvalid = c_genres >= 0
    gkeys = c_user[:, None].astype(np.int64) * n_genres + np.maximum(c_genres, 0).astype(np.int64)
    gpos = np.searchsorted(unique_genre_keys, gkeys)
    gmatch = gvalid & (gpos < len(unique_genre_keys))
    gmatch &= unique_genre_keys[np.minimum(gpos, max(len(unique_genre_keys) - 1, 0))] == gkeys if len(unique_genre_keys) else False
    for label in ["90d", "all"]:
        start = start_idx_by_window[label]
        genre_count = np.zeros(gpos.shape, dtype=np.float32)
        if len(unique_genre_keys):
            rr, cc = np.where(gmatch)
            pos = gpos[rr, cc].astype(np.int32)
            genre_count[rr, cc] = genre_prefix[c_day[rr], pos] - genre_prefix[start[rr], pos]
        features[f"user_genre_count_{label}"] = np.log1p(genre_count.sum(axis=1)).astype(np.float32)


    for j, name in enumerate(item_static_names):
        features[name] = item_static[candidate_item, j]
    for j, name in enumerate(user_static_names):
        features[name] = user_static[c_user, j]
    features["target_month"] = pd.DatetimeIndex(q_dates[candidate_q]).month.to_numpy(np.float32)
    features["target_weekday"] = pd.DatetimeIndex(q_dates[candidate_q]).dayofweek.to_numpy(np.float32)


    snap_indices = np.asarray([s[0] for s in snapshots], dtype=np.int32)
    snap_pos = np.searchsorted(snap_indices, c_day, side="right") - 1 if len(snapshots) else np.full(n_cand, -1)
    if len(snapshots):
        factor_products = np.zeros((n_cand, svd_k), dtype=np.float32)
        dot = np.zeros(n_cand, dtype=np.float32)
        cosine = np.zeros(n_cand, dtype=np.float32)
        for si, (_, u_factor, i_factor) in enumerate(snapshots):
            mask = snap_pos == si
            if not mask.any():
                continue
            uv = u_factor[c_user[mask]]
            iv = i_factor[candidate_item[mask]]
            prod = uv * iv
            factor_products[mask] = prod
            dot[mask] = prod.sum(axis=1)
            denom = np.linalg.norm(uv, axis=1) * np.linalg.norm(iv, axis=1)
            cosine[mask] = np.divide(dot[mask], denom, out=np.zeros(mask.sum(), np.float32), where=denom > 0)
        for k in range(svd_k):
            features[f"svd_product_{k:02d}"] = factor_products[:, k]
        features["svd_dot"] = dot
        features["svd_cosine"] = cosine
    else:
        for k in range(svd_k):
            features[f"svd_product_{k:02d}"] = np.zeros(n_cand, np.float32)
        features["svd_dot"] = np.zeros(n_cand, np.float32)
        features["svd_cosine"] = np.zeros(n_cand, np.float32)

    X = pd.DataFrame(features).astype(np.float32, copy=False)
    item_scores = {name: X[f"item_pop_{name}"].to_numpy() for name in ["7d", "30d", "90d", "365d", "all"]}
    row_split = q_split[candidate_q]
    train_rows = row_split == "train"
    val_rows = row_split == "val"
    test_rows = row_split == "test"

    static_cols = item_static_names + user_static_names + ["target_month", "target_weekday"]
    manual_windows = ["7d", "30d", "90d", "365d", "all"]
    auto_windows = [label for _, label in WINDOWS]
    manual_cols = static_cols + [f"{stem}_{w}" for w in manual_windows for stem in
        ["item_pop", "item_watch_mean", "user_hist", "user_watch_mean", "pair_hist"]] + [
        "user_genre_count_90d", "user_genre_count_all"]
    auto_cols = static_cols + [f"{stem}_{w}" for w in auto_windows for stem in
        ["item_pop", "item_watch_mean", "user_hist", "user_watch_mean", "pair_hist"]] + [
        "user_genre_count_90d", "user_genre_count_all"]
    svd_cols = [c for c in X.columns if c.startswith("svd_")]
    variants = {"manual_GBDT": manual_cols, "auto_windows_GBDT": auto_cols,
                "auto_windows_plus_SVD_GBDT": auto_cols + svd_cols}
    model_params = dict(max_iter=120, max_leaf_nodes=31, learning_rate=0.08,
                        l2_regularization=1.0, early_stopping=False, random_state=SEED)
    fitted = {}
    validation = {}
    for name, cols in variants.items():
        model = HistGradientBoostingClassifier(**model_params)
        model.fit(X.loc[train_rows, cols], labels[train_rows])
        val_scores = model.predict_proba(X.loc[val_rows, cols])[:, 1]
        validation[name] = ndcg_recall_mrr(candidate_q[val_rows], candidate_item[val_rows],
                                           labels[val_rows], val_scores)
        fitted[name] = model


    val_pop = {}
    for name, scores in item_scores.items():
        val_pop[name] = ndcg_recall_mrr(candidate_q[val_rows], candidate_item[val_rows],
                                        labels[val_rows], scores[val_rows])
    best_pop_window = max(val_pop, key=lambda x: val_pop[x]["ndcg@10"])


    trainval_rows = train_rows | val_rows
    test_results = {}
    test_scores_by_model = {}
    for name, cols in variants.items():
        model = HistGradientBoostingClassifier(**model_params)
        model.fit(X.loc[trainval_rows, cols], labels[trainval_rows])
        test_scores = model.predict_proba(X.loc[test_rows, cols])[:, 1]
        test_scores_by_model[name] = test_scores
        test_results[name] = ndcg_recall_mrr(candidate_q[test_rows], candidate_item[test_rows],
                                             labels[test_rows], test_scores)
        fitted[name] = model
    test_results[f"popularity_{best_pop_window}_selected_on_val"] = ndcg_recall_mrr(
        candidate_q[test_rows], candidate_item[test_rows], labels[test_rows], item_scores[best_pop_window][test_rows])
    test_results["SVD_dot_only"] = ndcg_recall_mrr(candidate_q[test_rows], candidate_item[test_rows],
                                                   labels[test_rows], X["svd_dot"].to_numpy()[test_rows])
    auto_test_scores = test_scores_by_model["auto_windows_GBDT"]
    test_q = candidate_q[test_rows]
    test_item = candidate_item[test_rows]
    test_y = labels[test_rows]
    repeated_candidate = q_repeated[test_q]
    test_results["auto_windows_GBDT_repeat_targets"] = ndcg_recall_mrr(
        test_q[repeated_candidate], test_item[repeated_candidate], test_y[repeated_candidate],
        auto_test_scores[repeated_candidate])
    test_results["auto_windows_GBDT_first_time_targets"] = ndcg_recall_mrr(
        test_q[~repeated_candidate], test_item[~repeated_candidate], test_y[~repeated_candidate],
        auto_test_scores[~repeated_candidate])
    pop_test_scores = item_scores[best_pop_window][test_rows]
    test_results[f"popularity_{best_pop_window}_repeat_targets"] = ndcg_recall_mrr(
        test_q[repeated_candidate], test_item[repeated_candidate], test_y[repeated_candidate],
        pop_test_scores[repeated_candidate])
    test_results[f"popularity_{best_pop_window}_first_time_targets"] = ndcg_recall_mrr(
        test_q[~repeated_candidate], test_item[~repeated_candidate], test_y[~repeated_candidate],
        pop_test_scores[~repeated_candidate])

    result = {
        "data": {
            "raw_archive_rows": int(len(raw)), "complete_rows_used": int(len(full)),
            "invalid_truncated_rows_dropped": invalid_rows, "full_snapshot_users": int(full.user_id.nunique()),
            "full_snapshot_items": int(full.item_id.nunique()), "sample_users": int(n_users),
            "sample_history_rows": int(len(sample)), "date_start": str(pd.Timestamp(dates[0]).date()),
            "date_end": str(pd.Timestamp(dates[-1]).date()), "unique_dates": n_dates,
            "train_end": str(pd.Timestamp(dates[train_cut - 1]).date()),
            "validation_end": str(pd.Timestamp(dates[val_cut - 1]).date()),
            "queries_by_split": target.groupby("split").size().to_dict(),
            "positive_events_by_split": target_events.assign(split=np.where(target_events["day_idx"] < train_cut, "train",
                np.where(target_events["day_idx"] < val_cut, "val", "test"))).groupby("split").size().to_dict(),
            "test_unique_users": int(target.loc[target["split"].eq("test"), "user_idx"].nunique()),
            "test_repeated_positive_events": int(q_repeated_positive_count[q_split == "test"].sum()),
            "test_first_time_positive_events": int(target.loc[target["split"].eq("test"), "positive_items"].map(len).sum()
                                                     - q_repeated_positive_count[q_split == "test"].sum()),
            "test_multi_positive_queries": int((target.loc[target["split"].eq("test"), "positive_items"].map(len) > 1).sum()),
            "candidates_by_split": {s: int(np.sum(row_split == s)) for s in ["train", "val", "test"]},
            "target_definition": "one user-day query with all items watched at least 50% as relevant; hard negatives sampled uniformly from the 500 most popular items not seen before that date",
            "candidate_counts": {"train_negatives": TRAIN_NEGATIVES, "validation_negatives": EVAL_NEGATIVES,
                                 "test_negatives": EVAL_NEGATIVES, "hard_negative_pool": HARD_NEGATIVE_POOL},
        },
        "validation_models": validation,
        "validation_popularity_windows": val_pop,
        "selected_popularity_window": best_pop_window,
        "test": test_results,
        "runtime_seconds": round(time.time() - started, 2),
        "feature_counts": {name: len(cols) for name, cols in variants.items()},
        "model_params": model_params,
    }
    out = args.output or (script_dir / "experiment_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x))


if __name__ == "__main__":
    main()
