#!/usr/bin/env julia
#
# Bidirectional Active Processing (BAP) implementation.
# Reimplements the algorithm from Bidirectional Active Processing.md.
# Classifiers: Decision Tree, K-Nearest Neighbors, Support Vector Machine.
# Requires: ScikitLearn.jl (and Python with scikit-learn) for classifiers.
#

using CSV
using DataFrames
using Dates
using Statistics
using Random
using TOML
using ScikitLearn
using ScikitLearn: fit!, predict
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier
@sk_import svm: SVC
@sk_import metrics: accuracy_score

# --- Configuration ---

const DEFAULTS = Dict{String, Any}(
    "split" => [0.8, 0.2],
    "folds" => 5,
    "classifier" => "dt",
    "knn_k" => 3,
    "distance" => "euclidean",
    "t" => 0.95,
    "direction" => "forward",
    "splits" => 1,
    "n" => 10,
    "m" => 5,
    "sampling" => "stratified",
    "seed" => 42,
    "output_dir" => "results",
    "seed_modulus" => 2^31,
    "split_seed_multiplier" => 7919,
    "exp_seed_split_offset" => 10000,
    "exp_seed_iter_offset" => 1000,
)

struct Config
    train::String
    test::String
    testing::String  # "fixed" | "split" | "cv"
    split::Vector{Float64}
    folds::Int
    classifier::String  # "dt" | "knn" | "svm"
    parameters::Dict{String, Any}
    distance::String
    t::Float64
    direction::String  # "forward" | "backward"
    splits::Int
    n::Int
    m::Int
    sampling::String  # "random" | "stratified"
    seed::Int
    output_dir::String
end

function Config(; kwargs...)
    d = Dict{String, Any}(
        "train" => "", "test" => "", "testing" => "split",
        "split" => copy(get(DEFAULTS, "split", [0.8, 0.2])),
        "folds" => get(DEFAULTS, "folds", 5), "classifier" => get(DEFAULTS, "classifier", "dt"),
        "parameters" => Dict{String, Any}(), "distance" => get(DEFAULTS, "distance", "euclidean"),
        "t" => get(DEFAULTS, "t", 0.95), "direction" => get(DEFAULTS, "direction", "forward"),
        "splits" => get(DEFAULTS, "splits", 1), "n" => get(DEFAULTS, "n", 10),
        "m" => get(DEFAULTS, "m", 5), "sampling" => get(DEFAULTS, "sampling", "stratified"),
        "seed" => get(DEFAULTS, "seed", 42), "output_dir" => get(DEFAULTS, "output_dir", "results")
    )
    for (k, v) in kwargs
        d[string(k)] = v
    end
    Config(
        get(d, "train", ""), get(d, "test", ""), get(d, "testing", "split"),
        get(d, "split", copy(get(DEFAULTS, "split", [0.8, 0.2]))), get(d, "folds", DEFAULTS["folds"]),
        get(d, "classifier", DEFAULTS["classifier"]), get(d, "parameters", Dict{String, Any}()),
        get(d, "distance", DEFAULTS["distance"]), get(d, "t", DEFAULTS["t"]),
        get(d, "direction", DEFAULTS["direction"]), get(d, "splits", DEFAULTS["splits"]),
        get(d, "n", DEFAULTS["n"]), get(d, "m", DEFAULTS["m"]), get(d, "sampling", DEFAULTS["sampling"]),
        get(d, "seed", DEFAULTS["seed"]), get(d, "output_dir", DEFAULTS["output_dir"])
    )
end

function load_config_toml(path::String)::Config
    data = TOML.parsefile(path)
    params = get(data, "parameters", Dict())
    params = Dict(string(k) => v for (k, v) in params)

    testing = "split"
    test_path = ""
    split_ratios = copy(get(DEFAULTS, "split", [0.8, 0.2]))
    folds = DEFAULTS["folds"]
    if haskey(data, "testing")
        t = data["testing"]
        if t isa Dict
            if haskey(t, "fixed")
                testing = "fixed"
                fix = t["fixed"]
                test_path = string(fix isa Dict ? get(fix, "test", get(data, "test", "")) : get(data, "test", ""))
            elseif haskey(t, "split")
                testing = "split"
                s = get(t, "split", DEFAULTS["split"])
                split_ratios = Float64.(collect(s))
            elseif haskey(t, "cv")
                testing = "cv"
                c = t["cv"]
                folds = Int(c isa Dict ? get(c, "folds", DEFAULTS["folds"]) : get(t, "folds", DEFAULTS["folds"]))
            end
        end
    end

    direction = "forward"
    if haskey(data, "direction")
        d = data["direction"]
        if d isa Dict && haskey(d, "backward")
            direction = "backward"
        end
    end

    sampling = "random"
    if haskey(data, "sampling")
        s = data["sampling"]
        if s isa Dict && haskey(s, "stratified")
            sampling = "stratified"
        end
    end

    goal_t = get(DEFAULTS, "t", 0.95)
    if haskey(data, "goal") && data["goal"] isa Dict && haskey(data["goal"], "t")
        goal_t = Float64(data["goal"]["t"])
    end

    Config(
        string(get(data, "train", "")),
        test_path,
        testing,
        split_ratios,
        folds,
        lowercase(string(get(data, "classifier", DEFAULTS["classifier"]))),
        params,
        string(get(data, "distance", DEFAULTS["distance"])),
        goal_t,
        direction,
        Int(get(data, "splits", DEFAULTS["splits"])),
        Int(get(data, "n", DEFAULTS["n"])),
        Int(get(data, "m", DEFAULTS["m"])),
        sampling,
        Int(get(data, "seed", DEFAULTS["seed"])),
        string(get(data, "output_dir", DEFAULTS["output_dir"]))
    )
end

# --- Data loading ---

function detect_class_column(names_vec)::Symbol
    for nm in names_vec
        lowercase(string(nm)) in ("class", "label", "target") && return nm isa Symbol ? nm : Symbol(nm)
    end
    error("No 'class', 'label', or 'target' column found (case-insensitive)")
end

function load_csv(path::String)::Tuple{DataFrame, Vector{String}}
    df = CSV.read(path, DataFrame)
    names_vec = propertynames(df)
    class_col = detect_class_column(names_vec)
    X = select(df, Not(class_col))
    y = string.(df[:, class_col])
    # Fill NaN with column mean
    for nm in names(X)
        col = X[!, nm]
        if any(ismissing, col)
            X[!, nm] = coalesce.(col, mean(skipmissing(col)))
        end
    end
    return X, y
end

function normalize_features(X_train::DataFrame, X_test::DataFrame)::Tuple{DataFrame, DataFrame}
    mn = [mean(collect(skipmissing(X_train[!, c]))) for c in names(X_train)]
    mx = [maximum(collect(skipmissing(X_train[!, c]))) for c in names(X_train)]
    for (i, c) in enumerate(names(X_train))
        rng = mx[i] - mn[i]
        rng = rng == 0 ? 1.0 : rng
        X_train[!, c] = (X_train[!, c] .- mn[i]) ./ rng
        X_test[!, c] = (X_test[!, c] .- mn[i]) ./ rng
    end
    return X_train, X_test
end

function stratified_kfold_split(X::DataFrame, y::Vector{String}, n_folds::Int, seed::Int, fold_index::Int)
    rng = MersenneTwister(seed)
    classes = unique(y)
    all_train = Int[]
    all_test = Int[]
    for c in classes
        idx = findall(==(c), y)
        n = length(idx)
        perm = randperm(rng, n)
        fold_size = max(1, div(n, n_folds))
        start_test = fold_index * fold_size + 1
        end_test = min((fold_index + 1) * fold_size, n)
        test_idx = idx[perm[start_test:end_test]]
        train_idx = setdiff(idx, test_idx)
        append!(all_train, train_idx)
        append!(all_test, test_idx)
    end
    shuffle!(rng, all_train)
    shuffle!(rng, all_test)
    X_train = X[all_train, :]
    X_test = X[all_test, :]
    y_train = y[all_train]
    y_test = y[all_test]
    return X_train, y_train, X_test, y_test
end

function stratified_split(X::DataFrame, y::Vector{String}, train_ratio::Float64, seed::Int)
    rng = MersenneTwister(seed)
    classes = unique(y)
    train_idx = Int[]
    test_idx = Int[]
    for c in classes
        idx = findall(==(c), y)
        n_train = max(1, min(length(idx) - 1, Int(round(train_ratio * length(idx)))))
        perm = shuffle!(rng, copy(idx))
        append!(train_idx, perm[1:n_train])
        append!(test_idx, perm[n_train+1:end])
    end
    shuffle!(rng, train_idx)
    shuffle!(rng, test_idx)
    X_train = X[train_idx, :]
    X_test = X[test_idx, :]
    y_train = y[train_idx]
    y_test = y[test_idx]
    return X_train, y_train, X_test, y_test
end

function load_data(config::Config, split_seed::Int, fold_index::Union{Int,Nothing}=nothing)::Tuple{DataFrame, Vector{String}, DataFrame, Vector{String}}
    if config.testing == "fixed" && config.test != ""
        X_train, y_train = load_csv(config.train)
        X_test, y_test = load_csv(config.test)
    elseif config.testing == "cv" && fold_index !== nothing
        X, y = load_csv(config.train)
        X_train, y_train, X_test, y_test = stratified_kfold_split(X, y, config.folds, split_seed, fold_index)
    else
        X, y = load_csv(config.train)
        tr = config.split[1]
        te = length(config.split) > 1 ? config.split[2] : 1.0 - tr
        X_train, y_train, X_test, y_test = stratified_split(X, y, tr, split_seed)
    end
    X_train, X_test = normalize_features(X_train, X_test)
    return X_train, y_train, X_test, y_test
end

# --- Classifiers ---

function make_classifier(name::String, seed::Int, distance::String, params::Dict)
    k = get(params, "k", get(params, "n_neighbors", DEFAULTS["knn_k"]))
    if name == "dt"
        return DecisionTreeClassifier(random_state=seed)
    elseif name == "knn"
        return KNeighborsClassifier(n_neighbors=k, metric=distance)
    elseif name == "svm"
        return SVC(kernel="rbf", random_state=seed)
    else
        error("Unknown classifier: $name. Use dt, knn, or svm.")
    end
end

# --- Sampling ---

function sample_indices(available::Vector{Int}, y::Vector{String}, n::Int, method::String, seed::Int)::Vector{Int}
    if length(available) <= n
        return copy(available)
    end
    rng = MersenneTwister(seed)
    if method == "stratified"
        avail_y = y[available]
        classes = unique(avail_y)
        if length(classes) < 2 || n < length(classes)
            return available[randperm(rng, length(available))[1:n]]
        end
        # Proportional stratified sample
        selected = Int[]
        by_class = Dict(c => Int[] for c in classes)
        for (i, yi) in zip(available, avail_y)
            push!(by_class[yi], i)
        end
        total = length(available)
        for c in classes
            pool = by_class[c]
            need = min(length(pool), max(1, Int(round(n * length(pool) / total))))
            append!(selected, pool[randperm(rng, length(pool))[1:need]])
        end
        shuffle!(rng, selected)
        return selected[1:min(n, length(selected))]
    end
    return available[randperm(rng, length(available))[1:n]]
end

# --- BAP Core ---

function run_single_iteration(
    exp_id::Int,
    X_train::DataFrame,
    y_train::Vector{String},
    X_test::DataFrame,
    y_test::Vector{String},
    config::Config,
    seed::Int,
    out_dir::String
)::Union{Dict, Nothing}
    rng = MersenneTwister(seed)
    clf_seed = rand(rng, 0:(DEFAULTS["seed_modulus"]-1))

    direction = config.direction
    m = config.m
    t = config.t
    classifier_name = config.classifier
    sampling = config.sampling
    n_train = nrow(X_train)

    case_indices = direction == "forward" ? Set{Int}() : Set(1:n_train)
    iteration = 0

    while true
        available_vec = collect(direction == "forward" ? (setdiff(Set(1:n_train), case_indices)) : case_indices)

        if direction == "forward"
            if isempty(available_vec)
                return nothing
            end
            take = min(m, length(available_vec))
        else
            if length(available_vec) < 2
                return nothing
            end
            take = min(m, length(available_vec) - 1)
            take < 1 && return nothing
        end

        selected = sample_indices(available_vec, y_train, take, sampling, seed + iteration)

        if direction == "forward"
            union!(case_indices, selected)
        else
            setdiff!(case_indices, selected)
        end

        idx_list = sort(collect(case_indices))
        X_sub = X_train[idx_list, :]
        y_sub = y_train[idx_list]

        if length(unique(y_sub)) < 2
            iteration += 1
            continue
        end

        clf = make_classifier(classifier_name, clf_seed + iteration, config.distance, config.parameters)
        fit!(clf, Matrix(X_sub), y_sub)
        y_pred = predict(clf, Matrix(X_test))
        accuracy = accuracy_score(y_test, y_pred)

        iteration += 1

        if accuracy >= t
            # Tabular export: feature columns (training order) then lowercase `class` (toolkit-style CSV).
            feats = propertynames(X_train)
            conv_df = DataFrame()
            for c in feats
                conv_df[!, c] = X_sub[!, c]
            end
            conv_df[!, :class] = y_sub
            fname = joinpath(out_dir, "converged_exp_$(exp_id)_seed$(seed).csv")
            CSV.write(fname, conv_df)
            n_train_total = nrow(X_train)
            dist = Dict{String, Int}()
            for c in string.(y_sub)
                dist[c] = get(dist, c, 0) + 1
            end
            pairs_sorted = sort([(k, v) for (k, v) in pairs(dist)])
            class_dist_str = join(["$k:$v" for (k, v) in pairs_sorted], ", ")
            return Dict(
                "exp_id" => exp_id, "seed" => seed,
                "cases_needed" => length(case_indices),
                "accuracy" => Float64(accuracy),
                "iteration" => iteration,
                "cases_pct" => 100.0 * length(case_indices) / n_train_total,
                "total_train" => n_train_total,
                "class_dist" => class_dist_str
            )
        end

        if direction == "forward" && length(case_indices) >= n_train
            return nothing
        end
        if direction == "backward" && length(case_indices) < 2
            return nothing
        end
    end
    return nothing
end

function run_split(
    split_id::Int,
    X_train::DataFrame,
    y_train::Vector{String},
    X_test::DataFrame,
    y_test::Vector{String},
    config::Config,
    base_seed::Int,
    out_dir::String,
    n_iterations::Union{Int,Nothing}=nothing
)::Vector{Dict}
    n = n_iterations === nothing ? config.n : n_iterations
    results = Dict[]
    for i in 1:n
        exp_seed = (base_seed + split_id * DEFAULTS["exp_seed_split_offset"] + i * DEFAULTS["exp_seed_iter_offset"]) % Int64(DEFAULTS["seed_modulus"])
        r = run_single_iteration(i, X_train, y_train, X_test, y_test, config, exp_seed, out_dir)
        if r !== nothing
            r["split"] = split_id
            push!(results, r)
        end
    end
    return results
end

function compute_statistics(results::Vector{Dict}, total_runs::Int)::Vector{Dict}
    isempty(results) && return []
    cases = [x["cases_needed"] for x in results]
    accs = [x["accuracy"] for x in results]
    total_train = get(first(results), "total_train", 1)
    mean_cases = mean(cases)
    min_c = minimum(cases)
    max_c = maximum(cases)
    min_result = first(r for r in results if r["cases_needed"] == min_c)
    max_result = first(r for r in results if r["cases_needed"] == max_c)
    [Dict(
        "total_runs" => total_runs,
        "converged" => length(results),
        "convergence_rate" => length(results) / total_runs,
        "mean_cases" => mean_cases,
        "std_cases" => std(cases),
        "min_cases" => min_c,
        "min_cases_class_dist" => get(min_result, "class_dist", ""),
        "max_cases" => max_c,
        "max_cases_class_dist" => get(max_result, "class_dist", ""),
        "min_cases_pct" => minimum(x["cases_pct"] for x in results),
        "max_cases_pct" => maximum(x["cases_pct"] for x in results),
        "mean_accuracy" => mean(accs),
        "mean_sureness" => 1.0 - mean_cases / total_train
    )]
end

function save_config(config::Config, out_path::String)
    open(out_path, "w") do f
        println(f, "BAP Configuration")
        println(f, "=" ^ 40)
        println(f, "train: ", config.train)
        println(f, "testing: ", config.testing)
        if config.testing == "fixed"
            println(f, "test: ", config.test)
        elseif config.testing == "split"
            println(f, "split: ", config.split[1], ":", config.split[2])
        elseif config.testing == "cv"
            println(f, "folds: ", config.folds)
        end
        println(f, "classifier: ", config.classifier)
        println(f, "parameters: ", config.parameters)
        println(f, "distance: ", config.distance)
        println(f, "t: ", config.t)
        println(f, "direction: ", config.direction)
        println(f, "splits: ", config.splits)
        println(f, "n: ", config.n)
        println(f, "m: ", config.m)
        println(f, "sampling: ", config.sampling)
        println(f, "seed: ", config.seed)
        println(f, "output_dir: ", config.output_dir)
    end
end

function save_statistics(stats::Vector{Dict}, out_path::String)
    isempty(stats) && return
    df = DataFrame(stats)
    CSV.write(out_path, df)
end

# --- Main ---

function parse_args()::Config
    config_path = ""
    train_path = ""
    test_path = ""
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if (a == "-c" || a == "--config") && i + 1 <= length(ARGS)
            i += 1
            config_path = ARGS[i]
        elseif a == "--train" && i + 1 <= length(ARGS)
            i += 1
            train_path = ARGS[i]
        elseif a == "--test" && i + 1 <= length(ARGS)
            i += 1
            test_path = ARGS[i]
        end
        i += 1
    end

    if config_path != "" && isfile(config_path)
        cfg = load_config_toml(config_path)
        tr = train_path != "" ? train_path : cfg.train
        te = test_path != "" ? test_path : cfg.test
        return Config(tr, te, cfg.testing, cfg.split, cfg.folds, cfg.classifier,
            cfg.parameters, cfg.distance, cfg.t, cfg.direction, cfg.splits,
            cfg.n, cfg.m, cfg.sampling, cfg.seed, cfg.output_dir)
    end
    Config(train=train_path, test=test_path)
end

function main()
    config = parse_args()

    if config.train == "" || !isfile(config.train)
        @error "Missing or invalid --train path"
        exit(1)
    end
    if config.testing == "fixed" && (config.test == "" || !isfile(config.test))
        @error "testing=fixed requires valid --test path"
        exit(1)
    end
    if config.testing == "cv"
        config = Config(config.train, config.test, config.testing, config.split, config.folds,
            config.classifier, config.parameters, config.distance, config.t, config.direction,
            config.folds, config.n, config.m, config.sampling, config.seed, config.output_dir)
        # n is total; divide across folds so total_runs = n
        div_n, rem_n = divrem(config.n, config.folds)
        n_per_split = [div_n + (i <= rem_n ? 1 : 0) for i in 1:config.folds]
    else
        n_per_split = fill(config.n, config.splits)
    end

    Random.seed!(config.seed)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_root = joinpath(config.output_dir, "bap_$(timestamp)")
    mkpath(out_root)

    save_config(config, joinpath(out_root, "config.txt"))

    all_results = Dict[]
    total_runs = 0

    for split_id in 1:config.splits
        split_seed = (config.seed + split_id * DEFAULTS["split_seed_multiplier"]) % Int(DEFAULTS["seed_modulus"])
        fold_idx = config.testing == "cv" ? split_id - 1 : nothing
        X_train, y_train, X_test, y_test = load_data(config, split_seed, fold_idx)

        split_dir = joinpath(out_root, "split_$(split_id)")
        mkpath(split_dir)

        n_this_split = n_per_split[split_id]
        split_results = run_split(split_id, X_train, y_train, X_test, y_test, config, config.seed, split_dir, n_this_split)
        total_train = nrow(X_train)
        for r in split_results
            r["total_train"] = total_train
        end
        append!(all_results, split_results)
        total_runs += n_this_split
    end

    stats = compute_statistics(all_results, total_runs)
    if !isempty(stats)
        save_statistics(stats, joinpath(out_root, "statistics.csv"))
    end

    println("Results: $(length(all_results))/$total_runs converged -> $out_root")
    if !isempty(all_results) && !isempty(stats)
        cases = [r["cases_needed"] for r in all_results]
        println("  Mean cases: $(round(mean(cases); digits=1)) ± $(round(std(cases); digits=1))")
        s = stats[1]
        min_dist = get(s, "min_cases_class_dist", "")
        max_dist = get(s, "max_cases_class_dist", "")
        println("  Min: $(s["min_cases"])" * (isempty(min_dist) ? "" : " [$min_dist]"))
        println("  Max: $(s["max_cases"])" * (isempty(max_dist) ? "" : " [$max_dist]"))
    end
end

# Run when executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
